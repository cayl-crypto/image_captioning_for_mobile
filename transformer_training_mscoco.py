
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.nn import Transformer


from torch.utils.tensorboard import SummaryWriter

from prepare_mscoco_dataset import MSCOCODataset
from prepare_vizwiz_dataset import VizWizDataset

from data_utils import get_loader_and_vocab
import math
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

torch.manual_seed(3)
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#vizwiz_dt = VizWizDataset()
dt = MSCOCODataset()

train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(dt)

annotation_file = dt.val_captions
annotation_name = str(annotation_file.parts[-1][:-5])
coco = COCO(str(annotation_file))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

VOCABULARY_SIZE = vocab.__len__()
stoi = vocab.get_stoi()
start_token = stoi['boc']
end_token = stoi['eoc']
pad_token = stoi['pad']

features, tokens = next(iter(train_loader))  
BATCH_SIZE, SEQ_LEN, FEATURE_SIZE = features.shape
BATCH_SIZE, CAPTION_LENGTH = tokens.shape

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # src shape: seq_len, batch_size, feature_size e.g. for ViT: 196x128x768
        # src mask: seq_len, seq_len e.g. 196x196
        return self.transformer.encoder(self.positional_encoding(src
                            ), src_mask)
        #embedded = self.src_tok_emb(src)
        #return self.transformer.encoder(self.positional_encoding(embedded
        #                    ), src_mask)
        #return self.transformer.encoder(self.positional_encoding(
        #                   self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # tgt shape: seq_len, batch_size e.g. for ViT: 17x16
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = torch.zeros((src.shape[1], src.shape[0]), device=src.device).bool()
    tgt_padding_mask = (tgt == pad_token).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

TGT_VOCAB_SIZE = VOCABULARY_SIZE
EMB_SIZE = FEATURE_SIZE
NHEAD = 8
FFN_HID_DIM = FEATURE_SIZE
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 8

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token)

#optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer = torch.optim.SGD(transformer.parameters(), lr=0.01)
writer = SummaryWriter(comment=f"_____|NUM_ENCODER_LAYERS_{NUM_ENCODER_LAYERS}|NUM_DECODER_LAYERS_{NUM_DECODER_LAYERS}|Dataset_{dt.name}")

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for i, (src, tgt) in tqdm(enumerate(train_loader)):
        src = src.to(DEVICE)
        src = src.permute(1, 0, 2)
        tgt = tgt.to(DEVICE)
        tgt = tgt.permute(1, 0)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / (i+1)






def greedy_decode(model, src, src_mask, max_len, start_symbol):
    seq_len, batch_size, _ = src.size()  # Get the batch size from the input

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(0)  # Expand the next_word to have the same shape as ys

        ys = torch.cat([ys, next_word.type_as(ys.data)], dim=0)

    
    captions = []
    itos = vocab.get_itos()
    for i in range(batch_size):
        caption = ""
        tokens = ys[:, i]
        for token in tokens[1:]:
            if token == end_token:
                break
            caption += itos[token] + " "
        captions.append(caption)
    return captions

def test_epoch(model, best_score):
    model.eval()
    data = []
    with torch.no_grad():
        for i, (src, ids) in tqdm(enumerate(val_loader)):  # Assuming you have a DataLoader called test_loader
            src = src.to(DEVICE)
            src = src.permute(1, 0, 2)

            src_mask, _, src_padding_mask, _ = create_mask(src, src[0:1])
            
            captions = greedy_decode(model, src, src_mask, max_len=17, start_symbol=start_token) # Assuming you have a BOS_IDX for the beginning of sequence token

            for caption, id in zip(captions, ids):
                data.append({
                    "image_id": id.item(),
                    "caption" : caption
                })
    
    json_file = f"results/{annotation_name}_result.json"
    with open(json_file, "w") as file:
        json.dump(data, file)
    
    coco_result = coco.loadRes(json_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

        if metric == "CIDEr":
            if score > best_score:
                best_score = score
                json_file = f"results/best_{annotation_name}_result.json"
                with open(json_file, "w") as file:
                    json.dump(data, file)
                
                torch.save(model.cpu(), "best_mscoco_model.pt")
                model = model.to(DEVICE)
    
        writer.add_scalar(f'{metric}', score, epoch)
    return best_score

from timeit import default_timer as timer
NUM_EPOCHS = 100
BEST_CIDER_SCORE = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    writer.add_scalar("Train Loss", train_loss, epoch)
    end_time = timer()
    BEST_CIDER_SCORE = test_epoch(transformer, BEST_CIDER_SCORE)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    with open('best_cider_score_mscoco.txt', 'w') as file:
        file.write(f"Best CIDEr Score: {BEST_CIDER_SCORE}")

writer.close()