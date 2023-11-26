import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nets import ICT
import numpy
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
#src = torch.randn((64, 196, 768))
#tgt = torch.randint(high=100, size=(64, 20))

vizwiz_dt = VizWizDataset()

train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(vizwiz_dt)

annotation_file = vizwiz_dt.val_captions
annotation_name = str(annotation_file.parts[-1][:-5])
coco = COCO(str(annotation_file))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

VOCABULARY_SIZE = vocab.__len__()

features, tokens = next(iter(train_loader))  
BATCH_SIZE, SEQ_LEN, FEATURE_SIZE = features.shape
BATCH_SIZE, CAPTION_LENGTH = tokens.shape

def train_step(net, X, y, optimizer, criterion):
    batch_size, caption_length = y.shape
    batch_size, seq_length, feature_size = X.shape
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    optimizer.zero_grad()
    tgt = y[:, :-1]
    tgt_y = y[:, 1:]
    output = net(X, tgt)
    loss = criterion(output.view(-1, VOCABULARY_SIZE), tgt_y.reshape(-1,))
    loss.backward()
    optimizer.step()
    return loss.item() / caption_length

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(memory.shape[0], 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (model.transformer.generate_square_subsequent_mask(ys.size(-1))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(-1)
        ys = torch.cat([ys, next_word], dim=-1)
    return ys

def test_step(net, X, y, vocab):
    batch_size, sequence_length, feature_size = X.shape
    X = X.to(DEVICE)
    X_mask = (torch.zeros(sequence_length, sequence_length)).type(torch.bool)
    stoi = vocab.get_stoi()
    start_token = stoi['boc']
    end_token = stoi['eoc']

    result = greedy_decode(
        net,  X, X_mask, max_len=17, start_symbol=start_token, end_symbol=end_token)


    captions = []
    itos = vocab.get_itos()
    for i in range(batch_size):
        caption = ""
        tokens = result[i, :]
        for token in tokens[1:]:
            if token == end_token:
                break
            caption += itos[token] + " "
        captions.append(caption)
    return captions, y


def train_model(net, vocab, epochs, comment):
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(comment=comment)
    features, tokens = next(iter(train_loader))
    
    # features = features.unsqueeze(0)
    net = net.to(DEVICE)
    features = features.to(DEVICE)
    tokens = tokens.to(DEVICE)
    #writer.add_graph(net, (features, tokens[:, :-1])) # does not work.
    for epoch in tqdm(range(epochs)):

        net.train()
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            # X: features, y: tokens
            loss = train_step(net, X, y, optimizer, criterion)
            running_loss += loss
            if i > 1000:
                break
        print(f"Loss: {running_loss / (i + 1)}")
        net.eval()
        data = []
        for X, y in val_loader:
            # X: features, y: ids 
            # TODO CHANGE X FROM (N X S X F) -> (S X N X F) BATCH SIZE SHOULD BE IN THE MIDDLE.
            with torch.no_grad():
                captions, ids = test_step(net, X, y, vocab)
            # print(f"id-{ids[0]}: {captions[0]}")
            for caption, id in zip(captions, ids):
                data.append({
                    "image_id": id.item(),
                    "caption" : caption
                })
            
            # raise NotImplementedError

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
        
            writer.add_scalar(f'{metric}', score, epoch)
        

        if test_loader is None:
            continue

        for X, y in test_loader:
            # X: features, y: ids
            with torch.no_grad():
                captions, ids = test_step(net, X, y, vocab)
            # raise NotImplementedError
            for caption, id in zip(captions, ids):
                data.append({
                    "image_id": id.item(),
                    "caption" : caption
                })
    writer.close()



model = ICT(vocab_len=vocab.__len__())
train_model(model, vocab=vocab, epochs=250, comment="test")
'''
for X, y in train_loader:
    tgt = y[:, :-1]
    tgt_y = y[:, 1:]
    pred = model(X, tgt)
'''