import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from prepare_mscoco_dataset import MSCOCODataset
from prepare_vizwiz_dataset import VizWizDataset


from data_utils import get_loader_and_vocab
from nets import *


from tqdm import tqdm
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

torch.manual_seed(3)
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dt = MSCOCODataset()
# dt = VizWizDataset()

annotation_file = dt.val_captions
annotation_name = str(annotation_file.parts[-1][:-5])
coco = COCO(str(annotation_file))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(dt)
VOCABULARY_SIZE = vocab.__len__()
features, tokens = next(iter(train_loader))  
BATCH_SIZE, FEATURE_SIZE = features.shape
BATCH_SIZE, CAPTION_LENGTH = tokens.shape
HIDDEN_SIZE = 512
EMBED_SIZE = 75

def train_step(net, X, y, optimizer, criterion):
    batch_size, caption_length = y.shape
    batch_size, feature_size = X.shape
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    decoder_hidden = torch.zeros((3, batch_size, feature_size), device=DEVICE)
    decoder_carry = torch.zeros((3, batch_size, feature_size), device=DEVICE)
    decoder_init_state = (decoder_hidden, decoder_carry)
    for i in range(3):
        decoder_hidden[i] = X
    #decoder_hidden = X.unsqueeze(0)
    loss = 0.0
    optimizer.zero_grad()
    for i in range(caption_length-1):
        output, decoder_hidden = net.forward(y[:, i], decoder_init_state)
        loss += criterion(output, y[:, i+1])
    loss.backward()
    optimizer.step()
    return loss.item() / caption_length



def test_step(net, X, y, vocab):
    batch_size, feature_size = X.shape
    X = X.to(DEVICE)
    decoder_hidden = torch.zeros((3, batch_size, feature_size), device=DEVICE)
    decoder_carry = torch.zeros((3, batch_size, feature_size), device=DEVICE)
    decoder_init_state = (decoder_hidden, decoder_carry)
    for i in range(3):
        decoder_hidden[i] = X
    # decoder_hidden = X.unsqueeze(0)
    loss = 0.0
    stoi = vocab.get_stoi()
    
    start_token = stoi['boc']
    end_token = stoi['eoc']
    result = torch.zeros((batch_size, CAPTION_LENGTH), dtype=torch.long, device=DEVICE)
    result[:, 0] = start_token
    for i in range(CAPTION_LENGTH-1):
        output, decoder_hidden = net.forward(result[:, i], decoder_init_state)
        result[:, i+1] = torch.argmax(output, dim=1)
    
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




def gnmt_train_step(net, X, y, optimizer, criterion):

    batch_size, caption_length = y.shape
    batch_size, feature_size = X.shape
    decoder_hidden = net.initHidden(batch_size)
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    loss = 0.0
    optimizer.zero_grad()
    for i in range(caption_length-1):
        if i == 0:
            output, attn_input, hiddens = net.first_step(X, y[:, i], decoder_hidden)
            loss += criterion(output, y[:, i+1])
        else:
            output, attn_input, hiddens = net.forward(X, y[:, i], attn_input, hiddens)
            loss += criterion(output, y[:, i+1])
    loss.backward()
    optimizer.step()
    return loss.item() / caption_length

def train_model(net, vocab, epochs, comment):
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # optimizer = optim.Adam(net.parameters())
    criterion = nn.NLLLoss()
    writer = SummaryWriter(comment=comment)
    features, tokens = next(iter(train_loader))
    
    # features = features.unsqueeze(0)
    features = features.to(DEVICE)
    tokens = tokens.to(DEVICE)
    decoder_hidden = torch.zeros((3, BATCH_SIZE, FEATURE_SIZE), device=DEVICE)
    decoder_carry = torch.zeros((3, BATCH_SIZE, FEATURE_SIZE), device=DEVICE)
    decoder_init_state = (decoder_hidden, decoder_carry)
    for i in range(3):
        decoder_hidden[i] = features
    writer.add_graph(net, (tokens[:, 0], decoder_init_state))
    for epoch in tqdm(range(epochs)):

        net.train()
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            # X: features, y: tokens
            loss = train_step(net, X, y, optimizer, criterion)
            running_loss += loss
        print(running_loss / (i + 1))
        net.eval()
        data = []
        for X, y in val_loader:
            # X: features, y: ids
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

# net = GNMT(FEATURE_SIZE, HIDDEN_SIZE, VOCABULARY_SIZE, EMBED_SIZE, DEVICE).to(DEVICE)
net = DecoderLSTM(FEATURE_SIZE, VOCABULARY_SIZE, EMBED_SIZE, 3)
net = net.to(DEVICE)
COMMENT = f"_TEST_10_EPOCH_{dt.name}_3_layer_LSTM"

train_model(net, vocab, epochs=10, comment=COMMENT)


