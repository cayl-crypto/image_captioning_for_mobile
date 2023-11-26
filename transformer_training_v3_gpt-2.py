
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.nn import Transformer

from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model = gpt2_model.to(DEVICE)

VOCABULARY_SIZE = vocab.__len__()
stoi = vocab.get_stoi()
start_token = stoi['boc']
end_token = stoi['eoc']
pad_token = stoi['pad']

features, tokens = next(iter(train_loader))  
BATCH_SIZE, SEQ_LEN, FEATURE_SIZE = features.shape
BATCH_SIZE, CAPTION_LENGTH = tokens.shape

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TGT_VOCAB_SIZE = VOCABULARY_SIZE
EMB_SIZE = FEATURE_SIZE
NHEAD = 8
FFN_HID_DIM = FEATURE_SIZE
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3



loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token)

optimizer = torch.optim.Adam(gpt2_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for i, (src, tgt) in tqdm(enumerate(train_loader)):
        src = src.to(DEVICE)
        src = src.permute(1, 0, 2)
        tgt = tgt.to(DEVICE)
        tgt = tgt.permute(1, 0)

        input_ids = src
        labels = tgt

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / (i+1)





def test_epoch(model, best_score):
    model.eval()
    data = []
    with torch.no_grad():
        for i, (src, ids) in tqdm(enumerate(val_loader)):  # Assuming you have a DataLoader called test_loader
            src = src.to(DEVICE)
            src = src.permute(1, 0, 2)

            input_ids = src
                        # Generate text
            generated = model.generate(input_ids, max_length=17, do_sample=True)

            captions = [gpt2_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated]


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
    
     #   writer.add_scalar(f'{metric}', score, epoch)
    return best_score

from timeit import default_timer as timer
NUM_EPOCHS = 15
BEST_CIDER_SCORE = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(gpt2_model, optimizer)
    end_time = timer()
    BEST_CIDER_SCORE = test_epoch(gpt2_model, BEST_CIDER_SCORE)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    with open('best_cider_score.txt', 'w') as file:
        file.write(f"Best CIDEr Score: {BEST_CIDER_SCORE}")