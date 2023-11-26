from numpy.lib.function_base import append
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time 


eng_prefixes = {
    "i am": "i m",
    "he is": "he s",
    "she is": "she s",
    "you are": "you re",
    "we are": "we re",
    "they are": "they re"
}
def get_captions_length(captions, tokenizer):
    captions_lengths = [0] * 175
    for caption in captions:
        _, processed_caption_list = preprocess_txt(caption, tokenizer)
        caption_length = len(processed_caption_list)
        captions_lengths[caption_length] += 1
    
    
    return captions_lengths

def filter_captions_through_lengths(paths, captions, tokenizer, min_length, max_length):

    filtered_paths = []
    filtered_captions = []
    for path, caption in zip(paths, captions):
        processed_caption, processed_caption_list = preprocess_txt(caption, tokenizer)
        len_of_caption = len(processed_caption_list)
        if (len_of_caption > min_length) and (len_of_caption <= max_length):
            for i in range(max_length - len_of_caption):
                processed_caption += " pad"
            filtered_captions.append(processed_caption)
            filtered_paths.append(path)
            
    return filtered_paths, filtered_captions



def preprocess_txt(text:str, tokenizer) -> str:
    punctuations="\\?:!.,;!\"#$%&()*+.,-/:;=?@[\]^_\'{|}~<>"
    filter_text = ""
    input_text = text.lower()
    for x, y in eng_prefixes.items():
        if input_text.find(y) != -1:
            input_text = input_text.replace(y, x)
    for word in tokenizer(input_text):
        if word not in punctuations and not word.isspace():
            filter_text += word + " "
    return filter_text[:-1], tokenizer(filter_text[:-1])

def yield_tokens(train_iter, tokenizer):
    for text in train_iter:
        _, text_list = preprocess_txt(text, tokenizer)
        yield text_list

def tokenize_captions(paths, captions, vocab, tokenizer, max_len):
    result_paths = []
    result_tokens = []
    for path, caption in zip(paths, captions):
        processed_caption, processed_caption_list = preprocess_txt(caption, tokenizer)
        result_paths.append(path)
        tokens = vocab(processed_caption_list)
        for i in range(max_len - len(processed_caption_list)):
            tokens.append(0)
        result_tokens.append(tokens)

    return result_paths, result_tokens


def get_vocab(data, min_freq=1, min_length=9, max_length=17):
    paths, captions, _ = zip(*data)
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    caption_lengths = get_captions_length(captions, tokenizer)
    print("Filtering Captions by lengths...")
    t_start = time.time()
    processed_paths, processed_captions = filter_captions_through_lengths(paths, captions, tokenizer, min_length, max_length)
    t_end = time.time()
    print("Done.")
    print(f"Filtering Time:      {t_end - t_start}")
    caption_iter = iter(processed_captions)
    print("Creating Vocabulary...")
    t_start = time.time()
    vocab = build_vocab_from_iterator(yield_tokens(caption_iter, tokenizer), min_freq=min_freq, specials=[])
    t_end = time.time()
    print("Done.")
    # vocab.set_default_index(vocab["<unk>"])
    print(f"Length of Vocab: {vocab.__len__()}")
    print(f"Vocab Creation Time:      {t_end - t_start}")
    print("Tokenizing Captions...")
    t_start = time.time()
    processed_paths, processed_captions = tokenize_captions(processed_paths, processed_captions, vocab, tokenizer, max_length)
    t_end = time.time()
    print("Done.")
    print(f"Tokenization Time:      {t_end - t_start}")
    return processed_paths, processed_captions, vocab, tokenizer



'''
import torch
from prepare_mscoco_dataset import MSCOCODataset
import spacy
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
dt = MSCOCODataset()
train_data, val_data, test_ids = dt.load_data()
train_ids, train_captions = zip(*train_data)
val_ids, val_captions = zip(*val_data)

train_iter = iter(train_captions)

print(next(train_iter))

t_start = time.time()
vocab = build_vocab_from_iterator(yield_tokens(train_iter), min_freq=7, specials=['<unk>', '<pad>'])
t_end = time.time()
vocab.set_default_index(vocab["<unk>"])
print(f"Length of vocab: {vocab.__len__()}")
print(f"Time:            {t_end - t_start}")

# print(vocab(["hers", "a", "an", "example", "on", "off", "should", "be", "bew", "me", "!"]))
for i in range(10):
    input_txt = preprocess_txt(train_captions[i])
    print(input_txt)
    print(tokenizer(input_txt))
    print(vocab(tokenizer(input_txt)))
'''
