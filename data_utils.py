# TODO tokenize captions first and then feed to the class Dataset.
# TODO document the processings and have comments in codes.
import torch

from text_processing import get_vocab, preprocess_txt

# Parameters
# Eva:
# PARAMS = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 16}

# ozkan lab computer:
PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}

class TestDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, ids, feature_folder):
        'Initialization'
        self.paths = paths
        self.ids = ids
        self.feature_folder = feature_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index] + ".pt"
        feature = torch.load(self.feature_folder / name, map_location='cpu')
        id = self.ids[index]

        return feature, id

class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, captions, vocab, tokenizer, preprocessor, feature_folder):
        'Initialization'
        self.ids = ids
        self.captions = captions
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.feature_folder = feature_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.ids[index] + ".pt"
        feature = torch.load(self.feature_folder / name, map_location='cpu')
        
        tokens = self.captions[index]
        tokens = torch.LongTensor(tokens)

        return feature, tokens

def get_loader_and_vocab(dt):
    train_data, val_data, test_data = dt.load_data()
    processed_paths, processed_captions, vocab, tokenizer = get_vocab(train_data)
    train_dataset = TrainDataset(processed_paths, processed_captions, vocab, tokenizer, preprocess_txt, feature_folder=dt.train_features_folder)
    train_loader = torch.utils.data.DataLoader(train_dataset, **PARAMS)
    val_paths, val_ids = zip(*val_data)
    val_dataset = TestDataset(val_paths, val_ids, dt.val_features_folder)
    val_loader = torch.utils.data.DataLoader(val_dataset, **PARAMS)

    if test_data == None:
        test_loader = None
    else:
        test_paths, test_ids = zip(*test_data)
        test_dataset = TestDataset(test_paths, test_ids, dt.test_features_folder)
        test_loader = torch.utils.data.DataLoader(test_dataset, **PARAMS)
    return train_loader, val_loader, test_loader, vocab
