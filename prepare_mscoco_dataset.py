from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import List, Tuple
import json
from torchvision.datasets.utils import download_and_extract_archive


def load_json_list(json_path: pathlib.Path) -> Tuple[List[str], List[str], List[int]]:
    # Initialize image_paths list.
    image_paths = []
    # Initialize ids list.
    image_ids = []
    # Initialize train captions list.
    captions = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    for annotation in tqdm(data['annotations']):
        # load caption and add start-of-caption and end-of-caption words.
        caption = 'boc ' + annotation['caption'] + ' eoc'
        # load id and add 0s till the id's string length is 12 which is the complete name of the image.
        id = annotation['image_id']
        path = '%012d' % id
        image_paths.append(path)
        image_ids.append(id)
        captions.append(caption)

    print('Data is loaded.')

    return image_paths, captions, image_ids

class MSCOCODataset():
    '''
        Utilities for image captioning dataset of MSCOCO

        Initialize:
        # dt = MSCOCODataset()

    '''
    def __init__(self) -> None:

        # name of the dataset.
        self.name = "MSCOCO"

        # url links for the dataset train images in zip format.
        self.train_path = "http://images.cocodataset.org/zips/train2017.zip"

        # url links for the dataset val images in zip format.
        self.val_path = "http://images.cocodataset.org/zips/val2017.zip"

        # url links for the dataset test images in zip format.
        self.test_path = "http://images.cocodataset.org/zips/test2017.zip"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # Project root Path
        self.root_folder = Path('/media/envisage/backup1/Ozkan/Datasets')
        
        # String length
        self.path_string_length = 12

        # dataset Path
        self.dataset_folder = self.root_folder / "MSCOCO"

        # train images Path
        self.train_folder = self.dataset_folder / "train2017"

        # val images Path
        self.val_folder = self.dataset_folder / "val2017"

        # test images Path
        self.test_folder = self.dataset_folder / "test2017"

        # train images Path
        self.train_features_folder = self.dataset_folder / "features_train"

        # val features Path
        self.val_features_folder = self.dataset_folder / "features_val"

        # test features Path
        self.test_features_folder = self.dataset_folder / "features_test"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # train captions path
        self.train_captions = self.annotations_folder / "captions_train2017.json"

        # val captions path
        self.val_captions = self.annotations_folder / "captions_val2017.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the MSCOCO folder.
        '''

        # dataset paths in a list.
        download_list = [self.train_path, self.val_path, self.test_path, self.annotations_path]

        for path in download_list:
            download_and_extract_archive(url=path, download_root=str(self.dataset_folder), remove_finished=True)
            
        # collect all the json files PosixPath into a list.
        list_of_annotations = list(self.annotations_folder.glob("*.json"))
        list_of_zips = list(self.dataset_folder.glob("*.zip"))

        # Delete all the json's other than train and val captions.
        list_of_annotations.extend(list_of_zips)
        for ann in list_of_annotations:
            # Check if it is train captions or val captions if not proceed to deletion process.
            if ann != self.train_captions and ann != self.val_captions:
                # Check if the json file exists.
                if ann.exists():
                    # Delete the json file.
                    ann.unlink()

    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        '''
            Load the MSCOCO captions and their corresponding image ids.
            zip(train_paths, train_captions, train_ids), zip(val_paths, val_ids), zip(test_paths, test_ids)
        '''

        get_id = lambda path: int(path)
        train_paths, train_captions, train_ids = load_json_list(self.train_captions)
        val_paths = [val_path.stem for val_path in list(self.val_folder.glob("*.jpg"))]
        val_ids = [get_id(val_path) for val_path in tqdm(val_paths)]
        test_paths = [test_path.stem for test_path in list(self.test_folder.glob("*.jpg"))]
        test_ids = [get_id(test_path) for test_path in tqdm(test_paths)]
        
        train_data = zip(train_paths, train_captions, train_ids)
        val_data = zip(val_paths, val_ids)
        test_data = zip(test_paths, test_ids)

        return train_data, val_data, test_data


