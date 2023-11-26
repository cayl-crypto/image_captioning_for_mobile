from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import List, Tuple
import json
from torchvision.datasets.utils import download_and_extract_archive

def load_json_list(json_path: pathlib.Path, tag:str, Val:bool=False) -> Tuple[List[str], List[str]]:

    # Initialize image_paths list.
    image_paths = []
    # Initialize ids list.
    image_ids = []
    # Initialize captions list.
    captions = []
    # Load json file in data
    data = json.loads(json_path.read_bytes())

    # Go through data.
    print(f'Loading {str(json_path)} data...')
    previous_id_len = 0
    for annotation in tqdm(data['annotations']):
        if annotation['is_precanned'] or annotation['is_rejected']:
            continue
        # load caption and add start-of-caption and end-of-caption words.
        caption = 'boc ' + annotation['caption'] + ' eoc'
        # load id and add 0s till the id's string length is 12 which is the complete name of the image.
        # In the VizWiz dataset add corresponding name tag for the images i.e. "VizWiz_train_"
        # For validation originate the ids to 00000000 by subtracting the number of train images.
        id = annotation['image_id']
        
        path = tag + '%08d' % id
        # Append path, id and captions to the list
        image_paths.append(path)
        image_ids.append(id)
        captions.append(caption)

    print('Data is loaded.')

    return image_paths, captions, image_ids

class VizWizDataset():
    '''
        Utilities for image captioning dataset of VizWiz

        Initialize:
        # dt = VizWizDataset()

    '''

    def __init__(self) -> None:

        # name of the dataset.
        self.name = "VizWiz"

        # url links for the dataset train images in zip format.
        self.train_path = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"

        # url links for the dataset val images in zip format.
        self.val_path = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"

        # url links for the dataset test images in zip format.
        self.test_path = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"

        # Project root Path
        self.root_folder = Path('/media/envisage/backup1/Ozkan/Datasets')

        # String length
        self.path_string_length = 8

        # dataset Path
        self.dataset_folder = self.root_folder / "VizWiz"

        # train images Path
        self.train_folder = self.dataset_folder / "train"

        # val images Path
        self.val_folder = self.dataset_folder / "val"

        # test images Path
        self.test_folder = self.dataset_folder / "test"

        # train features Path
        self.train_features_folder = self.dataset_folder / "features_train"

        # val features Path
        self.val_features_folder = self.dataset_folder / "features_val"

        # test features Path
        self.test_features_folder = self.dataset_folder / "features_test"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # train captions path
        self.train_captions = self.annotations_folder / "train.json"

        # val captions path
        self.val_captions = self.annotations_folder / "val.json"

        # test captions path
        self.test_captions = self.annotations_folder / "test.json"

        # train tag
        self.train_tag = "VizWiz_train_"
    
        # val tag
        self.val_tag = "VizWiz_val_"

        # val id start number
        self.val_id_start = 23431

        # test tag
        self.test_tag = "VizWiz_test_"

        # test id start number
        self.test_id_start = 31181

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the VizWiz folder.
        '''

        # dataset paths in a list.
        download_list = [self.train_path, self.val_path, self.test_path, self.annotations_path]

        for path in download_list:
            download_and_extract_archive(url=path, download_root=str(self.dataset_folder), remove_finished=True)

        list_of_zips = list(self.dataset_folder.glob("*.zip"))
        # Delete all the json's other than train and val captions.
        for ann in list_of_zips:
            # Check if the json file exists.
            if ann.exists():
                # Delete the json file.
                ann.unlink()

    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        '''
            Load the VizWiz captions and their corresponding image ids.
            zip(train_paths, train_captions, train_ids), zip(val_paths, val_ids), zip(test_paths, test_ids)
        '''
        get_id = lambda path, tag, id_start : int(path.replace(tag, "")) + id_start
        train_paths, train_captions, train_ids = load_json_list(self.train_captions, tag=self.train_tag)
        val_paths = [val_path.stem for val_path in tqdm(list(self.val_folder.glob("*.jpg")))]
        val_ids = [get_id(val_path, self.val_tag, self.val_id_start) for val_path in val_paths]
        test_paths = [test_path.stem for test_path in tqdm(list(self.test_folder.glob("*.jpg")))]
        test_ids = [get_id(test_path, self.test_tag, self.test_id_start) for test_path in test_paths]
        
        train_data = zip(train_paths, train_captions, train_ids)
        val_data = zip(val_paths, val_ids)
        test_data = zip(test_paths, test_ids)

        return train_data, val_data, test_data


