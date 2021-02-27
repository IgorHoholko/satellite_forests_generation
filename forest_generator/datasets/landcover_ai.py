"""
 File name   : 
 Description : http://landcover.ai/ dataset Wrapper.

 Date created : 27.02.2021
 Author:  Ihar Khakholka
"""
from forest_generator.datasets.base_dataset import BaseDataset

import os
import glob
import cv2
import shutil
from tqdm import tqdm
import requests, zipfile, io


TARGET_SIZE = 512

URL = "http://landcover.ai/download/landcover.ai.zip"
PATH = 'D:\Загрузки\landcover.ai.zip'

DATASET_NAME = 'LandcoverAI'

DATA_DIRNAME = BaseDataset.data_dirname() / f"{DATASET_NAME}_dataset"

ESSENTIAL_FILE = BaseDataset.data_dirname() / f"{DATASET_NAME}_loaded.flag"


class LandcoverAI(BaseDataset):
    """The LandCover.ai (Land Cover from Aerial Imagery) dataset is a dataset
     for automatic mapping of buildings, woodlands and water from aerial images.

    * land cover from Poland, Central Europe 1
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * segmentation masks for three classes: buildings, woodlands and water
    * total area of 216.27 km2 (1.85 km2 of buildings, 72.22 km2 of woodlands, 13.25 km2 of water)
    """
    def __init__(self):
        super(LandcoverAI).__init__()


    def load_or_generate_data(self):
        if not os.path.exists(ESSENTIAL_FILE):
            _get_and_process()








    @property
    def name(self):
        return self.__class__.__name__



def _get_and_process():
    if not os.path.exists(DATA_DIRNAME):
        os.makedirs(DATA_DIRNAME)

    loaded = False

    if PATH:
        try:
            print(f"Unpacking Dataset from {PATH} to {DATA_DIRNAME}...")
            shutil.unpack_archive(PATH, DATA_DIRNAME)
            loaded = True
        except:
            Warning(f'Failed to load data from {PATH}')

    if not loaded and URL:
        try:
            print(f'Beginning {DATASET_NAME} dataset download...')
            r = requests.get(URL)
            if r.ok:
                print(f"Unpacking Dataset from {URL} ...")
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(DATA_DIRNAME)
        except:
            raise ValueError("Specify PATH or URL to the dataset")

    print("Starting processing raw data...")
    _process_raw_data()


    with open(ESSENTIAL_FILE, 'w') as f:
        f.write(f"{DATASET_NAME} is loaded. OK")


def _process_raw_data():
    out_path = DATA_DIRNAME / 'processed'
    img_paths = glob.glob(os.path.join(DATA_DIRNAME / 'images', "*.tif"))
    mask_paths = glob.glob(os.path.join(DATA_DIRNAME / 'masks', "*.tif"))

    img_paths.sort()
    mask_paths.sort()

    os.makedirs(out_path, exist_ok=True)

    for i, (img_path, mask_path) in tqdm(enumerate(zip(img_paths, mask_paths)),
                                         total=len(img_paths), desc="Processing Raw images" ):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(out_path, "{}_{}.jpg".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(out_path, "{}_{}_m.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1



if __name__ == '__main__':
    dataset = LandcoverAI()
    dataset.load_or_generate_data()
    print(f"Dataset name: {dataset.name}")
    # print(f"Contains {len(dataset)} elements")