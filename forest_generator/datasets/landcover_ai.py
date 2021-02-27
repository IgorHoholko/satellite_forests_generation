"""
 File name   : 
 Description : description

 Date created : 27.02.2021
 Author:  Ihar Khakholka
"""
from forest_generator.datasets.base_dataset import BaseDataset
import os
import numpy as np
import pickle
import shutil
import urllib.request
import requests, zipfile, io


from collections import defaultdict
from tqdm import tqdm


URL = "http://landcover.ai/download/landcover.ai.zip"
PATH = 'D:\Загрузки\landcover.ai.zip'

DATASET_NAME = 'LandcoverAI'

DATA_DIRNAME = BaseDataset.data_dirname() / f"{DATASET_NAME}_dataset"

ESSENTIAL_FILE = BaseDataset.data_dirname() / f"{DATASET_NAME}_loaded.flag"


class LandcoverAI(BaseDataset):
    """LandcoverAI is ...

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




if __name__ == '__main__':
    dataset = LandcoverAI()
    dataset.load_or_generate_data()
    print(f"Dataset name: {dataset.name}")
    # print(f"Contains {len(dataset)} elements")