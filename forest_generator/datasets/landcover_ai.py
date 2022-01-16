"""
 File name   : landcover_ai.py
 Description : http://landcover.ai/ dataset Wrapper.

 Date created : 27.02.2021
 Author:  Ihar Khakholka
"""

from forest_generator.datasets.custom_dataset import CustomDataset
from forest_generator.datasets import BaseDataset


class LandcoverAI(CustomDataset):
    """The LandCover.ai (Land Cover from Aerial Imagery) dataset is a dataset
     for automatic mapping of buildings, woodlands and water from aerial images.

    * land cover from Poland, Central Europe 1
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * segmentation masks for three classes: buildings, woodlands and water
    * total area of 216.27 km2 (1.85 km2 of buildings, 72.22 km2 of woodlands, 13.25 km2 of water)
    """

    NAME = 'LandcoverAI'
    DATA_DIRNAME = BaseDataset.data_dirname() / NAME

    LABELS_NAMES = {0: 'Other', 1: 'Buildings', 2: 'Woodland', 3: 'Water'}
    COLORS = {0: [192, 192, 192],
               1: [102, 51, 0],
               2: [0, 153, 0],
               3: [0, 128, 250]}

    def __init__(self, metadata_name: str, *args, **kwargs):
        super(LandcoverAI, self).__init__(LandcoverAI.DATA_DIRNAME, metadata_name, 'processed', *args, **kwargs)

    @property
    def name(self):
        return self.__class__.__name__



class landcover_ai_train(object):
    def __new__(self,  *args, **kwargs) -> LandcoverAI:
        return LandcoverAI('train.txt')

class landcover_ai_test(object):
    def __new__(self,  *args, **kwargs) -> LandcoverAI:
        return LandcoverAI('test.txt')

class landcover_ai_val(object):
    def __new__(self,  *args, **kwargs) -> LandcoverAI:
        return LandcoverAI('val.txt')


if __name__ == '__main__':
    dataset = landcover_ai_train()
    dataset.load_data()
    print(dataset)

    dataset = landcover_ai_test()
    dataset.load_data()
    print(dataset)

    dataset = landcover_ai_val()
    dataset.load_data()
    print(dataset)