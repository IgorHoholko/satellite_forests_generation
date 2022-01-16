'''
 # @ Author: Igor Hoholko
 # @ Create Time: 2022-01-16 13:19:42
 # @ Modified by: Igor Hoholko
 # @ Modified time: 2022-01-16 15:04:10
 # @ Description:
 '''

from typing import Callable, List, Union, Tuple
import abc
from pathlib import Path
import logging

from torch.utils.data import Dataset
import numpy as np
from turbojpeg import TurboJPEG

from forest_generator.utils.io import read_image


class BaseDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(self, transforms: Callable = None, multy_gpu: bool = False, *args, **kwargs):
        """
        Ready data class for torch.Dataloader.
        :param transforms: Ablumentation transforms, by default None
        """
        super().__init__()

        self.transforms = transforms

        self.images: List[Union[str, np.ndarray]] = []
        self.masks:  List[Union[str, np.ndarray]] = []

        self.multy_gpu = multy_gpu

        self.jpeg_reader = None
        if not self.multy_gpu:
            try:
                self.jpeg_reader = TurboJPEG()
            except  Exception as e:
                logging.log(logging.ERROR, e)

    @abc.abstractmethod
    def load_data(self) -> "BaseDataset":
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    def __len__(self):
        return len(self.images)

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str,  np.ndarray, np.ndarray, np.ndarray]:

        image_fn = str( self.images[idx] )
        image = read_image(image_fn, self.jpeg_reader, use_turbo_jpeg= not self.multy_gpu)

        h, w, _ = image.shape

        if self.transforms:
            augmented = self.transforms(image = image, )

        return image


    def __str__(self) -> str:
        line = f"\nDataset: {self.name}" \
               f"\n\tTotal Images: {len(self.images)}"
        return line