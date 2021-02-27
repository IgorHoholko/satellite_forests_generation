"""
 File name   : 
 Description : description

 Date created : 27.02.2021
 Author:  Ihar Khakholka
"""

from pathlib import Path
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_or_generate_data(self):
        pass

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"



