"""
 File name   : helpers.py
 Description : description

 Date created : 22.01.2022
 Author:  Ihar Khakholka
"""


import abc
import torch.nn


class BaseModel(torch.nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseModel, self).__init__()



