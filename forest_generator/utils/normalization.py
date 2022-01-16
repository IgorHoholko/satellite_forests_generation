'''
 # @ Author: Igor Hoholko
 # @ Create Time: 2021-07-19 11:10:57
 # @ Modified by: Igor Hoholko
 # @ Modified time: 2021-07-19 11:11:52
 # @ Description: Function for image normalization
 '''

from torchvision import transforms as T
import albumentations as A
import torch
from typing import Union


def get_normalization(normalization: str, backend: str = 'A') -> Union[torch.nn.Module, A.BasicTransform]:
    """Get callable object for image normalization.

    Parameters
    ----------
    normalization : str
        Choose one of: none, standardization, normalization, centering05 or centering1.
    backend : str, optional
        A - albumentation transforms, T - Torchvision transforms., by default 'A'

    Returns
    -------
    Union[T.Compose, A.Compose]
        Callable object to transform image.

    Raises
    ------
    ValueError
    """
    t = T if backend == 'T' else A
    if normalization == 'none':
        return None
    elif normalization == 'standardization':
        return t.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif normalization == 'normalization':
        return t.Normalize((0, 0, 0), (1, 1, 1))
    elif normalization == 'centering05':
        return t.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    elif normalization == 'centering1':
        return t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError(f"There is no such normalization: {normalization}!")

def get_denormalization(normalization: str, backend: str = 'A') -> Union[T.Compose, A.Compose]:
    """Get callable object for image normalization.

    Parameters
    ----------
    normalization : str
        Choose one of: none, standardization, normalization, centering05 or centering1.
    backend : str, optional
        A - albumentation transforms, T - Torchvision transforms., by default 'A'

    Returns
    -------
    Union[T.Compose, A.Compose]
        Callable object to transform image.

    Raises
    ------
    ValueError
    """
    t = T if backend == 'T' else A
    if normalization == 'none':
        return None
    elif normalization == 'standardization':
        return t.Compose([
            t.Normalize((0,0,0), (1./0.229, 1./0.224, 1./0.225)),
            t.Normalize((-0.485, -0.456, -0.406), (1,1,1)),
            t.Lambda(lambda x: x*255)
        ])
    elif normalization == 'normalization':
        return t.Lambda(lambda x: x*255)
    elif normalization == 'centering05':
        return t.Compose([
            t.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            t.Lambda(lambda x: x * 255)
        ])
    elif normalization == 'centering1':
        return t.Compose([
            t.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
            t.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            t.Lambda(lambda x: x * 255)
        ])
    else:
        raise NotImplementedError(f"There is no such normalization: {normalization}!")
