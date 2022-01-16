"""
 File name   : tools.py
 Description : description

 Date created : 12.08.2021
 Author:  Ihar Khakholka
"""

import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import cv2
from typing import Union


def import_class_from(class_: str, from_: str):
    module = importlib.import_module(from_)
    return getattr(module, class_)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make_grid(images: Union[np.ndarray, str], nrows_ncols: tuple, figsize: tuple = (10, 10)):
    if not len(images):
        raise ValueError("Images can't be empty")

    if type(images[0]) == str:
        images[:] = map(lambda path: cv2.cvtColor(cv2.imread(f"/{path}"), cv2.COLOR_BGR2RGB), images)

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, images):
        ax.axis('off')
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    return fig