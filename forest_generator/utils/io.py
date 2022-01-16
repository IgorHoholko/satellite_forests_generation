'''
 # @ Author: Igor Hoholko
 # @ Create Time: 2021-07-19 10:57:15
 # @ Modified by: Igor Hoholko
 # @ Modified time: 2021-07-19 11:13:14
 # @ Description: Tools to work with data
 '''

import json
import importlib
import os
from typing import List
import numpy as np
import yaml

from PIL import Image
import cv2
from turbojpeg import TurboJPEG


def get_files_recursively(path: str) -> List[str]:
    list_of_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            list_of_files.append(os.path.join(root, file))
    return list_of_files


def load_json(fp: str) -> dict:
    with open(fp) as f:
        return json.load(f)

def dump_json(fp: str, data: dict):
    with open(fp, 'w') as f:
        json.dump(data, f)


def read_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        f_str = f.read()
        file = yaml.load(f_str, Loader=yaml.FullLoader)
    return file



def read_image_turbo_jpeg(path: str, jpeg_reader: TurboJPEG = None) -> np.array:
    if not jpeg_reader:
        jpeg_reader = TurboJPEG()
    file = open(path, "rb")
    image = jpeg_reader.decode(file.read(), 1)  # 0 - RGB, 1 - BGR
    return image

def read_image_pil(path: str) -> np.array:
    image = np.asarray(Image.open(path))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def read_image(path: str, turbo_jpeg_reader: TurboJPEG = None, use_turbo_jpeg: bool = True) -> np.array:
    if use_turbo_jpeg:
        try:
            image = read_image_turbo_jpeg(path, turbo_jpeg_reader )
        except Exception as e:
            pass
            try:
                image = read_image_pil(path)
            except Exception as e:
                pass
                try:
                    cv2.imread(path)
                except Exception as e:
                    pass
    else:
        try:
            image = read_image_pil(path)
        except Exception as e:
            pass
            try:
                cv2.imread(path)
            except Exception as e:
                pass
    return image