"""
 File name   : custom_dataset.py
 Description : description

 Date created : 12.08.2021
 Author:  Ihar Khakholka
"""

from forest_generator.datasets import BaseDataset
import os
from typing import Union
from pathlib import Path


class CustomDataset(BaseDataset):
    """
    If you want to load your own dataset - prepare it in in the CustomDataset format:
        dataset:
            {metadata}.txt
            {images}
    And place it to data folder.

    {images} is a folder with images.
    Files in {images} folder has to have names:  "{name}.{format}" for images and "{name}_m.{format}" for masks.

    {metadata}.txt can represent train, test, val parts of dataset (or other).
    {metadata}.txt is represented by list of <samples> included in this particular set of samples. Files are splitted by '\n'.
    Each <sample> is represented by "{image/mask-name} {image extension} {mask extension}"

    Example of {metadata.txt}:
        train.txt:
            images_train/image1 png jpg
            images_train/image2 png jpg
        images:
            image1.png
            image1_m.jpg
            image2.png
            image2_m.jpg

    NOTE: Files in {metadata} without extention!!!
    """

    def __init__(self, dataset_dirmame: Union[Path, str], metadata_name: str = "metadata.txt", prefix_folder: str = None, *args, **kwargs):
        """
        :param dataset_dirmame: Path to dataset dirname
        :param metadata_name: Name of metadata in the format f"{name}.txt"
        :param subfolder: If image name in metadata starts not from dataset dirname root - specify additional folder here
        """
        super(CustomDataset, self).__init__(*args, **kwargs)
        self.metadata_name = metadata_name
        self.dataset_dirmame  = Path(dataset_dirmame)
        self.prefix_folder = prefix_folder

        if not os.path.exists(str(self.dataset_dirmame)):
            raise ValueError(f"No dataset provided at {self.dataset_dirmame}, move it there and repeat!")


    def load_data(self) -> BaseDataset:
        with open(str(self.dataset_dirmame / self.metadata_name)) as f:
            metadata = f.readlines()

        for line in metadata:
            iname, img_ext, mask_ext = line.split()
            if self.prefix_folder:
                path = self.dataset_dirmame / self.prefix_folder / iname
            else:
                path = self.dataset_dirmame / iname

            img_path = "{}.{}".format(path, img_ext)
            mask_path = "{}.{}".format(path, mask_ext)

            self.images.append( img_path )
            self.masks.append( mask_path )

        #check path valid
        if not os.path.exists(str(self.images[0])):
            raise RuntimeError(f"Paths to images specified incorrectly for {self.name} dataset!")
        return self