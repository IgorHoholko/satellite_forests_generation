"""
 File name   : base_lit_model.py
 Description : description

 Date created : 22.01.2022
 Author:  Ihar Khakholka
"""



from typing import List, Union
import gc

from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np

from forest_generator.models import BaseModel


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model: BaseModel, cfg: DictConfig):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = model

        self.cfg = cfg

        # self.metrics: List[torchmetrics.Metric] = None

        self.sync_dist = False

    def generate_image(self, x: np.array) -> np.ndarray:
        raise NotImplementedError()

    def on_train_epoch_start(self) -> None:
        gc.collect()

    def on_validation_epoch_start(self) -> None:
        gc.collect()

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        See examples here: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def validation_epoch_end(self, outputs: List) -> None:
        pass

    def test_epoch_end(self, outputs: List) -> None:
        pass

    def set_requires_grad(self, nets: Union[List[nn.Module], nn.Module], requires_grad: bool = False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad