"""
 File name   : pix2pix.py
 Description : description

 Date created : 22.01.2022
 Author:  Ihar Khakholka
"""

from typing import List
import gc

import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np
from torch import nn
import torch

from .base_lit_model import BaseLitModel
from forest_generator.models import BaseModel
from training.losses import GANLoss, cal_gradient_penalty


class Pix2PixLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors

    def __init__(self, model: BaseModel, cfg: DictConfig):
        super(Pix2PixLitModel, self).__init__(model, cfg)
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # define loss functions
        self.criterionGAN = GANLoss(cfg.gan_mode).to(self.device)
        self.criterion_reconstruction = nn.L1Loss()

    def configure_optimizers(self):
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def training_step(self, batch, batch_idx, optimizer_idx):
        """ Same batch for <optimizer_idx=0,1>"""
        real_images, masks = batch

        if optimizer_idx == 0:
            self.fake_images = self.model(masks)

        #FIXME: add lambdas
        #FIXME: add logginh

        # GENERATOR optimization
        if optimizer_idx == 0:
            fake_input = torch.cat((masks, self.fake_images), 1)
            pred_fake = self.model.netD(fake_input)

            loss_g_gan = self.criterionGAN(pred_fake, target_is_real=True) # try to fool

            loss_g_reconstructed = self.criterion_reconstruction(real_images, self.fake_images)

            loss_g = loss_g_gan + loss_g_reconstructed
            return loss_g


        # DISCRIMINATOR optimization
        if optimizer_idx == 1:
            # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_input = torch.cat((masks, self.fake_images), 1)
            real_input = torch.cat((masks, real_images), 1)

            # Fake; stop backprop to the generator by detaching fake_images
            pred_fake = self.model.netD(fake_input.detach())
            pred_real = self.model.netD(real_input)

            loss_fake = self.criterionGAN(pred_fake, target_is_real=False)
            loss_real = self.criterionGAN(pred_real, target_is_real=True)
            loss_d_gan = (loss_fake + loss_real) / 2
            return loss_d_gan







        self.log("train/loss_localization", loss_localization, on_epoch=True, on_step=False, sync_dist=self.sync_dist)
        self.log("train/loss_classification", loss_classification, on_epoch=True, on_step=False, sync_dist=self.sync_dist)
        self.log("train/loss_landmarks", loss_landmarks, on_epoch=True, on_step=False, sync_dist=self.sync_dist)
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=self.sync_dist)
        # self.train_metrics.update(outputs, y)
        return loss