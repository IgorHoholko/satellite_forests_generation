"""
 File name   : helpers.py
 Description : description

 Date created : 22.01.2022
 Author:  Ihar Khakholka
"""


import torch

from .base_model import BaseModel
from .helpers import define_D, define_G


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """


    def __init__(self, cfg):
        """Initialize the pix2pix class.
        Parameters:
            cfg -- params for network initialization
        """
        BaseModel.__init__(self)
        # define networks (both generator and discriminator)
        self.netG = define_G(cfg.input_nc, cfg.output_nc, cfg.ngf, cfg.netG, cfg.norm, not cfg.no_dropout, cfg.init_type, cfg.init_gain)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = define_D(cfg.input_nc + cfg.output_nc, cfg.ndf, cfg.netD, cfg.n_layers_D, cfg.norm, cfg.init_type, cfg.init_gain)


    def forward(self, x):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        return self.netG(x)
