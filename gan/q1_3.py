import argparse
import os
from utils import get_args

import torch
import torch.nn as nn

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp=None, interp=None, lamb=None
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    criterion = nn.BCEWithLogitsLoss()
    loss_real = criterion(discrim_real, torch.ones_like(discrim_real))
    # print(loss_real)
    loss_fake = criterion(discrim_fake, torch.zeros_like(discrim_fake))
    loss = (loss_real + loss_fake)/2
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discrim_fake, torch.ones_like(discrim_fake))
    return loss


if __name__ == "__main__":
    args = get_args()
    disc = Discriminator().cuda()
    gen = Generator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    torch.cuda.empty_cache()
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=64,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
