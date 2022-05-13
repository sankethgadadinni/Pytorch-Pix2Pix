import argparse
from lib2to3.pgen2.pgen import generate_grammar
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

from PIL import Image
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model import Generator, Discriminator
from dataset import ImageDataset

Tensor = torch.FloatTensor

def train(generator, discriminator, dataloader, config):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    lambda_pixel = 100
    patch = (1, config.img_height // 2 ** 4, config.img_width // 2 ** 4)


    
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(batch['B'].type(Tensor))
            real_B = Variable(batch['A'].type(Tensor))
            
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            
        
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = config.num_epochs * len(dataloader) - batches_done

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]"
                % (
                    epoch,
                    config.num_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item()
                )
            )

            # # If at sample interval save image
            # if batches_done % config.sample_interval == 0:
            #     sample_images(batches_done)

    if config.checkpoint_interval != -1 and epoch % config.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (config.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (config.dataset_name, epoch))
        


if __name__ == '__main__':
    
    
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    configFile = OmegaConf.load(args.config)
    config = configFile.config
    
    
    transforms_ = [
    transforms.Resize((config.img_height, config.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset("/home/boltzmann/space/Sanketh/Pytorch-Pix2Pix/facades", transforms_=transforms_),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    generator = Generator()
    discriminator = Discriminator()
    
    train(generator=generator, discriminator=discriminator, dataloader=dataloader, config=config)