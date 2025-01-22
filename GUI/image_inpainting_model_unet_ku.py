# pip installs

import cv2
import os
import PIL
import random
import zipfile
import shutil
import zipfile

from matplotlib import pyplot as plt
from torchvision.datasets import CelebA
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# set default dtype to control memory usage
dtype = torch.float32

COLOR_RANGE = [-1, 1]
COLOR_SPECTRUM = 3
HEIGHT = 216
WIDTH = 176

# clip images to COLOR_RANGE
def clip_images(images, range=COLOR_RANGE):
    return images.clamp(min=range[0], max=range[1])


# helper function to convert GPU tensor to image on CPU
def tensor_to_image(x, permute=True):
    if len(x.shape) == 4:
        x = x[0]  # take first image in batch

    # if it contains values between -1 and 0, scale to [0, 1]
    x = clip_images(x)
    if x.min() < 0:
        x = (x + 1) / 2

    if permute:
        x = x.permute(1, 2, 0)
    x = x.to(torch.float32).cpu().detach().numpy()
    x = np.clip(x, 0, 1)
    return x

class Schedule:
    def __init__(self, T, beta_1=0.0001, beta_T=0.02, type="linear"):
        assert 0 < beta_1 < beta_T < 1, f"Invalid beta values: {beta_1}, {beta_T}"
        if type == "linear":
            self.beta = torch.linspace(beta_1, beta_T, T + 1)
            self.alpha = 1 - self.beta
        elif type == "cosine":
            t = torch.linspace(0, T - 1, T)
            precomputed = np.pi / 2 / T
            self.alpha = torch.cos(t * precomputed) ** 2
            self.beta = 1 - self.alpha
        else:
            raise ValueError(f"Unknown schedule type: {type}")

        self.T = T
        self.alpha_bar = torch.cumprod(self.alpha, 0)  # product of all elements in alpha from 0 to t
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)  # needed for noise generation and repainting
        self.one_minus_alpha_bar = 1 - self.alpha_bar  # needed for repainting
        self.sqrt_one_minus_alpha_bar = torch.sqrt(self.one_minus_alpha_bar)  # needed for noise generation
        self.one_over_sqrt_alpha = 1 / torch.sqrt(self.alpha)  # needed for sampling
        self.beta_over_sqrt_one_minus_alpha_bar = self.beta / self.sqrt_one_minus_alpha_bar  # needed for sampling
        self.sqrt_beta = torch.sqrt(self.beta)

    def to(self, device=device):
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.one_minus_alpha_bar = self.one_minus_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.one_over_sqrt_alpha = self.one_over_sqrt_alpha.to(device)
        self.beta_over_sqrt_one_minus_alpha_bar = self.beta_over_sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_beta = self.sqrt_beta.to(device)
        return self

# typical values
T = 1000
beta_1 = 0.0001
beta_T = 0.02

# UNet taken from "P12.ipynb"
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 216*176)
        )

        # Encoder
        self.enc1 = DoubleConv(4, 128)  # 4 input channels for image (RGB-image) + time embedding
        self.down1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.enc2 = DoubleConv(128, 256)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 256)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(512, 128)  # 512 because of skip connection (256 + 256)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(256, 64)   # 256 because of skip connection (128 + 128)

        # Final convolution
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, t, verbose=False):
        # Create time embeddings
        t_emb = self.time_mlp(t.view(-1, 1))
        t_emb = t_emb.view(-1, 1, 216, 176)

        # Concatenate time embedding with input
        x = torch.cat([x, t_emb], dim=1)

        # Encoder path
        enc1 = self.enc1(x)
        down1 = self.down1(enc1)
        down1 = F.leaky_relu(down1)
        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)
        down2 = F.leaky_relu(down2)

        # Bottleneck
        bottleneck = self.bottleneck(down2)

        # Decoder path
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        output = self.final_conv(dec1)

        if verbose:
            return output, [x, enc1, down1, enc2, down2, bottleneck,
                            up2, dec2, up1, dec1, output]

        return output

class DDPM(nn.Module):
    def __init__(self, schedule, sigma_t=0.05):
        super(DDPM, self).__init__()
        self.schedule = schedule.to(device)  # defines how much noise is added at each time step
        self.noise_predictor = UNet()  # predicts noise given a noisy image and a time t
        self.sigma_t = sigma_t  # variance of noise which is added back during sampling
        self.losses = []  # store training losses
        self.to(device)

    # generate noisy version of the image: x_0 => x_t
    @torch.no_grad()
    def forward_diffusion(self, x_0, t):
        epsilon = torch.randn(x_0.shape).to(device)  # add random noise
        sqrt_alpha_bar_t = self.schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)  # broadcast to batch size
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon  # add noise for first t time steps
        return x_t, epsilon

    # generates images from complete noise
    # https://arxiv.org/pdf/2006.11239, algorithm 2, page 4
    @torch.no_grad()
    def generate_images(self, num_images=1, shape=(COLOR_SPECTRUM, HEIGHT, WIDTH)):
        # shape: (C, H, W) => (num_images, C, H, W)
        shape = (num_images, *shape)  # generate a batch of images
        x_t = torch.randn(shape).to(device)  # complete random noise x_T
        one_over_sqrt_alpha = self.schedule.one_over_sqrt_alpha
        beta_over_sqrt_one_minus_alpha_bar = self.schedule.beta_over_sqrt_one_minus_alpha_bar
        sqrt_beta = self.schedule.sqrt_beta

        for t in range(self.schedule.T, 0, -1):
            # predict variance of noise and remove the noise from the image
            # broadcast scalars to batch size
            z = torch.randn(shape).to(device) if t > 1 else 0
            t_vector = torch.zeros(num_images).to(device).float() + t
            prediction = self.noise_predictor(x_t, t_vector)
            # remove predicted noise
            x_t = one_over_sqrt_alpha[t] * (x_t - beta_over_sqrt_one_minus_alpha_bar[t] * prediction)
            x_t += z * sqrt_beta[t]  # add noise back
        return x_t

    # RePaint: https://arxiv.org/pdf/2201.09865, algorithm 1, page 5
    @torch.no_grad()
    def inpaint(self, image, mask, resample_steps=10):
        # mask with same dimensions as image with 1 where image is known and 0 where image is missing
        mask_known = mask
        mask_unknown = 1 - mask

        sqrt_alpha_bar = self.schedule.sqrt_alpha_bar
        one_minus_alpha_bar = self.schedule.one_minus_alpha_bar
        one_over_sqrt_alpha = self.schedule.one_over_sqrt_alpha
        beta_over_sqrt_one_minus_alpha_bar = self.schedule.beta_over_sqrt_one_minus_alpha_bar
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_bar

        x_t = torch.randn(image.shape).to(device)  # complete random noise
        # repeat from complete noise x_unknown_T to restored pixels x_unknown_0
        for t in reversed(range(self.schedule.T)):  # T-1 to 0
            torch.cuda.empty_cache()
            # resample for each time step except the last one
            for u in range(1, resample_steps + 1):  # 1 to resample_steps
                torch.cuda.empty_cache()
                # define noise epsilon and z which are added back (except for t=0)
                if t > 0:
                    epsilon = torch.randn(image.shape).to(device)
                    z = torch.randn(image.shape).to(device)
                else:
                    epsilon = torch.zeros(image.shape).to(device)
                    z = torch.zeros(image.shape).to(device)

                # forward diffusion step => harmonize known and unknown pixels
                x_known_t = sqrt_alpha_bar[t] * image + one_minus_alpha_bar[t] * epsilon

                # predict variance of noise and remove the noise from the image
                t_vector = torch.full((1,), t, dtype=torch.float).to(device)  # vector of t for all images
                predicted_noise = self.noise_predictor(x_t, t_vector)
                x_unknown_t = one_over_sqrt_alpha[t] * (x_t - beta_over_sqrt_one_minus_alpha_bar[t] * predicted_noise)
                x_unknown_t += self.sigma_t * z  # add noise back

                # combine unmasked pixels of x_known_t and masked pixels of x_unknown_t
                known = x_known_t * mask_known  # set unknown pixels to 0
                unknown = x_unknown_t * mask_unknown  # set known pixels to 0
                x_t = known + unknown

                if u < resample_steps and t > 0:
                    noise = torch.randn(image.shape).to(device)
                    x_t = sqrt_alpha_bar[t] * x_t + sqrt_one_minus_alpha_bar[t] * noise

        return clip_images(x_t)

    def save_checkpoint(self, epoch, optimizer, checkpoint_path):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": self.losses,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, optimizer):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["train_losses"]