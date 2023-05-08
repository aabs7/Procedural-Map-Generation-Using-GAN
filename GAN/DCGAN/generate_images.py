from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, init_weights

import argparse
from skimage.metrics import structural_similarity as compare_ssim
import os


if __name__ == "__main__":
    Z_DIM = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(1000, Z_DIM, 1, 1).to(device)
    # load generator model and generate image from fixed noise
    generator = Generator(Z_DIM, 3, 64).to(device)
    generator.load_state_dict(torch.load('floor/models/model_floor_999.pt'))
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
        # CHANGE FAKE TO NUMPY ARRAY
        fake = fake.detach().cpu()
        fake_numpy = fake.numpy()
        images = fake_numpy[:, 0, 10:55, 10:55]
        image = images[0]
        import matplotlib.pyplot as plt
        plt.imshow(image, cmap='gray')
        plt.show()
        torchvision.utils.save_image(fake, 'floor/fake_floor_999.png', normalize=True)
    print(images.shape)