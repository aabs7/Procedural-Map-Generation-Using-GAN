import torch
import torch.nn as nn


# Implement Discriminator and Generator

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(
                    channels_img, features_d, kernel_size=4, stride=2, padding=1
                ),
                nn.LeakyReLU(0.2),
                self.layer(features_d, features_d * 2, 4, 2, 1), # 16 x 16
                self.layer(features_d * 2, features_d * 4, 4, 2, 1), # 8 x 8
                self.layer(features_d * 4, features_d * 8, 4, 2, 1), # 4 x 4
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
                nn.Sigmoid(),
            )

    def layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                self.layer(z_dim, features_g * 16, 4, 1, 0), # N x f_g x 16 x 4 x 4
                self.layer(features_g * 16, features_g * 8, 4, 2, 1), # 8 x 8
                self.layer(features_g * 8, features_g * 4, 4, 2, 1), # 16 x 16 
                self.layer(features_g * 4, features_g * 2, 4, 2, 1), # 32 x 32
                nn.ConvTranspose2d(
                    features_g * 2, channels_img, kernel_size=4, stride=2, padding=1,
                    ),
                nn.Tanh(), # [-1, 1]
            )

    def layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
    def forward(self, x):
        return self.model(x)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

