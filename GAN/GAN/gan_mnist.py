import torch
from torch import nn
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2gray
from skimage.io import imread 
import os
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def compute_ssim(img1, img2):
    ssim = compare_ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    torch.manual_seed(111)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    lr = 3e-4
    z_dim = 64
    image_dim = 28 * 28 * 1
    batch_size = 32
    num_epochs = 50

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
 
    # to load the mnist dataset, do conversion to tensor and normalization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset= torchvision.datasets.MNIST(
        root="./GAN/dataset", train=True, download=True, transform=transform
    )

    # create a pytorch data loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    opt_disc= torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen= torch.optim.Adam(gen.parameters(), lr=lr)
 
    criterion =  nn.BCELoss()
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataset):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ## Train Discriminator: max log(D(real)) + log(1 - D(G(z))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True) # retain_graph=True means don't clear the gradients
            opt_disc.step()

            ## Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                        f"epoch = [{epoch}/{num_epochs}] \ "
                        f"Loss D: {lossD: .4f}, Loss G: {lossG: .4f}"
                    )
                
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(img_grid_real.cpu().numpy()[1])
                    # plt.imshow(fake[0].cpu().numpy()[0])
                    # plt.imshow(data[0].cpu().numpy()[0])
                    # plt.show()
                    # plt.pause(10)
                    print(compute_ssim(data[0].cpu().numpy()[0], fake[0].cpu().numpy()[0]))
                    writer_fake.add_image(
                            "Mnist Fake Images", img_grid_fake, global_step=step)
                    writer_real.add_image(
                            "Mnist Real Images", img_grid_real, global_step=step)

                    step += 1

