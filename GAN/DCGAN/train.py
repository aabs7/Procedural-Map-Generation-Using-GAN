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


def compute_ssim(img1, img2):
    ssim = compare_ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--use_data', type=str)
    args = parser.parse_args()

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    if args.use_data == 'mnist':
        CHANNELS_IMG = 1
    elif args.use_data == 'celebrity' or args.use_data == 'floor' or args.use_data == 'mit':
        CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 1000
    FEATURES_DISC = 64
    FEATURES_GEN = 64
    transforms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)],
                )
            ]
    )

    if args.use_data == 'mnist':
        dataset = datasets.MNIST(root="resources/dataset/", train=True, transform=transforms, download=True)
    elif args.use_data == 'celebrity':
        dataset = datasets.ImageFolder(root="resources/dataset/celeb_dataset", transform=transforms)
    elif args.use_data == 'floor':
        dataset = datasets.ImageFolder(root="resources/dataset/floor_dataset", transform=transforms)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    discriminator = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    init_weights(generator)
    init_weights(discriminator)

    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    summ_writer_real = SummaryWriter(f"resources/logs/real_{args.use_data}")
    summ_writer_fake = SummaryWriter(f"resources/logs/fake_{args.use_data}")
    generator.train()
    discriminator.train()
    step = 0
    out_path = Path(f'resources/models/{args.use_data}/')
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = generator(noise)

            # Train the discriminator max log(D(x)) + log (1 - D(G(z)))
            disc_real = discriminator(real).reshape(-1)
            loss_disc_real= criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
            output = discriminator(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                        f"epoch = [{epoch}/{NUM_EPOCHS}] \ "
                        f"Loss D: {loss_disc: .4f}, Loss G: {loss_gen: .4f}"
                    )

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    print(compute_ssim(real[:32][0].cpu().numpy()[0], fake[:32][0].cpu().numpy()[0]))
                    summ_writer_real.add_image("Real", img_grid_real, global_step=step)

                    summ_writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
        torch.save(generator.state_dict(), out_path / f"model_{args.use_data}_{epoch}.pt")
            
