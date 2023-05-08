import numpy as np
import matplotlib.pyplot as plt
import core
import utils
from core.constants import *
import torch
# import torchvision
from GAN.DCGAN.model import Generator


if __name__ == '__main__':
    Z_DIM = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(1000, Z_DIM, 1, 1).to(device)
    # load generator model and generate image from fixed noise
    generator = Generator(Z_DIM, 3, 64).to(device)
    generator.load_state_dict(torch.load('freeze_files/final_model/floor_model.pt', map_location=torch.device('cpu')))
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
        # CHANGE FAKE TO NUMPY ARRAY
        fake = fake.detach().cpu()
        fake_numpy = fake.numpy()
        # images= fake_numpy[:, 0, 10:55, 10:55]
        images = utils.standard_scalar(fake_numpy[:, 0, 10:55, 10:55])

    parser = utils.get_parser()
    args = parser.parse_args()
    grid_size = args.grid_size
    num_of_hallways = args.num_hallways
    min_spacing_hallways = args.min_spacing_hallways
    boundary_threshold = args.boundary_threshold
    hallway_width = args.hallway_width
    room_width = args.room_width
    room_length_range = args.room_length
    seed = args.seed
    image_file = f"data/floor_plan_h{num_of_hallways}_{seed}.png"
    grid_with_lines, line_segment = core.generate_random_lines(
        num_of_lines=num_of_hallways, grid_size=grid_size, boundary_threshold=boundary_threshold, spacing_between_lines=min_spacing_hallways, seed=seed)
    grid_with_hallway = core.inflate_using_convolution(
        grid_with_lines,from_label=semantic_labels['lines'], to_label=semantic_labels['hallway'], inflation_scale=hallway_width)
    grid_with_hallway = grid_with_hallway.copy()

    grid_with_room_space = core.inflate_using_convolution(grid_with_hallway, from_label=semantic_labels['hallway'], to_label=semantic_labels['room'], inflation_scale=10, preserve_previous_label=True)
    # grid_with_special_rooms = core.add_special_room(
        # grid_with_hallway, hallway_inflation_scale=hallway_width, room_length_range=room_length_range)
    grid_with_rooms = core.add_rooms_from_GAN(line_segment, grid_with_hallway, hallway_inflation_scale=hallway_width,
                                room_b=room_width, room_l_range=room_length_range, rooms=images)   
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(grid_with_rooms > np.mean(grid_with_rooms[0]))
    plt.axis("off")
    plt.savefig(f"resources/maps_image/floor_plan_h{num_of_hallways}_{seed}.png")
    np.savetxt(f"resources/maps_txt/floor_plan_h{num_of_hallways}_{seed}.txt", grid_with_rooms, fmt="%i")
