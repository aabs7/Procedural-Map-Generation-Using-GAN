import matplotlib.pyplot as plt
import core
import utils
from core.constants import *

if __name__ == '__main__':
    parser = utils.get_parser()
    args = parser.parse_args()
    grid_size = args.grid_size
    num_of_hallways = args.num_hallways
    min_spacing_hallways = args.min_spacing_hallways
    boundary_threshold = args.boundary_threshold
    hallway_width = args.hallway_width
    seed = args.seed

    grid_with_lines, line_segment = core.generate_random_lines(
        num_of_lines=num_of_hallways, grid_size=grid_size, boundary_threshold=boundary_threshold, spacing_between_lines=min_spacing_hallways, seed=seed)
    grid_with_hallway = core.inflate_using_convolution(
        grid_with_lines,from_label=semantic_labels['lines'], to_label=semantic_labels['hallway'], inflation_scale=hallway_width)
    grid_with_hallway = grid_with_hallway.copy()
    grid_with_room_space = core.inflate_using_convolution(grid_with_hallway,from_label=semantic_labels['hallway'], to_label=semantic_labels['room'], inflation_scale=10, preserve_previous_label=True)
    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.imshow(grid_with_lines, cmap='viridis')
    plt.subplot(132)
    plt.imshow(grid_with_hallway, cmap='viridis')
    plt.subplot(133)
    plt.imshow(grid_with_room_space, cmap='viridis')
    plt.show()
