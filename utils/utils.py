import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid_size', type=int, nargs=2,
                        default=[200, 150], help='Size of grid')
    parser.add_argument('--num_hallways', type=int,
                        default=6, help='Number of hallways')
    parser.add_argument('--boundary_threshold', type=int,
                        default=25, help='Threshold of boundary (> room_breadth)')
    parser.add_argument('--min_spacing_hallways', type=int,
                        default=25, help='Spacing between hallways')
    parser.add_argument('--hallway_width', type=int,
                        default=4, help='hallway width')
    parser.add_argument('--room_width', type=int,
                        default=10, help='width of the room')
    parser.add_argument('--room_length', type=int, nargs=2,
                        default=[10, 20], help='Length range of room')
    parser.add_argument('--seed', type=int, default=2000,
                        help='random number seed')

    return parser


def standard_scalar(images):
    scaled_images = (images - np.mean(images))/np.std(images)
    return scaled_images