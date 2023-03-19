import numpy as np
from scipy.ndimage import label
import scipy
from skimage.morphology import skeletonize
import sknw

import matplotlib
import utils 
import matplotlib.pyplot as plt
from .constants import * 

matplotlib.use('TkAgg')

# L_TMP = 100
# L_UNSET = -1
# L_BKD = 0
# L_CLUTTER = 1
# L_DOOR = 2
# L_HALL = 3
# L_ROOM = 4
# L_UNK = 5
# L_LINES = 7

# ROOM_DOOR_SPACE = 1
# HALLWAY_ROOM_SPACE = 1
# DOOR_SIZE = 8

# semantic_labels = {
#     'background': L_BKD,
#     'clutter': L_CLUTTER,
#     'door': L_DOOR,
#     'hallway': L_HALL,
#     'room': L_ROOM,
#     'other': L_UNK,
#     'lines': L_LINES,
# }


def generate_random_lines(num_of_lines,
                          grid_size=[200, 150],
                          spacing_between_lines=20,
                          boundary_threshold=20,
                          max_iter=10000,
                          seed=1234):
    """Generate random horizontal and vertical lines in a grid

    Args:
        num_of_lines (integer): Number of lines
        grid_size (list, [x,y]): Size of the grid. Defaults to [200,150].
        spacing_between_lines (int): Spacing between two parallel lines. Defaults to 20.
        boundary_threshold (int, optional): Spacing between lines and boundary of grid. Defaults to 20.
        max_iter (int, optional): Maximum number of iterations to run. Defaults to 1000.
        seed (int, optional)andom seed. Defaults to seed.

    Returns:
        final_semantic_grid: Grid with horizontal and vertical lines
    """
    np.random.seed(seed)

    def _check_if_connected(semantic_grid, grid):
        # 8-neighbor structure/kernel for connected components
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        grid[semantic_grid == semantic_labels['lines']] = 0
        new_grid = 1 - grid.copy()
        _, num_features = label(new_grid, structure=s)
        if num_features > 1:
            return False
        else:
            return True

    counter = 0
    # rows and cols keeps track of the boundaries in which parallel line can't be drawn (within that boundaries)
    row = set()
    col = set()
    ii = num_of_lines
    # space between parallel hallways
    space_between_parallel = spacing_between_lines
    # lower bound of both x and y
    xy_lower_bound = 0 + boundary_threshold + 1
    x_upper_bound = grid_size[0] - boundary_threshold - 1
    y_upper_bound = grid_size[1] - boundary_threshold - 1

    # original_grid = np.ones(grid_size, dtype = int)
    # grid = original_grid.copy()
    grid = np.ones(grid_size, dtype=int)
    final_semantic_grid = grid.copy() * L_TMP
    intermediate_semantic_grid = grid.copy() * L_TMP
    line_segment = []

    while ii:
        counter += 1
        if counter == max_iter:
            print(f"Can't create more than {num_of_lines - ii} lines")
            break

        # Randomly pick a point that is at a safe distance from the
        # boundaries before the inflation
        random_point = np.random.randint(
            xy_lower_bound, [x_upper_bound, y_upper_bound]
        )

        # finds the distance from every boundaries
        distance_to_bounds = [x_upper_bound - random_point[0],
                              random_point[0] - xy_lower_bound + 1,
                              random_point[1] - xy_lower_bound + 1,
                              y_upper_bound - random_point[1]]

        # direction specifies where the line should proceed
        direction = np.argmax(distance_to_bounds)
        sorted_direction = np.argsort(distance_to_bounds)[::-1]

        for direction in sorted_direction:

            if direction == 0:  # Draws from top to bottom (top left is (0,0))
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[random_point[0]:x_upper_bound + 1,
                                           random_point[1]] = semantic_labels['lines']

                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == semantic_labels['lines']] = 0
                    lb = max(random_point[1] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[1] + space_between_parallel + 1, y_upper_bound)
                    lb_buffer = max(
                        random_point[0] - space_between_parallel, xy_lower_bound)

                    line_segment.append(([random_point[0], random_point[1]], [
                                        x_upper_bound, random_point[1]]))
                    # for val in range(lb,ub):
                    col.update(range(lb, ub))
                    row.update(range(lb_buffer, random_point[0]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 1:  # Draws from bottom to top
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[xy_lower_bound:random_point[0] + 1, random_point[1]] = \
                    semantic_labels['lines']

                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == semantic_labels['lines']] = 0
                    lb = max(random_point[1] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[1] + space_between_parallel + 1, y_upper_bound)
                    ub_buffer = min(
                        random_point[0] + space_between_parallel, x_upper_bound)

                    line_segment.append(([random_point[0], random_point[1]], [
                                        xy_lower_bound, random_point[1]]))
                    # for val in range(lb, ub):
                    col.update(range(lb, ub))
                    row.update(range(random_point[0], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 2:  # Draws from right to left
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0], xy_lower_bound:random_point[1] + 1] = \
                    semantic_labels['lines']
                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == semantic_labels['lines']] = 0
                    lb = max(random_point[0] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[0] + space_between_parallel + 1, x_upper_bound)
                    ub_buffer = min(
                        random_point[1] + space_between_parallel, y_upper_bound)

                    line_segment.append(([random_point[0], random_point[1]], [
                                        random_point[0], xy_lower_bound]))
                    # for val in range(lb, ub):
                    row.update(range(lb, ub))
                    col.update(range(random_point[1], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 3:  # Draws for left to right
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0], random_point[1]:y_upper_bound + 1] = \
                    semantic_labels['lines']
                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == semantic_labels['lines']] = 0
                    grid[final_semantic_grid == semantic_labels['lines']] = 0
                    lb = max(random_point[0] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[0] + space_between_parallel + 1, x_upper_bound)
                    lb_buffer = max(
                        random_point[1] - space_between_parallel, xy_lower_bound)

                    line_segment.append(([random_point[0], random_point[1]], [
                                        random_point[0], y_upper_bound]))
                    # for val in range(lb, ub):
                    row.update(range(lb, ub))
                    col.update(range(lb_buffer, random_point[1]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

        if line_connected:
            ii = ii - 1
            line_connected = False

    return final_semantic_grid, line_segment


def inflate_using_convolution(grid, from_label=semantic_labels['lines'], to_label=semantic_labels['hallway'],  inflation_scale=5, preserve_previous_label=False):
    """Inflate the lines by a kernel

    Args:
        grid (array of shape m x n): Grid with lines
        inflation_scale (int, optional): Number pixel to grow in 8-neighbor direction. Defaults to 5.

    Returns:
        grid_with_hallways: grid with inflated lines as hallways
    """
    original_grid = np.zeros_like(grid)
    original_grid[grid == from_label] = 1
    kernel_dim = 2 * inflation_scale + 1
    hallway_inflation_kernel = np.ones((kernel_dim, kernel_dim), dtype=int)

    new_grid= scipy.ndimage.convolve(
        original_grid, hallway_inflation_kernel)
    new_grid[new_grid> 0] = to_label
    new_grid[new_grid== 0] = semantic_labels['background']

    if preserve_previous_label:
        new_grid[grid == from_label] = from_label

    return new_grid.copy()


def determine_intersections(hallway_mask):
    sk = skeletonize(hallway_mask)
    graph = sknw.build_sknw(sk)
    vertex_data = graph.nodes()
    counter = {id: 0
               for id in vertex_data}
    edges = graph.edges()
    for s, e in edges:
        counter[s] += 1
        counter[e] += 1
    pendant_vertices = [key
                        for key in counter
                        if counter[key] == 1]
    intersection_vertices = list(set(vertex_data) - set(pendant_vertices))
    intersections = np.array([vertex_data[i]['o']
                             for i in intersection_vertices])
    pendants = np.array([vertex_data[i]['o']
                        for i in pendant_vertices])
    return {
        'intersections': intersections,
        'deadends': pendants
    }


def count_loops_in_hallways(hallway_mask):
    s = [[1, 1, 1],  # 8-neighbor structure/kernel for connected components
         [1, 1, 1],
         [1, 1, 1]]
    grid = 1 - hallway_mask
    _, num_features = label(grid, s)
    return num_features - 1


def get_properties_of_map():
    pass


# if __name__ == '__main__':
#     parser = utils.get_parser()
#     args = parser.parse_args()
#     grid_size = args.grid_size
#     num_of_hallways = args.num_hallways
#     min_spacing_hallways = args.min_spacing_hallways
#     boundary_threshold = args.boundary_threshold
#     hallway_width = args.hallway_width
#     seed = args.seed

#     grid_with_lines, line_segment = generate_random_lines(
#         num_of_lines=num_of_hallways, grid_size=grid_size, boundary_threshold=boundary_threshold, spacing_between_lines=min_spacing_hallways, seed=seed)
#     grid_with_hallway = inflate_using_convolution(
#         grid_with_lines,from_label=semantic_labels['lines'], to_label=semantic_labels['hallway'], inflation_scale=hallway_width)
#     grid_with_hallway = grid_with_hallway.copy()
#     grid_with_room_space = inflate_using_convolution(grid_with_hallway,from_label=semantic_labels['hallway'], to_label=semantic_labels['room'], inflation_scale=10, preserve_previous_label=True)
#     plt.figure(figsize=(16, 8))
#     plt.subplot(131)
#     plt.imshow(grid_with_lines, cmap='viridis')
#     plt.subplot(132)
#     plt.imshow(grid_with_hallway, cmap='viridis')
#     plt.subplot(133)
#     plt.imshow(grid_with_room_space, cmap='viridis')
#     plt.show()
