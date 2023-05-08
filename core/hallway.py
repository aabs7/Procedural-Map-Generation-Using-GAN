import numpy as np
from scipy.ndimage import label
import scipy
from skimage.morphology import skeletonize
import sknw

import matplotlib
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
                          spacing_between_lines=200,
                          boundary_threshold=200,
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


def add_rooms(line_segments, grid_with_hallway, hallway_inflation_scale, room_b, room_l_range):
    grid_with_room = grid_with_hallway.copy()
    for line in line_segments:
        start, end = line
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start

        if is_horizontal:
            # add rooms on horizontal hallway end points
            # end 1
            room_l = np.random.randint(*room_l_range)
            room_p1 = (start[0] - int(room_l / 2), start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
            room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                # add door
                door_p1 = (start[0] - int(DOOR_SIZE / 2), start[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE)
                door_p2 = (door_p1[0] + DOOR_SIZE, door_p1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

            # end 2
            room_l = np.random.randint(*room_l_range)
            room_q1 = (end[0] - int(room_l / 2), end[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1)
            room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                # add door
                door_q1 = (end[0] - int(DOOR_SIZE / 2), end[1] + hallway_inflation_scale + 1)
                door_q2 = (door_q1[0] + DOOR_SIZE, door_q1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

            # add rooms along horizontal hallway
            for y in range(start[1], end[1] - hallway_inflation_scale, 1):
                room_l = np.random.randint(*room_l_range)
                room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, y)
                room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                    # add door
                    door_p1 = (room_p2[0], room_p2[1] - ROOM_DOOR_SPACE - DOOR_SIZE)
                    door_p2 = (room_p2[0] + HALLWAY_ROOM_SPACE, room_p2[1] - ROOM_DOOR_SPACE)
                    # correction for door extending beyond hallway end 2
                    door_check_slice = grid_with_room[door_p2[0]:door_p2[0] + 1, door_p1[1]:door_p2[1]]
                    overflow_len = len(np.where(door_check_slice == semantic_labels['background'])[1])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + room_b, room_p1[1] + ROOM_DOOR_SPACE)
                        door_p2 = (door_p1[0] + HALLWAY_ROOM_SPACE, door_p1[1] + DOOR_SIZE)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

                room_l = np.random.randint(*room_l_range)
                room_q1 = (start[0] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE, y)
                room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                    # add door
                    door_q1 = (room_q1[0] - HALLWAY_ROOM_SPACE, room_q1[1] + ROOM_DOOR_SPACE)
                    door_q2 = (room_q1[0], room_q1[1] + ROOM_DOOR_SPACE + DOOR_SIZE)
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

        else:
            # add rooms on vertical hallway end points
            # end 1
            room_l = np.random.randint(*room_l_range)
            room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, start[1] - int(room_l / 2))
            room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                # add door
                door_p1 = (start[0] - hallway_inflation_scale - HALLWAY_ROOM_SPACE, start[1] - int(DOOR_SIZE / 2))
                door_p2 = (door_p1[0] + HALLWAY_ROOM_SPACE, door_p1[1] + DOOR_SIZE)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

            # end 2
            room_l = np.random.randint(*room_l_range)
            room_q1 = (end[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1, end[1] - int(room_l / 2))
            room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                # add door
                door_q1 = (end[0] + hallway_inflation_scale + 1, end[1] - int(DOOR_SIZE / 2))
                door_q2 = (door_q1[0] + HALLWAY_ROOM_SPACE, door_q1[1] + DOOR_SIZE)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

            # add rooms along vertical hallway
            for x in range(start[0], end[0] - hallway_inflation_scale, 1):
                room_l = np.random.randint(*room_l_range)
                room_p1 = (x, start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
                room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                    # add door
                    door_p1 = (room_p2[0] - ROOM_DOOR_SPACE - DOOR_SIZE, room_p2[1])
                    door_p2 = (room_p2[0] - ROOM_DOOR_SPACE, room_p2[1] + HALLWAY_ROOM_SPACE)
                    # correction for door extending beyond hallway end 2
                    door_check_slice = grid_with_room[door_p1[0]:door_p2[0], door_p2[1]:door_p2[1] + 1]
                    overflow_len = len(np.where(door_check_slice == semantic_labels['background'])[0])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + ROOM_DOOR_SPACE, room_p1[1] + room_b)
                        door_p2 = (door_p1[0] + DOOR_SIZE, door_p1[1] + HALLWAY_ROOM_SPACE)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

                room_l = np.random.randint(*room_l_range)
                room_q1 = (x, start[1] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE)
                room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                    # add door
                    door_q1 = (room_q1[0] + ROOM_DOOR_SPACE, room_q1[1] - HALLWAY_ROOM_SPACE)
                    door_q2 = (room_q1[0] + ROOM_DOOR_SPACE + DOOR_SIZE, room_q1[1])
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

    return grid_with_room


def add_rooms_from_GAN(line_segments, grid_with_hallway, hallway_inflation_scale, room_b, room_l_range, rooms=None):
    grid_with_room = grid_with_hallway.copy()
    for line in line_segments:
        start, end = line
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start
        room_l = rooms[0].shape[0]
        room_b = rooms[0].shape[1]
        # generate a random numbers between 1 to 1000 and store in rand
        # rand = 
        if is_horizontal:
            # add rooms on horizontal hallway end points
            # end 1
            room_p1 = (start[0] - int(room_l / 2), start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
            room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if (np.all(room_slice == semantic_labels['background'])): 
                # In every indices of rooms[0], if the value > 0.5 then set it to 1 else 0
                room_to_attach = rooms[np.random.randint(1, 1000)] 
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room']) 

            # end 2
            room_q1 = (end[0] - int(room_l / 2), end[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1)
            room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if (np.all(room_slice == semantic_labels['background'])): 
                room_to_attach = rooms[np.random.randint(1, 1000)] 
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = np.where(room_to_attach >  np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])

            # add rooms along horizontal hallway
            for y in range(start[1], end[1] - hallway_inflation_scale, 1):
                room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, y)
                room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if (np.all(room_slice == semantic_labels['background'])): 
                    room_to_attach = rooms[np.random.randint(1, 1000)] 
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room']) 

                
                room_q1 = (start[0] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE, y)
                room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if (np.all(room_slice == semantic_labels['background'])): 
                    room_to_attach = rooms[np.random.randint(1, 1000)] 
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])
                
        else:
            # add rooms on vertical hallway end points
            # end 1
            room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, start[1] - int(room_l / 2))
            room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if (np.all(room_slice == semantic_labels['background'])):
                room_to_attach = rooms[np.random.randint(1, 1000)] 
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])
            # end 2
            room_q1 = (end[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1, end[1] - int(room_l / 2))
            room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if (np.all(room_slice == semantic_labels['background'])):
                room_to_attach = rooms[np.random.randint(1, 1000)] 
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])

            # # add rooms along vertical hallway
            for x in range(start[0], end[0] - hallway_inflation_scale, 1):
                room_p1 = (x, start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
                room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if (np.all(room_slice == semantic_labels['background'])):
                    room_to_attach = rooms[np.random.randint(1, 1000)]
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])
                
                room_q1 = (x, start[1] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE)
                room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if (np.all(room_slice == semantic_labels['background'])):
                    room_to_attach = rooms[np.random.randint(1, 1000)]
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = np.where(room_to_attach > np.mean(room_to_attach), semantic_labels['background'], semantic_labels['room'])
                
    return grid_with_room



def add_special_room(grid, hallway_inflation_scale, room_length_range):

    def _check_intersection_or_hallway_end(side_point, extended_point):
        check_point_start = side_point[0]
        check_point_end = side_point[1]
        hallway_end_check_start = extended_point[0]
        hallway_end_check_end = extended_point[1]

        another_intersection_met, hallway_end = False, False

        if grid_with_hallway[check_point_start[0], check_point_start[1]] == semantic_labels['hallway']:
            another_intersection_met = True

        if grid_with_hallway[check_point_end[0], check_point_end[1]] == semantic_labels['hallway']:
            another_intersection_met = True

        if grid_with_hallway[hallway_end_check_start[0], hallway_end_check_start[1]] == semantic_labels['background']:
            hallway_end = True
        if grid_with_hallway[hallway_end_check_end[0], hallway_end_check_end[1]] == semantic_labels['background']:
            hallway_end = True

        return another_intersection_met, hallway_end

    grid_with_hallway = grid.copy()
    features = determine_intersections(
        grid_with_hallway == semantic_labels['hallway'])
    intersections = np.round(features['intersections']).astype(int)
    intersection_with_distance = []
    for i, inter in enumerate(intersections):
        x, y = inter[0], inter[1]
        for next_point in intersections[i + 1:]:
            if next_point[0] == x or next_point[1] == y:
                intersection_with_distance.append(
                    [[inter, next_point], np.linalg.norm(inter - next_point)])
    intersection_with_distance.sort(key=lambda x: x[1])

    # if intersection distance less than a certain room size; remove the intersection
    min_room_length, max_room_length = room_length_range[0], room_length_range[1]
    min_intersection_distance = min_room_length + 3 * hallway_inflation_scale
    max_intersection_distance = max_room_length + 5 * hallway_inflation_scale

    intersection_with_distance = [intersection for intersection in intersection_with_distance if (
        intersection[1] >= min_intersection_distance and intersection[1] < max_intersection_distance)]

    for intersection in intersection_with_distance:
        # find whether the line is horizontal or vertical.
        start, end = intersection[0]
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start
        distance = {}
        if (is_horizontal):
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the ascending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_ascending += 1
                poi_ascending = start[0] + distance_ascending

                check_point_start = [poi_ascending,
                                     start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_ascending,
                                   end[1] - hallway_inflation_scale - 1]

                hallway_end_check_start = [poi_ascending + 1, start[1]]
                hallway_end_check_end = [poi_ascending + 1, end[1]]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)

            distance['ascending'] = distance_ascending
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the descending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_descending += 1
                poi_descending = start[0] - distance_descending

                check_point_start = [poi_descending,
                                     start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_descending,
                                   end[1] - hallway_inflation_scale - 1]

                hallway_end_check_start = [poi_descending - 1, start[1]]
                hallway_end_check_end = [poi_descending - 1, end[1]]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)
            distance['descending'] = distance_descending

            # Add "special room" in the ascending direction
            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + distance['ascending'] - max_room_length,
                           start[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1]
                room_p2 = [end[0] + distance['ascending'], end[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE]
                room_slice = grid_with_hallway[room_p1[0] -
                                               1:room_p2[0] + 1, room_p1[1]:room_p2[1]]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_hallway[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = semantic_labels['room']
                    # add doors
                    grid_with_hallway[room_p1[0] + ROOM_DOOR_SPACE:room_p1[0] + ROOM_DOOR_SPACE + DOOR_SIZE,
                                      room_p1[1] - HALLWAY_ROOM_SPACE:room_p1[1]] = semantic_labels['door']
                    # TODO option for 1 or 2 doors
                    grid_with_hallway[room_p2[0] - ROOM_DOOR_SPACE - DOOR_SIZE:room_p2[0] - ROOM_DOOR_SPACE,
                                      room_p2[1]:room_p2[1] + HALLWAY_ROOM_SPACE] = semantic_labels['door']

            # Add "special room" in the descending direction
            if distance['descending'] > max_room_length:
                room_q1 = [start[0] - distance['descending'] + 1, start[1] +
                           hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1]
                room_q2 = [end[0] - distance['descending'] + max_room_length,
                           end[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE]
                room_slice = grid_with_hallway[room_q1[0] -
                                               1:room_q2[0] + 1, room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_hallway[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = semantic_labels['room']
                    # add doors
                    grid_with_hallway[room_q1[0] + ROOM_DOOR_SPACE:room_q1[0] + ROOM_DOOR_SPACE + DOOR_SIZE,
                                      room_q1[1] - HALLWAY_ROOM_SPACE:room_q1[1]] = semantic_labels['door']
                    # TODO option for 1 or 2 doors
                    grid_with_hallway[room_q2[0] - ROOM_DOOR_SPACE - DOOR_SIZE:room_q2[0] - ROOM_DOOR_SPACE,
                                      room_q2[1]:room_q2[1] + HALLWAY_ROOM_SPACE] = semantic_labels['door']

        else:
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the ascending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_ascending += 1
                poi_ascending = start[1] + distance_ascending
                check_point_start = [
                    start[0] + hallway_inflation_scale + 1, poi_ascending]
                check_point_end = [
                    end[0] - hallway_inflation_scale - 1, poi_ascending]

                hallway_end_check_start = [start[0], poi_ascending + 1]
                hallway_end_check_end = [end[0], poi_ascending + 1]
                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)

            distance['ascending'] = distance_ascending
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the descending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_descending += 1
                poi_descending = start[1] - distance_descending
                check_point_start = [
                    start[0] + hallway_inflation_scale + 1, poi_descending]
                check_point_end = [
                    end[0] - hallway_inflation_scale - 1, poi_descending]

                hallway_end_check_start = [start[0], poi_descending - 1]
                hallway_end_check_end = [end[0], poi_descending - 1]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)
            distance['descending'] = distance_descending

            # Add "special room" in the ascending direction
            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1,
                           start[1] + distance['ascending'] - max_room_length]
                room_p2 = [end[0] - hallway_inflation_scale -
                           HALLWAY_ROOM_SPACE, end[1] + distance['ascending'] - 1]
                room_slice = grid_with_hallway[room_p1[0]:room_p2[0], room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_hallway[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = semantic_labels['room']
                    # add doors
                    grid_with_hallway[room_p1[0] - HALLWAY_ROOM_SPACE:room_p1[0],
                                      room_p1[1] + ROOM_DOOR_SPACE:room_p1[1] + ROOM_DOOR_SPACE + DOOR_SIZE] = (
                        semantic_labels['door'])
                    # TODO option for 1 or 2 doors
                    grid_with_hallway[room_p2[0]:room_p2[0] + HALLWAY_ROOM_SPACE,
                                      room_p2[1] - ROOM_DOOR_SPACE - DOOR_SIZE:room_p2[1] - ROOM_DOOR_SPACE] = (
                        semantic_labels['door'])
            # Add "special room" in the descending direction
            if distance['descending'] > max_room_length:
                room_q1 = [start[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1,
                           start[1] - distance['descending']]
                room_q2 = [end[0] - hallway_inflation_scale - HALLWAY_ROOM_SPACE,
                           end[1] - distance['descending'] + max_room_length]
                room_slice = grid_with_hallway[room_q1[0] +
                                               1:room_q2[0] - 1, room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_hallway[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = semantic_labels['room']
                    # add doors
                    grid_with_hallway[room_q1[0] - HALLWAY_ROOM_SPACE:room_q1[0],
                                      room_q1[1] + ROOM_DOOR_SPACE:room_q1[1] + ROOM_DOOR_SPACE + DOOR_SIZE] = (
                        semantic_labels['door'])
                    # TODO option for 1 or 2 doors
                    grid_with_hallway[room_q2[0]:room_q2[0] + HALLWAY_ROOM_SPACE,
                                      room_q2[1] - ROOM_DOOR_SPACE - DOOR_SIZE:room_q2[1] - ROOM_DOOR_SPACE] = (
                        semantic_labels['door'])

    return grid_with_hallway


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
