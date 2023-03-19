import matplotlib
import utils
import matplotlib.pyplot as plt
import core


def test_generate_lines():
    grid_size = [200, 150] 
    num_of_hallways = 3
    min_spacing_hallways = 25
    hallway_width = 5
    boundary_threshold = 10
    seed = 2000

    grid_with_lines, line_segment = core.generate_random_lines(
        num_of_lines=num_of_hallways, grid_size=grid_size, boundary_threshold=boundary_threshold, spacing_between_lines=min_spacing_hallways, seed=seed)

    plt.figure()
    plt.imshow(grid_with_lines)
    plt.show()
