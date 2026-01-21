import random
import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def generate_forest(forest_size, initial_infections=None):
    """
    Generates a new forest as a frest_size x forest_size array
    itialize a diseased tree at the center of the forest.
    """

    forest = np.zeros((forest_size, forest_size), dtype=int)
    if initial_infections is not None:
        for i, j in initial_infections:
            forest[i, j] = 1  # Diseased tree
        return forest
    else:
        center = forest_size // 2
        forest[center, center] = 1  # Diseased tree
        return forest


def step_simulation(forest, infection_prob):
    """
    advance the simulation by one time step.
    find each diseased tree and attempt to infect it's neighbors.
    return the updated forest.
    input:
        forest: the current state of the forest.
        infection_prob: probability of an infection.
    returns: the updated forest.
    """
    forest_size = forest.shape[0]
    # we need to track new infections separately to avoid
    # causing a newly infected tree to infect others in the same step
    infected_trees = np.zeros_like(forest)

    # search for diseased trees
    for i in range(forest_size):
        for j in range(forest_size):
            if forest[i, j] == 1:
                # attempt to infect neighbors
                new_infections = infect_neighbors(forest, i, j, infection_prob)
                # track new infections this time step
                infected_trees += new_infections
    # Update the forest with new new_infections at the end of the step
    forest += infected_trees
    # now any non-zero value indicates a diseased infected_trees
    # >1 means the tree has been infected by multiple neighbors this step
    # clipping the values to 0 or 1 keeps the algoritm consistent
    forest = np.clip(forest, 0, 1)
    return forest


def infect_neighbors(forest, i, j, infection_prob):
    """
    attempt to infect the neighbors of the tree at position (i, j)
    input:
        forest: the current state of the forest.
        i, j: the position of the diseased tree.
        infection_prob: probability of an infection.
    returns: a map of new infections in this time step.
    """
    forest_size = forest.shape[0]
    new_infections = np.zeros_like(forest)
    # define the possible neighbor directions vertical, horizontal, diagonal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dir_i, dir_j in directions:
        # add direction to current position
        neighbor_i, neighbor_j = i + dir_i, j + dir_j
        # check if neighbor is within bounds
        if 0 <= neighbor_i < forest_size and 0 <= neighbor_j < forest_size:
            # attempt to infect the neighbor
            if forest[neighbor_i, neighbor_j] == 0:  # healthy tree
                if random.random() <= infection_prob:
                    # mark the neighbor as newly infected
                    new_infections[neighbor_i, neighbor_j] = 1
    return new_infections


def run_simulation(forest, infection_prob, max_steps):
    """
    run the forest infection simulation for a given number of steps.
    input:
        forest_size: size of the forest (forest_size x forest_size).
        infection_prob: probability of an infection.
        max_steps: number of time steps to simulate.
    returns: a list of forest states at each time step.
    """
    # save the initial state
    forest_states = [forest.copy()]

    for step in range(max_steps):
        forest = step_simulation(forest, infection_prob)
        forest_states.append(forest.copy())

    return forest_states


def create_movie(forest_states, filename="forest_infection", fps=10):
    """
    create a movie from the list of forest states.
    input:
        forest_states: list of forest states at each time step.
        filename: name of the output movie file.
        fps: frames per second for the movie.
    returns: saves the movie as an mp4 file.
    """
    forst_colors = ListedColormap(["green", "saddlebrown"])

    fig, ax = plt.subplots()
    images = []
    for forest in forest_states:
        img = ax.imshow(forest, cmap=forst_colors, vmin=0, vmax=1)
        images.append([img])

    ani = anim.ArtistAnimation(fig, images, interval=500, blit=True)
    ani.save(f"{filename}.mp4", writer="ffmpeg", fps=fps)
    # plt.show()
    # if ffmpeg isnt installed then pillow works, but exports a .gif
    # ani.save(f"{filename}.gif", writer="pillow", fps=fps)
    print("Animation saved as forest_infection.mp4")


def validate_probability(init_forest, infection_prob, trails=1000):
    """
    run multiple single step simulations to validate that the simulation
    is running correctly.
    input:
        init_forest: initial forest state with a single diseased tree.
        infection_prob: probability of an infection.
        trails: number of simulation trails to run.
    returns:
        Avg Matrix after all trails.
    """
    infection_count = np.zeros_like(init_forest, dtype=float)

    for _ in range(trails):
        new_forest = step_simulation(init_forest.copy(), infection_prob)
        infection_count += new_forest

    avg_infection = infection_count / trails
    return avg_infection


if __name__ == "__main__":
    forest_size = 1000
    infection_prob = 0.4
    max_steps = 500
    num_trials = 1000

    # the instructions say to start with an infection at 3,3
    # however im assuming the instructions are indexed from 1
    # because placing it at 3,3 in a 5x5 forest places it at the edge.
    # so ill place it at (2,2) which makes the numbers more clear
    # its easy to change if needed by changing the coordinates in the list.
    initial_infections = [(2, 2)]
    forest = generate_forest(forest_size, initial_infections)
    print("Initial forest state:")
    print(forest)

    # # Validate the infection probability
    # time_start_val = time.time()
    # avg_infection = validate_probability(forest, infection_prob, trails=num_trials)
    # print("Average infection matrix after validation:")
    # print(avg_infection)
    # time_end_val = time.time()
    # print(f"Validation completed in {time_end_val - time_start_val:.2f} seconds.\n")
    #
    # initial_infections = [(2, 2), (3, 3)]
    # forest = generate_forest(forest_size, initial_infections)
    # print("Initial forest state:")
    # print(forest)
    #
    # # Validate the infection probability
    # time_start_val = time.time()
    # avg_infection = validate_probability(forest, infection_prob, trails=num_trials)
    # print("Average infection matrix after validation:")
    # print(avg_infection)
    # time_end_val = time.time()
    # print(f"Validation completed in {time_end_val - time_start_val:.2f} seconds.\n")

    # Run the full simulations
    time_start_sim = time.time()
    print("Running full simulation...")
    forest_states = run_simulation(forest, infection_prob, max_steps)
    # log the forest states to stdout incase the movie fails.
    for step, forest in enumerate(forest_states):
        print(f"Step {step}:\n{forest}\n", end="", flush=True)
        time.sleep(0.1)

    create_movie(forest_states)
    time_end_sim = time.time()
    print(f"Simulation completed in {time_end_sim - time_start_sim:.2f} seconds.")
