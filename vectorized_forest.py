import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d


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
    neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # by convolving the forest with the neighbors kernel we can find
    # all the diseased trees and their neighbors in one operation
    neighbor_count = convolve2d(forest, neighbors, mode="same")
    # create a mask of healthy trees that have at least one diseased neighbors
    susceptible_trees = (forest == 0) & (neighbor_count > 0)
    # calculate the infection chance for each susceptible trees
    # p = 1 - (1 - infection_prob) ** neighbors
    infection_chance = 1 - (1 - infection_prob) ** neighbor_count[susceptible_trees]
    # generate random values for each susceptible trees
    random_values = np.random.rand(*infection_chance.shape)
    # determine which susceptible trees get infected
    new_infections = random_values < infection_chance
    # update the forest with new new_infections
    new_forest = forest.copy()
    new_forest[susceptible_trees] = new_infections.astype(int)
    return new_forest


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
    # init_infected is the index of the initially infected tree
    infection_count = np.zeros_like(init_forest, dtype=float)

    for _ in range(trails):
        new_forest = step_simulation(init_forest.copy(), infection_prob)
        infection_count += new_forest

    avg_infection = infection_count / trails

    return avg_infection


def analyze_validation(avg_infection, forest, prob):
    """
    analyze the average infection matrix from the validation step.
    input:
        avg_infection: average infection matrix from the validation step.
        infection_prob: probability of an infection.
    returns:
        prints the expected vs actual infection probabilities for neighbors.
    """
    neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    n_count = convolve2d(forest, neighbors, mode="same")
    expected = np.where(forest == 0, 1 - (1 - prob) ** n_count, 1.0)
    error = np.abs(avg_infection - expected)

    print(f"Mean absolute error: {np.mean(error):.4f}")
    center = forest.shape[0] // 2
    print("\nLocal Probability Check (3x3 center):")
    print(avg_infection[center - 1 : center + 2, center - 1 : center + 2])
    return error, expected


if __name__ == "__main__":
    forest_size = 1000
    infection_prob = 0.4
    max_steps = 400
    num_trials = 1000
    movie_filename = "forest_infection"
    movie_fps = 60

    # initial_infections = [(1, 1), (2, 2)]
    initial_infections = None
    forest = generate_forest(forest_size, initial_infections)

    # Validate the infection probability
    print("Validating infection probability...")
    print(
        f"Parameters: forest_size={forest_size}, "
        + f"infection_prob={infection_prob}, trials={num_trials}"
    )
    time_start_val = time.time()
    avg_infection = validate_probability(forest, infection_prob, trails=num_trials)
    analyze_validation(avg_infection, forest, infection_prob)
    time_end_val = time.time()
    print(f"Validation completed in {time_end_val - time_start_val:.2f} seconds.\n")

    # Run the full simulations
    print("Running full simulation...")
    time_start_sim = time.time()
    forest_states = run_simulation(forest, infection_prob, max_steps)
    time_end_sim = time.time()
    print(f"Simulation completed in {time_end_sim - time_start_sim:.2f} seconds.")

    create_movie(forest_states, filename=movie_filename, fps=movie_fps)
    print(f"Movie created: {movie_filename}.mp4")
