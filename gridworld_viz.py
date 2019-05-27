from typing import Optional

import gridworld
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def draw_grid(env: gridworld.Gridworld):
    plt.close(plt.gcf())
    fig = plt.figure()
    ax: plt.Axes = fig.gca()
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.axis('off')
    for x in range(env.width):
        for y in range(env.height):
            if env.is_wall(x, y):
                field = patches.Rectangle((x, y), 1, 1, fc='black', ec='black')
            else:
                field = patches.Rectangle((x, y), 1, 1, fc='white', ec='black')
            ax.add_patch(field)
    return fig, ax


def plot_value_grid(env: gridworld.Gridworld, value: np.ndarray, min_value: Optional[float], max_value: Optional[float]):
    min_value = np.min(value) if min_value is None else min_value
    max_value = np.max(value) if max_value is None else max_value
    color_value = np.clip(value, min_value, max_value)
    normalized_value = (color_value - min_value)/(max_value - min_value)
    plt.close(plt.gcf())
    fig = plt.figure()
    ax: plt.Axes = fig.gca()
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.axis('off')
    for x in range(env.width):
        for y in range(env.height):
            s = env.state_index(x, y)
            if env.is_wall(x, y):
                field = patches.Rectangle((x, y), 1, 1, fc='black', ec='black')
            else:
                field = patches.Rectangle((x, y), 1, 1, fc='red', ec='black', alpha=normalized_value[s])
            ax.add_patch(field)
            plt.text(x + 0.2, y + 0.3, f"{value[s]:.3}")
    return fig, ax
