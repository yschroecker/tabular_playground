from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from tabular import solve, gridworld, environment
from tabular.gridworld import W, G, S


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


def draw_arrow(ax: plt.Axes, x: float, y: float, dx: float, dy: float):
    ax.arrow(x + 0.5, y + 0.5, dx/2, dy/2, width=0.03, head_width=min(0.30 * (abs(dx + dy) + 0.01), 0.15))


def draw_policy_arrow_field(ax: plt.Axes, env: gridworld.Gridworld, policy: np.ndarray,
                            leave_out: Sequence[Tuple[int]]=()):
    for x in range(env.width):
        for y in range(env.height):
            if not (x, y) in leave_out:
                state = env.state_index(x, y)
                draw_arrow(ax, x, y, -policy[state, env.LEFT], 0)
                draw_arrow(ax, x, y, policy[state, env.RIGHT], 0)
                draw_arrow(ax, x, y, 0, policy[state, env.DOWN])
                draw_arrow(ax, x, y, 0, -policy[state, env.UP])


def _run():
    # create a new gridworld
    grid = gridworld.Gridworld(
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, W, 0, 0, G],
             [W, W, W, W, W, W, 0, W, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, W, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, W, 0, W, W],
             [0, 0, 0, 0, W, W, 0, W, 0, 0, 0],
             [0, 0, 0, 0, W, 0, W, W, W, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, W, 0, 0, 0, 0, W, 0],
             [0, 0, 0, 0, W, 0, 0, 0, 0, W, 0],
             [S, 0, 0, 0, W, 0, 0, 0, 0, W, 0]]
        ).T,
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, -0.1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).T
    )

    # solve with VI
    policy = solve.value_iteration_discounted(grid.mdp, grid.reward_vector, discount_factor=0.99, precision=1e-9)
    fig, ax = draw_grid(grid)
    draw_policy_arrow_field(ax, grid, policy.PI)
    plt.show()

    # solve with PI
    policy = solve.policy_iteration_discounted(grid.mdp, grid.reward_vector, discount_factor=0.99)
    fig, ax = draw_grid(grid)
    draw_policy_arrow_field(ax, grid, policy.PI)
    plt.show()

    # solve with linprog
    policy = solve.discounted_dual_linprog(grid.mdp, grid.reward_vector, discount_factor=0.99)
    fig, ax = draw_grid(grid)
    draw_policy_arrow_field(ax, grid, policy.PI)
    plt.show()

    # solve with q learning
    env = environment.TabularEnv(grid.mdp, grid.reward_matrix)
    q_values = np.ones((grid.num_states, grid.num_actions)) * np.max(grid.reward_matrix)
    num_steps = 100000
    learning_rate = 0.1
    epsilon = 0.1
    discount_factor = 0.99

    state = env.reset()
    for step in range(num_steps):
        action = np.argmax(q_values[state]) if np.random.rand() > epsilon \
                 else np.random.choice(grid.num_actions)
        next_state, reward, is_terminal, _ = env.step(action)
        if is_terminal:
            q_values[state, action] += learning_rate * (reward - q_values[state, action])
            state = env.reset()
        else:
            q_values[state, action] += learning_rate * (reward + discount_factor * np.max(q_values[next_state]) -
                                                        q_values[state, action])
            state = next_state
    policy = np.zeros((grid.num_states, grid.num_actions))
    policy[np.arange(grid.num_states), np.argmax(q_values, axis=1)] = 1
    fig, ax = draw_grid(grid)
    draw_policy_arrow_field(ax, grid, policy)
    plt.show()


if __name__ == '__main__':
    _run()
