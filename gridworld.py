from typing import Tuple, Union

import numpy as np
import markov_decision_process


G = 1
S = 2
W = 3


def state_reward_to_matrix(reward_vector: np.ndarray, num_actions: int) -> np.ndarray:
    return np.repeat(np.atleast_2d(reward_vector), len(reward_vector) * num_actions, axis=0)


class Gridworld:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def state_index(self, x: int, y: int) -> int:
        return self.width * y + x

    def _pos(self, state_index: int) -> Tuple[int, int]:
        return state_index % self.width, state_index // self.width

    def _consolidate_pos(self, transition_grid, new, original):
        if self.width > new[0] >= 0 and self.height > new[1] >= 0 and transition_grid[new] != W:
            return new
        else:
            return original

    def is_wall(self, x: int, y: int) -> bool:
        return self._transition_grid[x, y] == W

    def __init__(self, transition_grid: np.ndarray, reward_grid: np.ndarray, transition_noise: float = 0):
        self.height = transition_grid.shape[1]
        self.width = transition_grid.shape[0]
        self.num_states = self.width * self.height
        self.num_actions = 4
        self._transition_grid = transition_grid

        # Reward matrix
        reward_vector = np.zeros((self.num_states,))
        terminal_vector = np.zeros((self.num_states,))
        initial_vector = np.zeros((self.num_states,))
        for state in range(self.num_states):
            reward_vector[state] = reward_grid[self._pos(state)]
            terminal_vector[state] = transition_grid[self._pos(state)] == G
            initial_vector[state] = transition_grid[self._pos(state)] == S
        self.reward_matrix = state_reward_to_matrix(reward_vector, self.num_actions)
        initial_vector /= np.sum(initial_vector)

        # Transition matrix
        transition_matrix = np.zeros((self.num_states * self.num_actions, self.num_states))
        for state in range(self.num_states):
            if terminal_vector[state]:
                transition_matrix[state * self.num_actions: (state+1) * self.num_actions, :] = 0
            else:
                pos = self._pos(state)
                left = self.state_index(*self._consolidate_pos(transition_grid, (pos[0] - 1, pos[1]), pos))
                right = self.state_index(*self._consolidate_pos(transition_grid, (pos[0] + 1, pos[1]), pos))
                up = self.state_index(*self._consolidate_pos(transition_grid, (pos[0], pos[1] - 1), pos))
                down = self.state_index(*self._consolidate_pos(transition_grid, (pos[0], pos[1] + 1), pos))

                transition_matrix[state * self.num_actions + self.LEFT, left] += 1 - transition_noise
                transition_matrix[state * self.num_actions + self.LEFT, up] += transition_noise / 2
                transition_matrix[state * self.num_actions + self.LEFT, down] += transition_noise / 2

                transition_matrix[state * self.num_actions + self.RIGHT, right] += 1 - transition_noise
                transition_matrix[state * self.num_actions + self.RIGHT, up] += transition_noise / 2
                transition_matrix[state * self.num_actions + self.RIGHT, down] += transition_noise / 2

                transition_matrix[state * self.num_actions + self.UP, up] += 1 - transition_noise
                transition_matrix[state * self.num_actions + self.UP, left] += transition_noise / 2
                transition_matrix[state * self.num_actions + self.UP, right] += transition_noise / 2

                transition_matrix[state * self.num_actions + self.DOWN, down] += 1 - transition_noise
                transition_matrix[state * self.num_actions + self.DOWN, left] += transition_noise / 2
                transition_matrix[state * self.num_actions + self.DOWN, right] += transition_noise / 2

        self.mdp = markov_decision_process.MDP(transition_matrix, terminal_vector, initial_vector)
        self.reward_vector = (transition_matrix * self.reward_matrix).sum(axis=1)

    def state_repr(self, state: Union[int, np.ndarray]) -> Tuple[int, int]:
        if type(state) is int:
            return self._pos(state)
        else:
            return self._pos(np.asscalar(np.argmax(state)))


def noisy_simple_grid1(transition_noise: float) -> Gridworld:
    return Gridworld(
        np.array(
            [[0, 0, 0, 0, G],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [S, 0, 0, 0, 0]]
        ).T,
        np.array(
            [[0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ).T,
        transition_noise
    )


simple_grid1 = noisy_simple_grid1(0)

simple_grid2 = Gridworld(
    np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, G],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [S, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T,
    np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T
)

maze = Gridworld(
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
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T
)
maze_no_g = Gridworld(
    np.array(
        [[0, 0, 0, 0, 0, 0, 0, W, 0, 0, 0],
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
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).T
)
