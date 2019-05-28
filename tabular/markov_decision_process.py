from typing import Tuple
import numpy as np
from tabular import markov_chain, policies


class MDP:
    def __init__(self, transition_matrix: np.ndarray, terminal_states: np.ndarray, initial_states: np.ndarray):
        """
        MDP\R
        :param transition_matrix: |S||A|x|S|
        :param initial_states: |S|
        :param terminal_states: |S|
        """
        self.transition_matrix = transition_matrix
        self.initial_states = initial_states
        self.terminal_states = terminal_states
        self.num_states = transition_matrix.shape[1]
        self.num_actions = transition_matrix.shape[0]//transition_matrix.shape[1]

    def restarting_mdp(self) -> '''MDP''':
        wrap_around_matrix = self.transition_matrix.copy()
        for state in range(self.num_states):
            if self.terminal_states[state]:
                for action in range(self.num_actions):
                    sa = self.sa2idx(state, action)
                    wrap_around_matrix[sa, :] = self.initial_states
        return MDP(wrap_around_matrix, self.terminal_states, self.initial_states)

    def absorbing_mdp(self) -> '''MDP''':
        absorbing_matrix = self.transition_matrix.copy()
        for state in range(self.num_states):
            if self.terminal_states[state]:
                for action in range(self.num_actions):
                    sa = self.sa2idx(state, action)
                    absorbing_matrix[sa, :] = 0
                    absorbing_matrix[sa, state] = 1
        return MDP(absorbing_matrix, self.terminal_states, self.initial_states)

    def state_action_to_index(self, state: int, action: int) -> int:
        return self.num_actions * state + action

    def state_action_from_index(self, index: int) -> Tuple[int, int]:
        return index // self.num_actions, index % self.num_actions

    sa2idx = state_action_to_index
    idx2sa = state_action_from_index

    def state_process(self, policy: policies.Policy) -> markov_chain.MarkovChain:
        return markov_chain.MarkovChain(policy.E() @ self.T)

    def state_action_process(self, policy: policies.Policy) -> markov_chain.MarkovChain:
        return markov_chain.MarkovChain(self.T @ policy.E())

    @property
    def T(self) -> np.ndarray:
        """
        :return: transition matrix |S||A|x|S|
        """
        return self.transition_matrix


