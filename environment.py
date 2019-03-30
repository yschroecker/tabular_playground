from typing import Optional, Tuple, Any
import markov_decision_process
import policies
import numpy as np
import collections


class TabularEnv:
    def __init__(self, mdp: markov_decision_process.MDP, reward: Optional[np.ndarray]):
        self._mdp = mdp
        self._state = None
        self._reward = reward

    def reset(self) -> int:
        self._state = np.random.choice(self._mdp.num_states, p=self._mdp.initial_states)
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, Any]:
        sa = self._mdp.sa2idx(self._state, action)
        self._state = np.random.choice(self._mdp.num_states, p=self._mdp.T[sa, :])
        is_terminal = self._mdp.terminal_states[self._state]
        if self._reward is None:
            return self._state, 0, is_terminal, None
        else:
            return self._state, self._reward[sa, self._state], is_terminal, None

    @property
    def num_states(self):
        return self._mdp.num_states

    @property
    def num_actions(self):
        return self._mdp.num_actions


Trajectory = collections.namedtuple('Trajectory', ['states', 'actions', 'rewards'])


def run_trajectory(env: TabularEnv, policy: policies.Policy) -> Trajectory:
    states = [env.reset()]
    actions = []
    rewards = []
    is_terminal = False
    while not is_terminal:
        actions.append(np.random.choice(env.num_actions, p=policy.PI[states[-1], :]))
        state, reward, is_terminal, _ = env.step(actions[-1])
        states.append(state)
        rewards.append(reward)

    return Trajectory(states, actions, rewards)
