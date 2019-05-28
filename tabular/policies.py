import numpy as np


class Policy:
    def __init__(self, policy: np.ndarray):
        """
        :param policy: |S|x|A|
        """
        self.policy = policy

    @property
    def PI(self) -> np.ndarray:
        """
        :return: policy |S|x|A|
        """
        return self.policy

    def E(self) -> np.ndarray:
        """
        :return: policy |S|x|S||A|
        """
        selector = np.zeros((self.policy.shape[0], self.policy.shape[0] * self.policy.shape[1]))
        for state in range(self.policy.shape[0]):
            selector[state, state*self.policy.shape[1]:(state+1)*self.policy.shape[1]] = self.policy[state, :]
        return selector

    def sample(self, state_index: int) -> int:
        return np.random.choice(self.policy.shape[1], p=self.policy[state_index])

