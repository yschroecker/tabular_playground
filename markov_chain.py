import numpy as np


class MarkovChain:
    def __init__(self, transition_matrix: np.ndarray):
        """
        :param transition_matrix: |X|x|X|
        """
        self.transition_matrix = transition_matrix

    def stationary_distribution(self) -> np.ndarray:
        values, vectors = np.linalg.eig(self.transition_matrix.T)
        for value, vector in zip(values, vectors.T):
            # noinspection PyTypeChecker
            if np.isclose(value, 1):
                if np.max(vector) <= 0:
                    vector *= -1
                if np.min(vector) < 0:
                    vector -= np.min(vector)
                return np.real(vector / np.sum(vector))
        raise RuntimeError("Unable to compute stationary distribution")

    def force_ergodic(self):
        self.transition_matrix += 1e-7
        self.transition_matrix /= self.transition_matrix.sum(axis=1)
        return self

    def reverse_time(self) -> '''MarkovChain''':
        if self.transition_matrix.min() < 1e-7:
            self.force_ergodic()
        d = self.stationary_distribution()
        return MarkovChain(self.T.T * d[np.newaxis, :] / d[:, np.newaxis]).force_ergodic()

    @property
    def T(self):
        """
        :return: transition matrix |X|x|X|
        """
        return self.transition_matrix

    @property
    def num_states(self):
        return self.transition_matrix.shape[0]
