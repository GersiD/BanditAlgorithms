import numpy as np
import scipy.special as sps

class EXP3(object):
    """Implementation of the EXP3 algorithm (naieve)."""
    def __init__(self, n_arms, horizon, eta = None):
        self.n_arms = n_arms
        self.S = np.zeros(n_arms)
        self.eta = np.sqrt(np.log(n_arms) / (horizon * n_arms))
        if eta:
            self.eta = eta

    def select_arm(self):
        """Selects the arm to pull."""
        # Select the arm with the highest EXP3 value.
        exp3_values = sps.softmax(self.eta * self.S)
        return np.random.multinomial(1, exp3_values).argmax()

    def update(self, arm, reward):
        """Updates the algorithm's beliefs."""
        for i in range(self.n_arms):
            if i != arm:
                # for every other arm add 1
                self.S[i] += 1
            else:
                self.S[arm] = self.S[arm] + 1 - ((1 - reward) / sps.softmax(self.eta * self.S)[arm])
