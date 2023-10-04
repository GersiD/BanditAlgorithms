import numpy as np

class UCB(object):
    """Implementation of the UCB algorithm (naieve)."""
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self.delta = 1.0 / (horizon * horizon)

    def select_arm(self):
        """Selects the arm to pull."""
        # Pull each arm once.
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # Select the arm with the highest UCB value.
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            bonus = np.sqrt(2 * np.log(1.0 / self.delta) / self.counts[arm])
            ucb_values[arm] = self.means[arm] + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Updates the algorithm's beliefs."""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.means[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.means[arm] = new_value
