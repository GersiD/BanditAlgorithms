from UCB import UCB
from EXP3 import EXP3
import numpy as np
import matplotlib.pyplot as plt

class BernouliBandit(object):
    """Implementation of a Bernouli bandit."""
    def __init__(self, delta):
        self.n_arms = 2
        self.means = [0.5, 0.5 + delta]

    def pull(self, arm):
        """Pulls the arm and returns a reward."""
        return np.random.binomial(1, self.means[arm])

    def run(self, alg, horizon):
        """Runs the algorithm on the bandit."""
        # Run the algorithm.
        regret = 0
        for _ in range(horizon):
            # Select an arm.
            arm = alg.select_arm()
            # Pull the arm.
            reward = self.pull(arm)
            # Update the algorithm.
            alg.update(arm, reward)
            # Update the regret.
            regret += self.means[1] - self.means[arm]
        return regret

def main():
    # Set up the experiment.
    # we have a two armed bernoulli bandit with means 0.5 and 0.5 + delta
    delta = 0.05
    bandit = BernouliBandit(delta)
    # plot the regret of the UCB and EXP3 algorithms
    # as a function of the horizon
    # for horizon in range(10, 10^5, 100):
    ucb_regret = []
    exp3_regret = []
    horizon_list = []
    for horizon in range(10, 100000, 100):
        ucb = UCB(bandit.n_arms, horizon)
        exp3 = EXP3(bandit.n_arms, horizon)
        ucb_regret.append(bandit.run(ucb, horizon))
        exp3_regret.append(bandit.run(exp3, horizon))
        horizon_list.append(horizon)
    plt.plot(horizon_list, ucb_regret, label="UCB")
    plt.plot(horizon_list, exp3_regret, label="EXP3")
    plt.xlabel("Horizon")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("BA11_A.png")
    plt.clf()
    
    # now fix the horizon = 10^5 and plot the regret as a function of eta
    # for eta in range(0.01, 1, 0.01):
    ucb_regret = []
    exp3_regret = []
    min_exp3_regret = float("inf")
    min_exp3_eta = None
    eta_list = []
    horizon = 100000
    ucb = UCB(bandit.n_arms, horizon)
    for eta in range(1, 100, 1):
        exp3 = EXP3(bandit.n_arms, horizon, eta)
        ucb_regret.append(bandit.run(ucb, horizon))
        exp3_regret.append(bandit.run(exp3, horizon))
        eta_list.append(eta)
        if exp3_regret[-1] < min_exp3_regret:
            min_exp3_regret = exp3_regret[-1]
            min_exp3_eta = eta
    print(f"Best empirical eta for delta = {delta}: ", min_exp3_eta)
    plt.plot(eta_list, ucb_regret, label="UCB")
    plt.plot(eta_list, exp3_regret, label="EXP3")
    plt.xlabel("Eta")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("BA11_B.png")



if __name__ == "__main__":
    main()
