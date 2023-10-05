from UCB import UCB
from EXP3 import EXP3
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

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
    
    def run_experiment(self, horizon):
        print(f"Running experiment for horizon = {horizon}...")
        ucb = UCB(self.n_arms, horizon)
        exp3 = EXP3(self.n_arms, horizon)
        ucb_regret = self.run(ucb, horizon)
        exp3_regret = self.run(exp3, horizon)
        return ucb_regret, exp3_regret

    def run_experiment_eta(self, eta):
        print(f"Running experiment for eta = {eta}...")
        horizon = 100000
        exp3 = EXP3(self.n_arms, horizon, eta)
        exp3_regret = self.run(exp3, horizon)
        return exp3_regret

def main():
    # Set up the experiment.
    # we have a two armed bernoulli bandit with means 0.5 and 0.5 + delta
    delta = 0.05
    bandit = BernouliBandit(delta)
    print("Setup complete. Running experiments...")
    # plot the regret of the UCB and EXP3 algorithms
    # as a function of the horizon
    # for horizon in range(10, 10^5, 100):
    ucb_regret = []
    exp3_regret = []
    horizon_list = list(range(10, 100000, 1000))
    with Pool() as p:
        results = p.map(bandit.run_experiment, range(10, 100000, 1000))
        for ucb_r, exp3_r in results:
            ucb_regret.append(ucb_r)
            exp3_regret.append(exp3_r)
    plt.plot(horizon_list, ucb_regret, label="UCB")
    plt.plot(horizon_list, exp3_regret, label="EXP3")
    # add the theoretical bounds for exp3
    # should be bounded by 2 * sqrt(horizon * num_arms * log(num_arms))
    plt.plot(horizon_list, 2 * np.sqrt(np.array(horizon_list) * bandit.n_arms * np.log(bandit.n_arms)), label="EXP3 bound")
    plt.xlabel("Horizon")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("BA11_A.png")
    plt.clf()
    print("Plotting complete. Running experiment part B...") 
    # now fix the horizon = 10^5 and plot the regret as a function of eta
    # for eta in range(0.01, 1, 0.01):
    exp3_regret = []
    min_exp3_regret = float("inf")
    min_exp3_eta = None
    eta_list = []
    with Pool() as p:
        results = p.map(bandit.run_experiment_eta, np.arange(0.01, 0.11, 0.01))
        for eta, exp3_r in zip(np.arange(0.01, 0.11, 0.01), results):
            eta_list.append(eta)
            exp3_regret.append(exp3_r)
            if exp3_r < min_exp3_regret:
                min_exp3_regret = exp3_r
                min_exp3_eta = eta
    print(f"Best empirical eta for delta = {delta}: ", min_exp3_eta)
    plt.plot(eta_list, exp3_regret, label="EXP3")
    plt.xlabel("Eta")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("BA11_B.png")
    plt.clf()
    print("Done.")


if __name__ == "__main__":
    main()
