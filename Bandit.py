"""
A/B Testing using Epsilon Greedy and Thompson Sampling.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon Greedy algorithm implementation for multi-armed bandits.
    """
    
    def __init__(self, p):
        """
        Initialize the Epsilon Greedy bandit.
        
        :param p: List of true means for each bandit arm.
        :type p: list of float
        """
        self.p = p
        self.k = len(p)
        self.optimal_mean = max(p)
        
        # Trackers
        self.estimated_means = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.rewards_log = []
        self.regrets_log = []
        self.cumulative_rewards = []
        self.cumulative_regrets = []
        self.action_history = []
        self.estimated_means_trace = {i: [] for i in range(self.k)}
        
        self.t = 0 # Current trial step

    def __repr__(self):
        """
        Return string representation of the algorithm.
        
        :return: Name of the algorithm
        :rtype: str
        """
        return "Epsilon-Greedy"

    def pull(self, a):
        """
        Pull a specific bandit arm and return a normally distributed reward.
        
        :param a: Index of the arm to pull.
        :type a: int
        :return: Sampled reward.
        :rtype: float
        """
        # Reward follows N(mean, std=1)
        return np.random.randn() + self.p[a]

    def update(self, a, reward):
        """
        Update the estimated mean of the chosen arm.
        
        :param a: Index of the pulled arm.
        :type a: int
        :param reward: The reward received.
        :type reward: float
        """
        self.action_counts[a] += 1
        n = self.action_counts[a]
        # Incremental mean update
        self.estimated_means[a] = (1 - 1.0/n) * self.estimated_means[a] + (1.0/n) * reward

    def experiment(self, num_trials=20000):
        """
        Run the Epsilon Greedy experiment.
        
        :param num_trials: Total number of pulls, defaults to 20000.
        :type num_trials: int
        """
        logger.info(f"Starting {self.__repr__()} experiment for {num_trials} trials.")
        cumulative_reward = 0
        cumulative_regret = 0

        for trial in range(1, num_trials + 1):
            self.t = trial
            epsilon = 1.0 / self.t  # Decay epsilon by 1/t
            
            # Select action
            if np.random.random() < epsilon:
                action = np.random.choice(self.k) # Explore
            else:
                action = np.argmax(self.estimated_means) # Exploit
                
            # Pull arm and get reward
            reward = self.pull(action)
            regret = self.optimal_mean - self.p[action] # True regret based on means
            
            # Update estimates
            self.update(action, reward)
            
            # Logging
            self.rewards_log.append(reward)
            self.regrets_log.append(regret)
            self.action_history.append(action)
            
            cumulative_reward += reward
            cumulative_regret += regret
            self.cumulative_rewards.append(cumulative_reward)
            self.cumulative_regrets.append(cumulative_regret)
            
            for i in range(self.k):
                self.estimated_means_trace[i].append(self.estimated_means[i])

    def report(self):
        """
        Log the average reward and regret, and format data for CSV export.
        
        :return: DataFrame containing experiment results.
        :rtype: pd.DataFrame
        """
        avg_reward = np.mean(self.rewards_log)
        avg_regret = np.mean(self.regrets_log)
        
        logger.success(f"{self.__repr__()} - Average Reward: {avg_reward:.4f}")
        logger.success(f"{self.__repr__()} - Average Regret: {avg_regret:.4f}")
        logger.success(f"{self.__repr__()} - Cumulative Reward: {self.cumulative_rewards[-1]:.4f}")
        logger.success(f"{self.__repr__()} - Cumulative Regret: {self.cumulative_regrets[-1]:.4f}")
        
        # Prepare data for CSV
        df = pd.DataFrame({
            'Bandit': self.action_history,
            'Reward': self.rewards_log,
            'Algorithm': self.__repr__()
        })
        return df

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling algorithm implementation for multi-armed bandits.
    Uses a Gaussian prior and updates based on known precision.
    """
    
    def __init__(self, p, true_precision=1.0):
        """
        Initialize the Thompson Sampling bandit.
        
        :param p: List of true means for each bandit arm.
        :type p: list of float
        :param true_precision: The known precision (1/variance) of the reward distribution.
        :type true_precision: float
        """
        self.p = p
        self.k = len(p)
        self.optimal_mean = max(p)
        self.true_precision = true_precision
        
        # Bayesian Priors: Start with mean 0 and very low precision (high uncertainty)
        self.m = np.zeros(self.k) 
        self.tau = np.ones(self.k) * 0.0001 
        
        # Trackers
        self.rewards_log = []
        self.regrets_log = []
        self.cumulative_rewards = []
        self.cumulative_regrets = []
        self.action_history = []
        self.estimated_means_trace = {i: [] for i in range(self.k)}

    def __repr__(self):
        """
        Return string representation of the algorithm.
        
        :return: Name of the algorithm
        :rtype: str
        """
        return "Thompson-Sampling"

    def pull(self, a):
        """
        Pull a specific bandit arm.
        
        :param a: Index of the arm to pull.
        :type a: int
        :return: Sampled reward.
        :rtype: float
        """
        # True variance = 1 / true_precision
        true_std = 1.0 / np.sqrt(self.true_precision)
        return np.random.randn() * true_std + self.p[a]

    def update(self, a, reward):
        """
        Update the posterior Gaussian distribution (mean and precision) for the chosen arm.
        
        :param a: Index of the pulled arm.
        :type a: int
        :param reward: The reward received.
        :type reward: float
        """
        # Gaussian conjugate prior update rules
        old_tau = self.tau[a]
        old_m = self.m[a]
        
        # Update precision and mean
        self.tau[a] = old_tau + self.true_precision
        self.m[a] = (old_tau * old_m + self.true_precision * reward) / self.tau[a]

    def experiment(self, num_trials=20000):
        """
        Run the Thompson Sampling experiment.
        
        :param num_trials: Total number of pulls, defaults to 20000.
        :type num_trials: int
        """
        logger.info(f"Starting {self.__repr__()} experiment for {num_trials} trials.")
        cumulative_reward = 0
        cumulative_regret = 0

        for trial in range(num_trials):
            # Sample from the posterior distribution for each arm
            samples = [np.random.randn() / np.sqrt(self.tau[i]) + self.m[i] for i in range(self.k)]
            action = np.argmax(samples)
            
            # Pull arm and get reward
            reward = self.pull(action)
            regret = self.optimal_mean - self.p[action]
            
            # Update posteriors
            self.update(action, reward)
            
            # Logging
            self.rewards_log.append(reward)
            self.regrets_log.append(regret)
            self.action_history.append(action)
            
            cumulative_reward += reward
            cumulative_regret += regret
            self.cumulative_rewards.append(cumulative_reward)
            self.cumulative_regrets.append(cumulative_regret)
            
            for i in range(self.k):
                self.estimated_means_trace[i].append(self.m[i])

    def report(self):
        """
        Log the average reward and regret, and format data for CSV export.
        
        :return: DataFrame containing experiment results.
        :rtype: pd.DataFrame
        """
        avg_reward = np.mean(self.rewards_log)
        avg_regret = np.mean(self.regrets_log)
        
        logger.success(f"{self.__repr__()} - Average Reward: {avg_reward:.4f}")
        logger.success(f"{self.__repr__()} - Average Regret: {avg_regret:.4f}")
        logger.success(f"{self.__repr__()} - Cumulative Reward: {self.cumulative_rewards[-1]:.4f}")
        logger.success(f"{self.__repr__()} - Cumulative Regret: {self.cumulative_regrets[-1]:.4f}")
        
        # Prepare data for CSV
        df = pd.DataFrame({
            'Bandit': self.action_history,
            'Reward': self.rewards_log,
            'Algorithm': self.__repr__()
        })
        return df

#--------------------------------------#

class Visualization():
    """Class to handle plotting and comparisons of Bandit algorithms."""
    
    def plot1(self, algo):
        """
        Visualize the learning process of a bandit algorithm (Estimated Means Convergence).
        
        :param algo: An instantiated and run bandit algorithm object.
        :type algo: Bandit
        """
        trials = range(1, len(algo.estimated_means_trace[0]) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Learning Process: {algo.__repr__()}', fontsize=16)

        # Linear Scale
        for i in range(algo.k):
            ax1.plot(trials, algo.estimated_means_trace[i], label=f'Arm {i} (True Mean={algo.p[i]})')
        ax1.set_xlabel('Trials (Linear Scale)')
        ax1.set_ylabel('Estimated Mean')
        ax1.legend()
        ax1.set_title('Estimated Means Convergence')

        # Log Scale
        for i in range(algo.k):
            ax2.plot(trials, algo.estimated_means_trace[i], label=f'Arm {i}')
        ax2.set_xscale('log')
        ax2.set_xlabel('Trials (Log Scale)')
        ax2.set_ylabel('Estimated Mean')
        ax2.legend()
        ax2.set_title('Estimated Means Convergence (Log Scale)')

        plt.tight_layout()
        plt.show()

    def plot2(self, algo_eg, algo_ts):
        """
        Compare E-greedy and Thompson Sampling cumulative rewards and regrets.
        
        :param algo_eg: Ran Epsilon Greedy algorithm instance.
        :type algo_eg: EpsilonGreedy
        :param algo_ts: Ran Thompson Sampling algorithm instance.
        :type algo_ts: ThompsonSampling
        """
        trials = range(1, len(algo_eg.cumulative_rewards) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Performance Comparison: E-Greedy vs Thompson Sampling', fontsize=16)

        # Cumulative Rewards
        ax1.plot(trials, algo_eg.cumulative_rewards, label='Epsilon-Greedy')
        ax1.plot(trials, algo_ts.cumulative_rewards, label='Thompson Sampling')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Cumulative Reward')
        ax1.legend()
        ax1.set_title('Cumulative Rewards over Time')

        # Cumulative Regrets
        ax2.plot(trials, algo_eg.cumulative_regrets, label='Epsilon-Greedy')
        ax2.plot(trials, algo_ts.cumulative_regrets, label='Thompson Sampling')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('Cumulative Regret')
        ax2.legend()
        ax2.set_title('Cumulative Regrets over Time')

        plt.tight_layout()
        plt.show()

#--------------------------------------#

def comparison():
    """
    Run experiments, store results in a CSV, and trigger visualizations.
    """
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000

    # 1. Initialize and run Epsilon Greedy
    eg = EpsilonGreedy(p=bandit_rewards)
    eg.experiment(num_trials)
    df_eg = eg.report()

    # 2. Initialize and run Thompson Sampling
    ts = ThompsonSampling(p=bandit_rewards, true_precision=1.0)
    ts.experiment(num_trials)
    df_ts = ts.report()

    # 3. Store the rewards in a CSV file
    df_combined = pd.concat([df_eg, df_ts], ignore_index=True)
    csv_filename = "experiment_results.csv"
    df_combined.to_csv(csv_filename, index=False)
    logger.info(f"Results saved to {csv_filename}")

    # 4. Visualization
    viz = Visualization()
    logger.info("Plotting Learning Processes...")
    viz.plot1(eg)
    viz.plot1(ts)
    
    logger.info("Plotting Cumulative Comparisons...")
    viz.plot2(eg, ts)

if __name__=='__main__':
    # Initial log tests as requested by the template
    logger.debug("System Check: debug message")
    logger.info("System Check: info message")
    logger.warning("System Check: warning message")
    logger.error("System Check: error message")
    logger.critical("System Check: critical message")
    
    logger.info("--- STARTING A/B TESTING EXPERIMENT ---")
    comparison()
