# A/B Testing: Multi-Armed Bandit Experiment

This repository contains an implementation of a multi-armed bandit experiment to simulate A/B testing scenarios. The project evaluates and compares two classic exploration-exploitation algorithms: **Epsilon-Greedy** and **Thompson Sampling**.

## Project Overview

The experiment consists of four advertisement options (bandits) with continuous Gaussian rewards. The true underlying means of the rewards are `[1, 2, 3, 4]`. The goal of the agent is to maximize the cumulative reward over 20,000 trials by finding and exploiting the optimal bandit while minimizing cumulative regret.

* **Epsilon-Greedy:** Explores random arms with probability epsilon (decaying at 1/t) and exploits the best-known arm with probability 1 - epsilon.
* **Thompson Sampling:** A Bayesian approach that updates a Gaussian conjugate prior (with known precision) to balance exploration and exploitation based on posterior uncertainty.

## Requirements

Ensure you have Python installed along with the following dependencies:
`pip install numpy pandas matplotlib loguru`

*(Optional) To auto-format the docstrings, the `pyment` package is recommended:*
`pip install pyment`

## How to Run

To run the experiment, execute the main Python script from your terminal:

`python Bandit.py`

### Expected Outputs
1. **Console Logs:** The `loguru` logger will output the step-by-step progress, average rewards, and average regrets directly to the terminal.
2. **CSV Export:** A file named `experiment_results.csv` will be generated containing the complete history of `{Bandit, Reward, Algorithm}` for both algorithms.
3. **Visualizations:** Matplotlib will render three separate plot windows:
   * **Plot 1:** Learning Process for Epsilon-Greedy (Estimated Means Convergence on Linear and Log scales).
   * **Plot 2:** Learning Process for Thompson Sampling (Estimated Means Convergence on Linear and Log scales).
   * **Plot 3:** Performance Comparison (Cumulative Rewards and Cumulative Regrets over time comparing both algorithms).
---

## BONUS: Suggested Better Implementation Plan (10 Points)

While the provided object-oriented template is excellent for an initial introduction to A/B testing and bandit algorithms, it violates a few core software engineering principles—most notably the **Single Responsibility Principle (SRP)**. 

If this experiment were to be scaled for a production environment or a more complex reinforcement learning task, I would suggest the following implementation plan:

### 1. Decouple the Environment from the Agent
Currently, the algorithm classes (e.g., `EpsilonGreedy`) act as both the **Environment** (they hold the true means and generate the rewards via `pull()`) and the **Agent** (they hold the estimated means and decide which arm to pull). 
* **The Fix:** Create a distinct `BanditEnvironment` class that holds the true reward distribution and takes an `action` as input to return a `reward`. Create separate `Agent` classes (E-Greedy, Thompson) that only take in observations and output actions. This prevents "data leakage," ensuring the agent code cannot accidentally access the true environment parameters.

### 2. Create a Dedicated Experiment Runner
The `experiment()` method is currently embedded inside the agent class. This makes it difficult to run multi-agent comparisons or hyperparameter sweeps cleanly.
* **The Fix:** Implement an `ExperimentRunner` or `Simulation` class. This class would take an `Environment` and a list of `Agents` as inputs, handle the loop over N trials, facilitate the action-reward exchange, and aggregate the data. 

### 3. Optimize Memory and Data Logging
In the current implementation, every reward, regret, and action is appended to dynamic Python lists during the 20,000 trials. For massive experiments, `list.append()` becomes memory-inefficient and slow.
* **The Fix:** Pre-allocate NumPy arrays at the start of the experiment (e.g., `self.rewards = np.zeros(num_trials)`) since the total number of trials is known beforehand. This provides a massive speedup. 

### 4. Utilize Modern Experiment Tracking
The `Visualization` class and custom CSV export logic work for this small scale, but they do not scale well when testing dozens of different epsilon values or precision settings.
* **The Fix:** Integrate an MLOps tracking tool like **MLflow** or **Weights & Biases (WandB)**. Instead of manually plotting Matplotlib charts, these tools automatically log hyperparameters, log real-time metrics (like cumulative regret), and generate interactive, shareable dashboards natively.
