# A/B Testing: Multi-Armed Bandit Experiment

This repository contains an implementation of a multi-armed bandit experiment to simulate A/B testing scenarios. The project evaluates and compares two classic exploration-exploitation algorithms: **Epsilon-Greedy** and **Thompson Sampling**.

## Project Overview

The experiment consists of four advertisement options (bandits) with continuous Gaussian rewards. The true underlying means of the rewards are `[1, 2, 3, 4]`. The goal of the agent is to maximize the cumulative reward over 20,000 trials by finding and exploiting the optimal bandit while minimizing cumulative regret.

* **Epsilon-Greedy:** Explores random arms with probability epsilon (decaying at 1/t) and exploits the best-known arm with probability 1 - epsilon.
* **Thompson Sampling:** A Bayesian approach that updates a Gaussian conjugate prior (with known precision) to balance exploration and exploitation based on posterior uncertainty.

## Requirements

Ensure you have Python installed along with the following dependencies:
`pip install numpy pandas matplotlib loguru`

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
