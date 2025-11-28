# PPO for Finance : PPO Framework for finance

![Python versions](https://img.shields.io/badge/python-3.12.3%2B-blue)
![Status](https://img.shields.io/badge/status-inactive-red)

## Project Overview

This project aims to create an agent capable of **trading on the market** using **PPO (Proximal Policy Optimization)**.  
Training data comes from the **VIX market** (CSV time series). The repository implements :

- A **Gymnasium** environment that builds episodes from CSV market files.
- A modular **PPO** training script (compatible with `stable-baselines3` or custom PPO implementations).
- Utility scripts to load CSVs and evaluate trained agents.
- Minimal monitor and logging support.

> The goal is an easy-to-use framework : provide a CSV, configure the task, train with PPO, then evaluate using the given scripts.



## Repository Structure
```bash
├ data/                 # Saved csv & datasets for markets
├ example/              # Example tasks, configs, demos
├ library/              # Core PPO logic with an installable library
├ model/                # Saved trained models
├ monitor/              # Real-time training visualizer
├ multitask_PPO/        # Multitask PPO training code
├ optuna_optimization/  # Hyperparameter tuning
├ report/               # Auto-generated reports
├ utils/                # Helpers & utilities
├ main.py               # Main training runner
└ README.md
```
---

# Installation

We created a homemade environement to easily launch the PPO agent training logic and Analysis that can be found ine the [library file](./library). Also we will detail the requirement of the enviornement the project has been created.

## Requirement

```bash
pip install -r requirements.txt
```

| Monitor Libraries      | Environment Libraries | Reinforcement Learning / Data Science |
|------------------------|-----------------------|---------------------------------------|
| ansi2html==1.9.2       | gym==0.26.2           | matplotlib==3.10.7                    |
| fastapi==0.122.0       | yfinance==0.2.65      | numpy==2.3.5                          |
| uvicorn==0.38.0        |                       | pandas==2.3.3                         |
|                        |                       | psutil==7.0.0                         |
|                        |                       | scikit_learn==1.7.2                   |
|                        |                       | seaborn==0.13.2                       |
|                        |                       | setuptools==80.9.0                    |
|                        |                       | torch==2.7.1                          |
|                        |                       | tqdm==4.67.1                          |
|                        |                       | transformers==4.55.2                  |


## PPO_Library - Homemade Library

You can find the details of this library in the [PPO Library documentation](./library/README.md)

```bash
pip install -e library/
```

### Library Components

The library automatically exposes the following classes :

| Class             | Description                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| **TradingEnv**    | Gymnasium-compatible trading environment built from CSV market data.          |
| **DataLoader**    | Utility class for loading, preprocessing and spliting market data.            |
| **QLearning**     | Classic tabular Q-Learning agent for discrete state experiments.              |
| **DeepQLearning** | DQN-style agent using a neural network for value approximation.               |
| **PPOAgent**      | Proximal Policy Optimization agent for continuous or discrete action trading. |
| **ACAgent**       | Actor-Critic baseline agent (PPO backbone variant).                           |
| **ModelReport**   | Automatic generation of performance reports, graphs, metrics.                 |
| **ModelTest**     | Backtesting, evaluation tools and metrics for trained agents.                 |


# General usage of the repository

## [main.py](./main.py)
## [PPO_Library](./library)
## [optuna_optimization](./optuna_optimization/)
## [monitor](./monitor/)

# Example of usage

Some examples scripts can be found in the [example section](./example/) :

- [Deep Reinforcement Learning Example](./example/01_DeepReinforcementLearning.py) : showing off how to train a Deep Learning RL agent based on GPT neural network
- [PPO Example](./example/02_PPO.py) : showing off how to use the different method of the PPO library by implementing a basic PPO training loop

# Report 

The [report](./report/) gather the weekly reports of the project with their associated presentations :
- Report :
    - [Week 07 - 20](./report/W07_20/main/main.pdf)
    - [Week 21 - 27](./report/W21_27/main/main.pdf)

- Presentations :
    - [Week 10 - The Greedy Policy](./report/W07_20/presentationW10/presentationW10.pdf)
    - [Week 11 - QLearning results](./report/W07_20/presentationW11/presentationW11.pdf)
    - [Week 14 - DQLearning theory](./report/W07_20/presentationW14/presentationW14.pdf)
    - [Week 15 - Experiment on RL rewards](./report/W07_20/presentationW15/presentationW15.pdf)
    - [Week 16 - Optimizing DQL with Optuna](./report/W07_20/presentationW16/presentationW16.pdf)
    - [Week 17 - Comparing GPT transformer and homemade's](./report/W07_20/presentationW17/presentationW17.pdf)
    - [Week 18 - PPO Agent Theory](./report/W07_20/presentationW18/presentationW18.pdf)
    - [Week 20 - PPO Agent Rework](./report/W07_20/presentationW20/presentationW20.pdf)
    - [Week 21 - Optimizing PPO's episode](./report/W21_27/presentationW21/presentationW21.pdf)
    - [Week 23 - Optimizing PPO (1 day tick)](./report/W21_27/presentationW23/presentationW23.pdf)
    - [Week 24 - Optimizing PPO v1 (5 minute tick)](./report/W21_27/presentationW24/presentationW24.pdf)
    - [Week 26 - Optimizing PPO v2 (5 minute tick)](./report/W21_27/presentationW26/presentationW25.pdf)