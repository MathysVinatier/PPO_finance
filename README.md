# PPO for Finance : PPO Framework for Finance

![Python versions](https://img.shields.io/badge/python-3.12.3%2B-blue)
![Status](https://img.shields.io/badge/status-inactive-red)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Library Components](#ppo_library---homemade-library)
5. [General Usage](#general-usage-of-the-repository)
6. [Monitor](#monitor)
7. [Optuna Optimization](#optuna_optimization)
8. [Examples](#example-of-usage)
9. [Report](#report)

---

## Project Overview

This project aims to create an agent capable of **trading on the market** using **PPO (Proximal Policy Optimization)**.
Training data comes from the **VIX market** (CSV time series). The repository implements:

* A **Gymnasium** environment that builds episodes from CSV market files.
* A modular **PPO** training script (compatible with `stable-baselines3` or custom PPO implementations).
* Utility scripts to load CSVs and evaluate trained agents.
* Minimal monitor and logging support.

> The goal is an easy-to-use framework: provide a CSV, configure the task, train with PPO, then evaluate using the given scripts.

---

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

## Installation

We created a homemade environment to easily launch PPO agent training and analysis.

### Requirements

```bash
pip install -r requirements.txt
```

| Monitor Libraries | Environment Libraries | Reinforcement Learning / Data Science |
| ----------------- | --------------------- | ------------------------------------- |
| ansi2html==1.9.2  | gym==0.26.2           | matplotlib==3.10.7                    |
| fastapi==0.122.0  | yfinance==0.2.65      | numpy==2.3.5                          |
| uvicorn==0.38.0   |                       | pandas==2.3.3                         |
|                   |                       | psutil==7.0.0                         |
|                   |                       | scikit_learn==1.7.2                   |
|                   |                       | seaborn==0.13.2                       |
|                   |                       | setuptools==80.9.0                    |
|                   |                       | torch==2.7.1                          |
|                   |                       | tqdm==4.67.1                          |
|                   |                       | transformers==4.55.2                  |

---

## PPO_Library - Homemade Library

Install the library in editable mode:

```bash
pip install -e library/
```

### Library Components

The library automatically exposes the following classes:

| Class             | Description                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| **TradingEnv**    | Gymnasium-compatible trading environment built from CSV market data.          |
| **DataLoader**    | Utility class for loading, preprocessing, and splitting market data.          |
| **QLearning**     | Classic tabular Q-Learning agent for discrete state experiments.              |
| **DeepQLearning** | DQN-style agent using a neural network for value approximation.               |
| **PPOAgent**      | Proximal Policy Optimization agent for continuous or discrete action trading. |
| **ACAgent**       | Actor-Critic baseline agent (PPO backbone variant).                           |
| **ModelReport**   | Automatic generation of performance reports, graphs, metrics.                 |
| **ModelTest**     | Backtesting and evaluation tools for trained agents.                          |

---

## General Usage of the Repository

### [main.py](./main.py)

*main.py* is used to instantly launch training or analysis:

#### Example of Training

```bash
python main.py --mode training --episode 10 --epoch 10 --batch 64
```

#### Example of Analysis

```bash
python main.py --mode analysis --path ./tmp --task ppo_transformer
```

---

## [monitor](./monitor/)

The monitor folder launches the framework used to train and optimize agents.
It shows PC stats, logs, and plots from training.

```bash
python ./monitor/PPO_training_monitor/main.py
```

Training or optimization steps are saved in databases like [PPO_optuna.db](./monitor/PPO_training_monitor/PPO_optuna.db).

---

## [optuna_optimization](./optuna_optimization/)

The folder stores Optuna optimization files.
Use the utils script to move and format tasks:

```bash
python utils/move_and_format.py --copy ./optuna_optimization/trained_model/ --paste ./multitask_PPO/<your_folder>/data_training/trained_model
```

---

## Example of Usage

Example scripts are in the [example section](./example/):

* [Deep Reinforcement Learning Example](./example/01_DeepReinforcementLearning.py)
* [PPO Example](./example/02_PPO.py)

---

## Report

Reports are saved in [report](./report/) with PDFs and presentations:

### Weekly Reports

* [Week 07 - 20](./report/W07_20/main/main.pdf)
* [Week 21 - 27](./report/W21_27/main/main.pdf)

### Presentations

* [Week 10 - The Greedy Policy](./report/W07_20/presentationW10/presentationW10.pdf)
* [Week 11 - QLearning Results](./report/W07_20/presentationW11/presentationW11.pdf)
* [Week 14 - DQLearning Theory](./report/W07_20/presentationW14/presentationW14.pdf)
* [Week 15 - Experiment on RL Rewards](./report/W07_20/presentationW15/presentationW15.pdf)
* [Week 16 - Optimizing DQL with Optuna](./report/W07_20/presentationW16/presentationW16.pdf)
* [Week 17 - Comparing GPT Transformer and Homemade](./report/W07_20/presentationW17/presentationW17.pdf)
* [Week 18 - PPO Agent Theory](./report/W07_20/presentationW18/presentationW18.pdf)
* [Week 20 - PPO Agent Rework](./report/W07_20/presentationW20/presentationW20.pdf)
* [Week 21 - Optimizing PPO's Episode](./report/W21_27/presentationW21/presentationW21.pdf)
* [Week 23 - Optimizing PPO (1-day tick)](./report/W21_27/presentationW23/presentationW23.pdf)
* [Week 24 - Optimizing PPO v1 (5-minute tick)](./report/W21_27/presentationW24/presentationW24.pdf)
* [Week 26 - Optimizing PPO v2 (5-minute tick)](./report/W21_27/presentationW26/presentationW25.pdf)
