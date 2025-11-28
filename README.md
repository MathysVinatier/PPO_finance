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

# Installation + homemade library

# Example of usage

## Main
## Optuna Optimization
## Monitor