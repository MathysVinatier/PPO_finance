# PPO Finance Library

A Python library for **Reinforcement Learning-based trading** using PPO, AC, Q-Learning, and Deep Q-Learning methods.
This library provides tools to create trading environments, train agents on market data, and evaluate their performance.

---

## Table of Contents

* [Installation](#installation)
* [Library Structure](#library-structure)
* [Modules & Classes](#modules--classes)
* [Usage Examples](#usage-examples)
* [Environment Requirements](#environment-requirements)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MathysVinatier/PPO_finance.git
cd PPO_finance
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the library in editable mode:

```bash
pip install -e library/
```

---

## Library Structure

```bash
library/
├── PPO_Library/
│   ├── __init__.py
│   ├── Environment.py
│   ├── RLMethod.py
│   ├── PPO.py
│   ├── PPO_analysis.py
│   └── ...
```

---

## Modules & Classes

### `Environment.py`

* **TradingEnv** : Custom Gym environment for trading.

  * Handles positions (long/short), balance, rewards.
  * Methods: `reset()`, `step(action)`, `set_data(df)`, `get_valid_actions()`, `sample_valid_action()`.
* **DataLoader** : Utility for fetching, saving, and loading financial datasets.

  * Methods: `save_company()`, `save_companies()`, `read(folder_path)`, `split_train_test()`.

### `RLMethod.py`

* **QLearning**: Standard Q-Learning agent.
* **DeepQLearning**: Deep Q-Learning with optional GPT transformer backbone.

  * Integrates `DecisionTransformerQ` model for state-action prediction.
* **PPOAgent**: Proximal Policy Optimization agent.
* **DecisionTransformerQ**: Transformer-based Q-network.

### `PPO.py`

* **ACAgent**: Actor-Critic PPO agent.
* **ActorNetwork**: Transformer-based actor for action selection.
* **CriticNetwork**: LSTM + Time2Vec critic network.
* Supports `choose_action`, `learn`, `save_models`, `load_models`.

### `PPO_analysis.py`

* **ModelReport**: Visualizes training information and performance metrics.
* **ModelTest**: Evaluates trained models on datasets, calculates metrics like:

  * Total reward
  * Annual return
  * Sharpe ratio
  * Maximum drawdown
  * Action distribution

---

## Usage Examples

### 1. Load data and split train/test

```python
from PPO_Library import DataLoader

loader = DataLoader()
df_train, df_test = loader.split_train_test("data/AAPL_2020.csv", training_size=0.8)
```

### 2. Create a trading environment

```python
from PPO_Library import TradingEnv

env = TradingEnv(df_train, broker_fee=True)
```

### 3. Train a Deep Q-Learning agent

```python
from PPO_Library import DeepQLearning

agent = DeepQLearning(env, gpt=True)
agent.train(episodes=1000)
```

### 4. Train an Actor-Critic PPO agent

```python
from PPO_Library import ACAgent

agent = ACAgent(n_actions=3, num_features=env.observation_space.shape[0], seq_len=10)
agent.learn()
```

### 5. Evaluate a model

```python
from PPO_Library import ModelReport

report = ModelReport(task_path="model/", seq_size=10)
agent = report.get_model(model_episode="latest")
report.plot(show=True)
```

---

## Environment Requirements

| Category                   | Libraries & Versions                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Environment**            | gym==0.26.2, yfinance==0.2.65                                                                                                                                              |
| **Reinforcement Learning** | matplotlib==3.10.7, numpy==2.3.5, pandas==2.3.3, psutil==7.0.0, scikit_learn==1.7.2, seaborn==0.13.2, setuptools==80.9.0, torch==2.7.1, tqdm==4.67.1, transformers==4.55.2 |

---

## Notes

* The library supports **both discrete and continuous action spaces**.
* Transformers and LSTMs are used in PPO and Deep Q-Learning agents for sequence modeling.
* Utilities for plotting, reporting, and testing are included to streamline experiments.

---

This library is designed for **research and prototyping of RL-based financial trading strategies**.
