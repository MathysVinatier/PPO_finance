from utils import DataLoader, TradingEnv,  QLearning

import optuna
import multiprocessing

import os
import optuna
from optuna.storages import RDBStorage

def optimization():
    fdir = "data/General/^VIX_2015_2025.csv"
    dataloader = DataLoader()
    df = dataloader.read(fdir)

    # Define Optuna objective
    def objective(trial):
        n_training_episodes = trial.suggest_categorical("n_training_episodes", [100, 500, 1000, 1500])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, step=0.05)
        gamma = trial.suggest_float("gamma", 0.5, 0.99, step=0.05)

        # Ensure min_epsilon < max_epsilon
        max_epsilon = trial.suggest_categorical("max_epsilon", [0.5, 1.0])

        # Suggest min_epsilon within [0.1, max_epsilon - small_margin]
        min_epsilon = trial.suggest_float(
            "min_epsilon",
            0.1,
            max_epsilon - 0.05,
            step=0.05
        )

        decay_rate = trial.suggest_categorical("decay_rate", [0.0015, 0.0035, 0.005, 0.01])

        # Init env + agent
        env = TradingEnv(df, broker_fee=True)
        QL = QLearning(env)

        Qtable = QL.train(
            df=df,
            reward_type="portfolio_diff",
            reward_evolution="value",
            train_size=0.8,
            n_training_episodes=n_training_episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            max_epsilon=max_epsilon,
            min_epsilon=min_epsilon,
            decay_rate=decay_rate
        )

        actions_taken, prices, df_indices, equity_curve, reward_list = QL.get_actions_and_prices(Qtable=Qtable, df=df,reward_type="portfolio" ,reward_evolution="additive")
        score = equity_curve[-1]

        return score

    # Use persistent storage for multi-agent coordination
    storage_url = "sqlite:///QLearning_optimization.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="Qlearning_optimization",
        storage=storage_url,
        load_if_exists=True
    )

    # Worker function
    def run_worker():
        study.optimize(objective, n_trials=100, show_progress_bar=False)

    # Launch multiple workers in parallel
    n_workers = 8
    processes = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=run_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def get_best_param():
    # Connect to the same storage and study name used in optimization()
    storage = RDBStorage(
        url="sqlite:///QLearning_optimization.db"  # must match the storage used before
    )

    study = optuna.load_study(
        study_name="QLearning_optimization",
        storage=storage
    )

    print("Best trial:")
    print(study.best_trial)
    print("Best params:", study.best_trial.params)
    print("Best score:", study.best_trial.value)

    return study.best_trial.params, study.best_trial.value

if __name__ == "__main__":
    optimization()
    #get_best_param()