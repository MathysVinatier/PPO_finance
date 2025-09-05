from utils import DataLoader, TradingEnv,  QLearning

import optuna
import multiprocessing

import os

import sqlite3
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

def test_QLearning(fdir, with_broker=False):

    dataloader = DataLoader()
    dataloader.save_companies()
    
    list_dir = os.listdir(fdir)
    
    for data in list_dir :
        df = dataloader.read(fdir+data)
        env = TradingEnv(df, broker_fee=with_broker)
        print(f"TRAINING ON {data}")
        QL = QLearning(env)
        Qtable = QL.train(df=df, train_size=0.8, n_training_episodes=1000)
        QL.plot(df=df, model=Qtable, train_size=0.8, name=data.split('_')[0], save=True, show=True)

def experiment1():

    start_date = 2010
    stop_date = 2024

    dir = './data/1yearData/'
    company = 'AAPL'
    year_list = [str(i) for i in range(start_date,stop_date)]
    data_list = [company+'_'+year_list[i]+'_'+year_list[i+1]+'.csv' for i in range(len(year_list)-1)]

    data_loader = DataLoader()

    list_dir = os.listdir(dir)

    for df_name in data_list :
        if df_name not in list_dir:
            _, start, stop = df_name[:-4].split('_')
            data_loader.save(df_full_path = dir, df_fname = company, start_date = f"{start}-01-01", end_date = f"{stop}-01-01")

    training_size = 0.8
    episode = 1000

    list_dir = os.listdir(dir)

    for file in list_dir:
        print(f"\n\n File => {file}")
        df = data_loader.read(dir+file)
        env = TradingEnv(df)
        QL = QLearning(env)

        Qtable_trained = QL.train(df, n_training_episodes=episode, train_size=training_size)
        QL.plot(df, Qtable_trained, training_size, save=True, show=True, name=company)

def experiment2():
    start_date = 2016
    stop_date = 2024

    dir = './data/10yearData/'
    company = 'RNO.PA'
    data_loader = DataLoader()
    list_dir = os.listdir(dir)
    fname = f"{company}_{start_date}_{stop_date}.csv"
    if fname not in list_dir:
        data_loader.save(df_full_path = dir, df_fname = company, start_date = f"{start_date}-01-01", end_date = f"{stop_date}-01-01")

    training_size = 0.8
    episode = 1000

    print(f"\n\n File => {fname}")
    print(f'Saving as {company.split('.')[0]}')
    df = data_loader.read(dir+fname)
    env = TradingEnv(df)
    QL = QLearning(env)

    Qtable_trained = QL.train(df, n_training_episodes=episode, train_size=training_size)
    QL.plot(df, Qtable_trained, training_size, save=True, show=True, name=company.split('.')[0])

def optimization1():
    dataloader = DataLoader()
    dataloader.save_companies()

    general_dir = './data/General/'
    list_dir = [os.path.join(general_dir, fname) for fname in os.listdir(general_dir)]

    fdir = "./data/General/O_2016_2024.csv"
    df  = dataloader.read(fdir)

    # Define the objective
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.1, 0.9, step=0.1)
        gamma = trial.suggest_float("gamma", 0.85, 0.99, step=0.01)
        decay_rate = trial.suggest_float("decay_rate", 0.0001, 0.01, log=True)

        # Randomly choose a dataset per trial
        #fdir = list_dir[trial.number%6]
        score_market = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        env = TradingEnv(df)
        QL  = QLearning(env, log=False)

        train_size = 0.8
        n_episode = 500

        Qtable = QL.train(
            df, train_size=train_size, n_training_episodes=n_episode,
            learning_rate=learning_rate, gamma=gamma,
            max_epsilon=1.0, min_epsilon=0.05, decay_rate=decay_rate
        )

        df_train, df_test = QL.split_data(df, train_size)
        _, _, _, equity_test, reward_test = QL.get_actions_and_prices(Qtable, df_test)
        score_equity = (equity_test[-1]-equity_test[0])/equity_test[0]
        score = score_equity - score_market
        print(f"TRIAL #{trial.number} -> market : {score_market} / equity : {score_equity} / score {score}")
        return score

    # Use persistent storage for multi-agent coordination
    storage_url = "sqlite:///Qlearning_optimization.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="QLearning_Study",
        storage=storage_url,
        load_if_exists=True
    )

    # Worker function
    def run_worker():
        study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Run in parallel
    n_workers = 6
    processes = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=run_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def get_best_parameter(storage_url):
    study = optuna.create_study(
            direction="maximize",
            study_name="QLearning_Study",
            storage=storage_url,
            load_if_exists=True
        )
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)


def compute_function(trial_id: int, df) -> dict:

    env = TradingEnv(df, broker_fee=True)
    QL = QLearning(env, log=True)
    Qtable = QL.train(df=df, train_size=0.8, n_training_episodes=1000)
    _, _, _, equity, reward = QL.get_actions_and_prices(Qtable, df)
    result = equity[-1]
    return {
        "trial_id": trial_id,
        "result": result,
        "equity": equity,
        "reward": reward
    }


def worker(trial_id: int, df, fname, db_name):
    output = compute_function(trial_id, df)

    conn = sqlite3.connect(db_name, timeout=30)  # timeout helps with write locks
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS results (trial_id INTEGER, result INTEGER, equity TEXT, reward TEXT)"
    )
    cursor.execute(
        "INSERT INTO results (trial_id, result, equity, reward) VALUES (?, ?, ?, ?)",
        (
            output["trial_id"],
            output["result"],
            json.dumps(output["equity"]),
            json.dumps(output["reward"])
        ),
    )
    conn.commit()
    conn.close()
    return output

def get_last_trial_id(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(trial_id) FROM results")
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return 0

def optimization2():
    fname = "AAPL_2010_2024.csv"
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_on_reward_portefolio.db"

    number_of_trial = 100
    last_trial = get_last_trial_id(db_name)
    trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, t, df, fname, db_name) for t in trials]
        for future in futures:
            _ = future.result()

def read_db():
    conn = sqlite3.connect("results_AAPL_2010_2024_on_reward_portefolio.db")
    cursor = conn.cursor()
    cursor.execute("SELECT trial_id, result, equity, reward FROM results")
    rows = cursor.fetchall()
    conn.close()

    records = []
    for trial_id, result, equity, reward in rows:
        equity_list = json.loads(equity)
        reward_list = json.loads(reward)
        records.append({
            "id": trial_id,
            "score": result,
            "equity": equity_list,
            "reward": reward_list
        })
        #print(trial_id, result, equity_list[:5], reward_list[:5])

    df = pd.DataFrame(records).set_index("id")
    return df

def compute_max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()  # worst drawdown
    return max_dd, drawdowns

def optimization2_plot(df, num_trials_to_show=None):
    if num_trials_to_show == None :
        num_trials_to_show = len(df)

    base_color='steelblue'

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes = axes.flatten()

    # --- Prepare equity matrix ---
    max_len = max(len(r) for r in df["equity"])
    padded_equity = [r + [np.nan]*(max_len - len(r)) for r in df["equity"]]
    equity_matrix = np.array(padded_equity, dtype=float)

    # --- Compute global last value ---
    mean_equity = np.nanmean(equity_matrix, axis=0)

    # Take the last value of the mean equity
    mean_last_value = mean_equity[-1]

    # Compute distance of each trial's last equity to the mean last value
    trial_last_values = equity_matrix[:, -1]
    distances = np.abs(trial_last_values - mean_last_value)

    # Optional: log scale for intensity
    epsilon = 1e-10
    log_distances = (distances + epsilon)
    norm_log = (log_distances - log_distances.min()) / (log_distances.max() - log_distances.min() + 1e-9)
    alphas = 1.0 - 0.8 * norm_log

    # --- Plot individual equity curves with intensity ---
    for i, row in enumerate(equity_matrix[:num_trials_to_show]):
        axes[1].plot(row, color=base_color, alpha=alphas[i])
    axes[1].set_title(f"Equity Curves ({num_trials_to_show} curves)")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Equity")
    axes[1].grid()

    # --- Top-left: Average Equity Curve ± Std ---
    std_equity = np.nanstd(equity_matrix, axis=0)
    axes[0].plot(mean_equity, color="darkorange", lw=2, label="Mean Equity")
    axes[0].fill_between(range(len(mean_equity)),
                         mean_equity - std_equity,
                         mean_equity + std_equity,
                         color="orange", alpha=0.3, label="±1 Std Dev")
    mdd, _ = compute_max_drawdown(mean_equity)
    axes[0].set_title(f"Average Equity Curve ± Std (Max Drawdown={mdd:.2%})")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid()

    # --- Bottom-left: Average Reward Curve ± Std ---
    max_len = max(len(r) for r in df["reward"])
    padded_reward = [r + [np.nan]*(max_len - len(r)) for r in df["reward"]]
    reward_matrix = np.array(padded_reward, dtype=float)
    mean_reward = np.nanmean(reward_matrix, axis=0)
    std_reward = np.nanstd(reward_matrix, axis=0)
    axes[2].plot(mean_reward, color="darkgreen", lw=2, label="Mean Reward")
    axes[2].fill_between(range(len(mean_reward)),
                         mean_reward - std_reward,
                         mean_reward + std_reward,
                         color="green", alpha=0.3, label="±1 Std Dev")
    axes[2].set_title("Average Reward Curve ± Std")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Reward")
    axes[2].legend()
    axes[2].grid()

    # --- Bottom-right: Distribution of final equity ---
    sns.histplot(df["score"], bins=20, kde=True, ax=axes[3], color="steelblue")
    axes[3].set_title("Distribution of Final Equity (Model Robustness)")
    axes[3].set_xlabel("Final Equity / Score")
    axes[3].set_ylabel("Count")
    axes[3].grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    optimization2()

    #df = read_db()
    #optimization2_plot(df)
