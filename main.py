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
    print(db_name)
    print(os.listdir("./"))
    if db_name in os.listdir("./"):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trial_id) FROM results")
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    return 0

def optimization2(fname):
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_on_reward_portefolio_slope_log.db"

    number_of_trial = 100
    last_trial = get_last_trial_id(db_name)
    if last_trial == 0 :
        trials = range(0, number_of_trial)
    else:
        trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, t, df, fname, db_name) for t in trials]
        for future in futures:
            _ = future.result()

def read_db(fname):
    conn = sqlite3.connect(fname)
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

def compute_drawdown(equity_curve):
    peak = np.maximum.accumulate(np.nan_to_num(equity_curve, nan=-np.inf))
    dd = (equity_curve - peak) / peak
    return dd


def optimization2_plot(df, fname, num_trials_to_show=None):
    if num_trials_to_show is None:
        num_trials_to_show = len(df)

    base_color = 'steelblue'

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    axes = axes.flatten()

    # --- Prepare equity matrix ---
    max_len = max(len(r) for r in df["equity"])
    padded_equity = [r + [np.nan] * (max_len - len(r)) for r in df["equity"]]
    equity_matrix = np.array(padded_equity, dtype=float)

    # --- Prepare reward matrix ---
    max_len = max(len(r) for r in df["reward"])
    padded_reward = [r + [np.nan] * (max_len - len(r)) for r in df["reward"]]
    reward_matrix = np.array(padded_reward, dtype=float)
    mean_reward = np.nanmean(reward_matrix, axis=0)
    std_reward = np.nanstd(reward_matrix, axis=0)

    # --- Compute mean equity ---
    mean_equity = np.nanmean(equity_matrix, axis=0)

    # --- Compute Drawdown ---
    drawdown_matrix = np.array([compute_drawdown(curve) for curve in equity_matrix])
    mean_drawdown = np.nanmean(drawdown_matrix, axis=0)
    arg_mmd = np.argmax(np.abs(mean_drawdown))
    mmd = mean_drawdown[arg_mmd]
    final_drawdowns = drawdown_matrix[:, -1]

    # =====================================================
    # Top-left: Equity heatmap
    # =====================================================
    std_equity = np.nanstd(equity_matrix, axis=0)
    axes[0].plot(mean_equity, color="steelblue", lw=2, label="Mean Equity")
    axes[0].fill_between(
        range(len(mean_equity)),
        mean_equity - std_equity,
        mean_equity + std_equity,
        color="blue", alpha=0.3, label="±1 Std Dev"
    )
    axes[0].set_title(f"Average Equity Curve ± Std (Max Drawdown={mmd:.2%})")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid()

    # =====================================================
    # Top-middle: Spaghetti equity curves
    # =====================================================
    for row in equity_matrix[:num_trials_to_show]:
        axes[1].plot(row, color=base_color, alpha=0.2)
    axes[1].plot(mean_equity, color="blue", lw=2, label="Mean Equity", alpha=0.5)
    axes[1].set_title(f"Equity Curves ({num_trials_to_show} trials)")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Equity")
    axes[1].legend()
    axes[1].grid()

    # =====================================================
    # Top-right: Drawdown heatmap
    # =====================================================
    sns.histplot(df["score"] - 100, bins=40, kde=True,
                 ax=axes[2], color="steelblue")
    axes[2].set_title("Distribution of Equity Profit Robustness")
    axes[2].set_xlabel("Equity Profit")
    axes[2].set_ylabel("Count")
    axes[2].grid()

    # =====================================================
    # Bottom-left: Reward curve
    # =====================================================
    axes[3].plot(mean_reward, color="darkgreen", lw=2, label="Mean Reward")
    axes[3].fill_between(
        range(len(mean_reward)),
        mean_reward - std_reward,
        mean_reward + std_reward,
        color="green", alpha=0.3, label="±1 Std Dev"
    )
    axes[3].set_title("Average Reward Curve ± Std")
    axes[3].set_xlabel("Time Step")
    axes[3].set_ylabel("Reward")
    axes[3].legend()
    axes[3].grid()

    # =====================================================
    # Bottom-middle: Spaghetti drawdown curves
    # =====================================================
    for row in drawdown_matrix[:num_trials_to_show]:
        axes[4].plot(row, color="red", alpha=0.15)
    axes[4].plot(mean_drawdown, color="black", lw=2, label="Mean Drawdown")
    axes[4].set_title(f"Drawdown Curves ({num_trials_to_show} trials)")
    axes[4].set_xlabel("Time Step")
    axes[4].set_ylabel("Drawdown (fraction)")
    axes[4].legend()
    axes[4].grid()

    # =====================================================
    # Bottom-right: Final drawdown distribution
    # =====================================================
    sns.histplot(final_drawdowns, bins=40, kde=True,
                 ax=axes[5], color="red")
    axes[5].set_title("Distribution of Final Drawdowns")
    axes[5].set_xlabel("Drawdown (fraction)")
    axes[5].set_ylabel("Count")
    axes[5].grid()

    title = fname.split(".")[0]
    title = " ".join(title.split("_"))
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fname = "AAPL_2010_2024.csv"
    #optimization2(fname=fname)

    fname_db = "results_AAPL_2010_2024_on_reward_portefolio_slope_log.db"
    df = read_db(fname=fname_db)
    optimization2_plot(df, fname = fname_db)