from utils import DataLoader, TradingEnv,  QLearning, DeepQLearning, PPOAgent, ACAgent

import optuna
import multiprocessing

import os

import sqlite3
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler

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


def compute_function(trial_id: int, df, reward_type, reward_evolution) -> dict:

    env = TradingEnv(df, broker_fee=True)
    QL = QLearning(env, log=True)
    Qtable = QL.train(df=df, reward_type=reward_type, reward_evolution=reward_evolution, train_size=0.8, n_training_episodes=1500)
    _, _, _, equity, reward = QL.get_actions_and_prices(Qtable, df, reward_type, reward_evolution)
    result = equity[-1]
    return {
        "trial_id": trial_id, 
        "result": result,
        "equity": equity,
        "reward": reward,
        "qtable": Qtable
    }


def worker(trial_id: int, df, db_name, reward_type, reward_evolution):
    output = compute_function(trial_id, df, reward_type, reward_evolution)

    conn = sqlite3.connect(db_name, timeout=30)  # timeout helps with write locks
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS results (
               trial_id INTEGER, 
               result INTEGER, 
               equity TEXT, 
               reward TEXT,
               qtable TEXT
           )"""
    )

    # Convert Q-table to JSON (handle numpy arrays)
    if isinstance(output["qtable"], np.ndarray):
        qtable_json = json.dumps(output["qtable"].tolist())
    else:
        qtable_json = json.dumps(output["qtable"])

    cursor.execute(
        "INSERT INTO results (trial_id, result, equity, reward, qtable) VALUES (?, ?, ?, ?, ?)",
        (
            output["trial_id"],
            output["result"],
            json.dumps(output["equity"]),
            json.dumps(output["reward"]),
            qtable_json,
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

def optimization2(fname,reward_type ,reward_evolution):
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_on_reward_{reward_type}_{reward_evolution}.db"

    number_of_trial = 100
    last_trial = get_last_trial_id(db_name)
    if last_trial == 0 :
        trials = range(0, number_of_trial)
    else:
        trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, t, df, db_name, reward_type, reward_evolution) for t in trials]
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

def optimization2_optimized(fname,reward_type ,reward_evolution):
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_on_reward_{reward_type}_{reward_evolution}_optimized_parameters.db"

    number_of_trial = 100
    last_trial = get_last_trial_id(db_name)
    if last_trial == 0 :
        trials = range(0, number_of_trial)
    else:
        trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, t, df, db_name, reward_type, reward_evolution) for t in trials]
        for future in futures:
            _ = future.result()

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

def optimization_optimized_plot(fname):
    dataloader = DataLoader()
    df = dataloader.read("data/General/^VIX_2015_2025.csv")
    df_db = read_db(fname)
    df_db_best = df_db[df_db["score"] == df_db["score"].max()]

    df_equity = df_db_best["equity"].values[0]
    df_reward = df_db_best["reward"].values[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    training_size = int(0.8 * len(df_equity))

    training_equity = df_equity[:training_size]
    scaled_training_equity = [val - 100 + df["Close"].iloc[0] for val in training_equity]

    testing_equity = df_equity[training_size-1:]
    scaled_testing_equity = [val - testing_equity[0] + df["Close"].iloc[training_size-1] for val in testing_equity]

    x_train = df.index[:training_size]
    x_test  = df.index[training_size:training_size + len(scaled_testing_equity)]

    profit_train = (training_equity[-1]-training_equity[0])/training_equity[0]*100
    profit_test  = (testing_equity[-1]-testing_equity[0])/testing_equity[0]*100
    profit_index  = (df["Close"][-1]-df["Close"][0])/df["Close"][0]*100

    axes[0].plot(x_train, scaled_training_equity, label=f"Training Equity Curve ({profit_train:.2f}%)")
    axes[0].plot(x_test, scaled_testing_equity, label=f"Testing Equity Curve ({profit_test:.2f}%)")
    axes[0].plot(df.index, df["Close"].values, label=f"Close Price ({profit_index:.2f}%)", alpha=.5)
    axes[0].set_title("Equity Curve")
    axes[0].legend()
    axes[0].set_ylabel("Profit ($)")
    axes[0].grid(True)
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[0].tick_params(labelbottom=False)

    axes[1].plot(df.index[:-1], df_reward, label="Reward")
    axes[1].set_title("Reward")
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].grid(True)

    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def compute_DLTraining(trial_id: int, df, reward_type, reward_evolution) -> dict:

    env = TradingEnv(df, broker_fee=True)
    model = DeepQLearning(env, gpt=True,log=True)
    policy_net = model.train(df=df, train_size=0.8)
    _, _, _, equity, reward = model.get_actions_and_prices(policy_net, df, reward_type, reward_evolution)
    test_data_equity = equity[int(0.8*len(equity)):]
    result = (test_data_equity[-1]-test_data_equity[0])/test_data_equity[0]
    return {
        "trial_id": trial_id, 
        "result": result,
        "equity": equity,
        "reward": reward
    }

def worker_DeepLearning(trial_id: int, df, db_name, reward_type, reward_evolution):
    output = compute_DLTraining(trial_id, df, reward_type, reward_evolution)

    conn = sqlite3.connect(db_name, timeout=30)  # timeout helps with write locks
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS results (
               trial_id INTEGER, 
               result INTEGER, 
               equity TEXT, 
               reward TEXT
           )"""
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

def optimization3(fname, reward_type="portfolio_diff" ,reward_evolution="value"):
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_DeepQLearning.db"

    number_of_trial = 100
    last_trial = get_last_trial_id(db_name)
    if last_trial == 0 :
        trials = range(0, number_of_trial)
    else:
        trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_DeepLearning, t, df, db_name, reward_type, reward_evolution) for t in trials]
        for future in futures:
            _ = future.result()

def plot_PPO(df):

    plt.figure(figsize=(14, 6))
    train_value, test_value = DataLoader().split_train_test("./data/General/^VIX_2015_2025.csv")

    env_train = TradingEnv(train_value)
    ppo_train = PPOAgent(env=env_train, log=False)
    actions_train, equity_train = ppo_train.test()
 
    env_test = TradingEnv(test_value)
    ppo_test= PPOAgent(env=env_test, log=False)
    actions_test, equity_test = ppo_test.test()

    plt.subplot(2, 1, 1)

    list_train_buy = [i for i, action in enumerate(actions_train) if action == 1]
    list_train_sell = [i for i, action in enumerate(actions_train) if action == 2]

    buy_dates_train = train_value["Close"].index[list_train_buy]
    sell_dates_train = train_value["Close"].index[list_train_sell]

    buy_prices_train = train_value["Close"].values[list_train_buy]
    sell_prices_train = train_value["Close"].values[list_train_sell]

    plt.plot(train_value["Close"].index, train_value["Close"], label="Training Data", linewidth=1)
    plt.scatter(buy_dates_train, buy_prices_train, marker="^", c='green', s=22, zorder=3)
    plt.scatter(sell_dates_train, sell_prices_train, marker="v", c='red', s=22, zorder=3)

    list_test_buy = [i for i, action in enumerate(actions_test) if action == 1]
    list_test_sell = [i for i, action in enumerate(actions_test) if action == 2]

    buy_dates_test = test_value["Close"].index[list_test_buy]
    sell_dates_test = test_value["Close"].index[list_test_sell]

    buy_prices_test = test_value["Close"].values[list_test_buy]
    sell_prices_test = test_value["Close"].values[list_test_sell]

    plt.plot(test_value["Close"].index, test_value["Close"], label="Testing Data", linewidth=1)
    plt.scatter(buy_dates_test, buy_prices_test, marker="^", c='green', s=22, zorder=3)
    plt.scatter(sell_dates_test, sell_prices_test, marker="v", c='red', s=22, zorder=3)

    plt.title(f"Buy & Sell Signals on VIX in {df.index[0][:4]}")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    plt.xticks(rotation=45)
    plt.xlabel("Date")

    plt.subplot(2, 1, 2)

    x_train = range(len(equity_train))
    x_test = range(len(equity_train), len(equity_train) + len(equity_test))

    plt.plot(x_train, equity_train, label="Equity Curve Train", linewidth=1)
    plt.plot(x_test, equity_test, label="Equity Curve Test", linewidth=1)

    plt.title("Equity Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def compute_PPOTraining(trial_id: int, df) -> dict:

    env = TradingEnv(df=df, broker_fee=True)
    model = PPOAgent(env, log=True)

    model.train(n_games=100)
    _, equity, reward = model.test()
    test_data_equity = equity[int(0.8*len(equity)):]
    result = (test_data_equity[-1]-test_data_equity[0])/test_data_equity[0]
    return {
        "trial_id": trial_id, 
        "result": result,
        "equity": equity,
        "reward": reward
    }

def compute_PPOTesting(trial_id: int, df) -> dict:

    env = TradingEnv(df=df[int(0.8*len(df)):], broker_fee=True)
    model = PPOAgent(env, log=True)

    _, equity, reward = model.test(greedy=False)
    test_data_equity = equity
    result = (test_data_equity[-1]-test_data_equity[0])/test_data_equity[0]
    return {
        "trial_id": trial_id, 
        "result": result,
        "equity": equity,
        "reward": reward
    }

def worker_PPO(trial_id: int, df, db_name):
    output = compute_PPOTesting(trial_id, df)

    conn = sqlite3.connect(db_name, timeout=30)  # timeout helps with write locks
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS results (
               trial_id INTEGER, 
               result INTEGER, 
               equity TEXT, 
               reward TEXT
           )"""
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

def optimization_PPO(fname):
    data = "data/General/"+fname
    dataloader = DataLoader()
    df = dataloader.read(data)
    print(f"TRAINING ON {data}")
    db_name = f"results_{fname.split('.')[0]}_PPO.db"

    number_of_trial = 500
    last_trial = get_last_trial_id(db_name)
    if last_trial == 0 :
        trials = range(0, number_of_trial)
    else:
        trials = range(last_trial + 1, number_of_trial)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_PPO, t, df[:int(0.8*len(df))], db_name) for t in trials]
        for future in futures:
            _ = future.result()

def train_ppo(df, n_episode, n_epoch_per_episode, batch_size, checkpoint_step=False):

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    features = df[["Close", "High", "Low", "Open", "Volume"]]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = TradingEnv(df, broker_fee=False)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = 7
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=n_epoch_per_episode)

    # -----------------------------
    # Training loop
    # -----------------------------
    threshold = 0.65
    action_names = {0: "hold", 1: "buy", 2: "sell"}
    best_reward = 0

    for ep in range(n_episode):
        obs = env.reset()
        done = False
        total_reward = 0
        seq_buffer = []
        actions_taken = []

        while not done:
            # Build sequence for Transformer
            seq_buffer.append(obs)
            if len(seq_buffer) > seq_len:
                seq_buffer.pop(0)
            if len(seq_buffer) < seq_len:
                pad_len = seq_len - len(seq_buffer)
                seq = [seq_buffer[0]]*pad_len + seq_buffer
            else:
                seq = seq_buffer

            # Get valid actions
            valid_actions = env.get_valid_actions()

            seq_array = np.array(seq, dtype=np.float32)
            seq_array = np.expand_dims(seq_array, axis=0)  # Add batch dimension

            # Choose action
            action, log_prob, value = agent.choose_action(seq_array, valid_actions, threshold=threshold)
            actions_taken.append(action)  # <-- Save action

            # Step environment
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            # Store experience
            agent.remember(seq_array, action, log_prob, value, reward, done)
            obs = next_obs

        # Update agent after each episode
        agent.learn()

        # Count actions and map to names
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        if checkpoint_step!=False:
            if total_reward >= best_reward:
                agent.save_models()
                best_reward = total_reward

        print(f"Episode {ep+1}/{n_episode} finished, "
              f"total reward  : {total_reward:.3f}, "
              f"actions taken : {action_summary}")


def test_ppo(df):
    env = TradingEnv(df, broker_fee=False)

    seq_len = 7
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=1, n_epochs=1)
    agent.load_models()

    threshold = 0.65
    action_names = {0: "hold", 1: "buy", 2: "sell"}

    obs = env.reset()
    done = False
    total_reward = 0
    seq_buffer = []
    actions_taken = []
    portfolio_values = []
    probs_history = []

    while not done:
        seq_buffer.append(obs)
        if len(seq_buffer) > seq_len:
            seq_buffer.pop(0)
        if len(seq_buffer) < seq_len:
            pad_len = seq_len - len(seq_buffer)
            seq = [seq_buffer[0]] * pad_len + seq_buffer
        else:
            seq = seq_buffer

        valid_actions = env.get_valid_actions()
        seq_array = np.array(seq, dtype=np.float32)
        seq_array = np.expand_dims(seq_array, axis=0)

        # Forward pass for visualization
        with torch.no_grad():
            state = torch.tensor(seq_array, dtype=torch.float32).to(agent.actor.device)
            dist = agent.actor(state)
            probs = dist.probs.cpu().numpy().squeeze()
        probs_history.append(probs)

        # Choose action
        action, log_prob, value = agent.choose_action(seq_array, valid_actions, threshold=threshold)
        actions_taken.append(action)
        next_obs, reward, done, current_portfolio_value = env.step(action)
        total_reward += reward
        portfolio_values.append(current_portfolio_value)

        agent.remember(seq_array, action, log_prob, value, reward, done)
        obs = next_obs

    # --- Summary ---
    unique, counts = np.unique(actions_taken, return_counts=True)
    action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}
    print(f"total reward: {total_reward:.3f}, actions taken: {action_summary}")

    return actions_taken, probs_history, portfolio_values, total_reward

def plot_testppo(df, actions_taken, probs_history, portfolio_values):
    # --- Visualization ---
    prices = df["Close"].values[:len(actions_taken)]
    probs_history = np.array(probs_history)
    steps = np.arange(len(prices))

    fig, (ax_main, ax_probs) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1]})

    # === Price + Portfolio ===
    ax_main.plot(steps, prices, color="black", label="Market Price", linewidth=1.2)
    buy_idx = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_idx = [i for i, a in enumerate(actions_taken) if a == 2]
    ax_main.plot(buy_idx, prices[buy_idx], "^", color="tab:green", label="Buy", markersize=8)
    ax_main.plot(sell_idx, prices[sell_idx], "v", color="tab:red", label="Sell", markersize=8)
    ax_main.set_ylabel("Price")
    ax_main.set_title("Market Price & Portfolio Value")

    # twin y-axis for portfolio
    ax_portfolio = ax_main.twinx()
    ax_portfolio.plot(steps, portfolio_values, color="blue", label="Portfolio Value", linewidth=1.2, alpha=0.7)
    ax_portfolio.set_ylabel("Portfolio Value")

    # combine legends
    lines_1, labels_1 = ax_main.get_legend_handles_labels()
    lines_2, labels_2 = ax_portfolio.get_legend_handles_labels()
    ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    ax_main.grid(True, linestyle='--', alpha=0.6)

    # === Action probability bars ===
    width = 0.25
    #ax_probs.bar(steps - width, probs_history[:, 0], width, label="Hold", color="gray", alpha=0.7)
    ax_probs.bar(steps, probs_history[:, 1], width, label="Buy", color="tab:gray", alpha=0.5)
    ax_probs.bar(steps + width, probs_history[:, 2], width, label="Sell", color="tab:red", alpha=1)
    ax_probs.set_ylim(0, 1)
    ax_probs.set_xlabel("Step")
    ax_probs.set_ylabel("Probability")
    ax_probs.set_title("Action Probabilities per Step")
    ax_probs.legend()
    ax_probs.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    df_train, df_test = DataLoader().split_train_test("data/General/^VIX_2015_2025.csv")
    train_ppo(df_train, n_episode=150, n_epoch_per_episode=100, batch_size=128, checkpoint_step=True)
    actions_taken, probs_history, portfolio_values, total_reward = test_ppo(df_train)
    plot_testppo(df_train, actions_taken, probs_history, portfolio_values)
