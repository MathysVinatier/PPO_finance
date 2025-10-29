import os
import sys
import time
from datetime import datetime

from sklearn.preprocessing import StandardScaler
import torch

import sqlite3

import numpy as np
import matplotlib.pyplot as plt

from conf_PPO import *

current_dir = os.getcwd()

with os.scandir("/home/mathys/Documents/PPO_finance"):
    os.chdir("/home/mathys/Documents/PPO_finance")
    sys.path.insert(0, os.getcwd())
    from PPO_Library import DataLoader, TradingEnv, ACAgent

os.chdir(current_dir)

LOG_FOLDER = os.path.join(current_dir, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(LOG_FOLDER, f"training.log")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout


def train_ppo(agent_id, df, n_episode, n_epoch_per_episode, batch_size, gamma, alpha, gae,
                 policy_clip,checkpoint_step=False):

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    create_trial_table(int(agent_id))
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
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=n_epoch_per_episode, gamma=gamma, alpha=alpha, gae_lambda=gae,
                 policy_clip=policy_clip, chkpt_dir=MODELS_PATH, agent_id=agent_id)

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
            actions_taken.append(action)

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
            if (ep+1)%checkpoint_step==0:
                agent.save_models(episode=f"_{ep:04}")
            if total_reward >= best_reward:
                agent.save_models(episode=f"_best_reward")
                print(f"Reward is {total_reward}")
                best_reward = total_reward

        print(f"Episode {ep+1}/{n_episode} finished, "
              f"total reward  : {total_reward:.3f}, "
              f"actions taken : {action_summary}")

        save_episode_to_trial(
            trial_id=int(agent_id),
            episode_number=ep + 1,
            mean_actor_loss=np.mean(agent.all_actor_losses),
            mean_critic_loss=np.mean(agent.all_critic_losses),
            actions_dict=action_summary,
            final_reward=total_reward
        )

    return agent, total_reward


def test_ppo(agent_id, df, trading_days_per_year=252):
    env = TradingEnv(df, broker_fee=False)

    seq_len = 7
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(
        n_actions=n_actions,
        num_features=num_features,
        seq_len=seq_len,
        batch_size=1,
        n_epochs=1,
        chkpt_dir=MODELS_PATH,
        agent_id=agent_id
    )
    agent.load_models(episode="_latest")

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
        seq = seq_buffer if len(seq_buffer) == seq_len else [seq_buffer[0]]*(seq_len-len(seq_buffer)) + seq_buffer

        valid_actions = env.get_valid_actions()
        seq_array = np.array(seq, dtype=np.float32)[None, ...]  # batch dimension

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

    # --- Trading metrics ---
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Annualized return
    total_period = len(df)
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    annual_return = (1 + total_return) ** (trading_days_per_year / total_period) - 1

    # Average daily profit
    avg_profit = np.mean(returns)

    # Annualized volatility
    annual_vol = np.std(returns) * np.sqrt(trading_days_per_year)

    # Sharpe ratio (risk-free rate = 0)
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(trading_days_per_year) if np.std(returns) != 0 else 0

    # Max drawdown
    cumulative = portfolio_values / portfolio_values[0]
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_drawdown = drawdowns.min()

    # --- Summary ---
    unique, counts = np.unique(actions_taken, return_counts=True)
    action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}
    print(f"Total reward: {total_reward:.3f}, actions: {action_summary}")
    print(f"Annual return: {annual_return:.3f}, Avg profit: {avg_profit:.3f}, "
          f"Volatility: {annual_vol:.3f}, Sharpe: {sharpe_ratio:.3f}, Max DD: {max_drawdown:.3f}")

    return {
        "actions_taken": actions_taken,
        "probs_history": probs_history,
        "portfolio_values": portfolio_values,
        "total_reward": total_reward,
        "annual_return": annual_return,
        "average_profit": avg_profit,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

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
    return fig


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            trained_reward REAL,
            loss_actor_mean REAL,
            loss_critic_mean REAL,
            tested_reward REAL,
            annual_return REAL,
            average_profit REAL,
            annual_volatility REAL,
            sharpe_ratio REAL,
            max_drawdown REAL
        )
    """)
    conn.commit()
    conn.close()

def create_trial_table(trial_id):
    """Create a table named 'trial_<id>' to store episode stats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table_name = f"trial_{trial_id}"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            episode_number INTEGER PRIMARY KEY,
            mean_actor_loss REAL,
            mean_critic_loss REAL,
            actions_buy INTEGER,
            actions_hold INTEGER,
            actions_sell INTEGER,
            final_reward REAL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Created table {table_name}")

def save_episode_to_trial(trial_id, episode_number, mean_actor_loss, mean_critic_loss, actions_dict, final_reward):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table_name = f"trial_{trial_id}"

    cursor.execute(f"""
        INSERT INTO {table_name} (
            episode_number, mean_actor_loss, mean_critic_loss,
            actions_buy, actions_hold, actions_sell, final_reward
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_number,
        float(mean_actor_loss),
        float(mean_critic_loss),
        int(actions_dict.get("buy", 0)),
        int(actions_dict.get("hold", 0)),
        int(actions_dict.get("sell", 0)),
        float(final_reward)
    ))
    conn.commit()
    conn.close()

def get_last_trial_id():
    """Return last trial ID or 0 if none exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM trials")
    result = cursor.fetchone()[0]
    conn.close()
    return result if result is not None else 0

def save_trial(
    trained_reward, loss_actor_mean, loss_critic_mean,
    tested_reward, annual_return, average_profit,
    annual_volatility, sharpe_ratio, max_drawdown
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trials (
            timestamp,
            trained_reward,
            loss_actor_mean,
            loss_critic_mean,
            tested_reward,
            annual_return,
            average_profit,
            annual_volatility,
            sharpe_ratio,
            max_drawdown
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            float(trained_reward),
            float(loss_actor_mean),
            float(loss_critic_mean),
            float(tested_reward),
            float(annual_return),
            float(average_profit),
            float(annual_volatility),
            float(sharpe_ratio),
            float(max_drawdown)
        )
    )
    conn.commit()
    conn.close()
    print(f"[DB] Saved trial: trained={trained_reward:.3f}, tested={tested_reward:.3f}")


def main(n_episode, n_epoch, batch_size, gamma, alpha, gae, policy_clip, chckpt, n_trial):

    init_db()

    start_id = get_last_trial_id() + 1
    print(f"Starting from trial ID {start_id}")

    df_train, df_test = DataLoader().split_train_test(DATASET_PATH)

    try:
        for id in range(start_id, start_id + n_trial):

            id_str = f"{id:03}"

            agent_trained, reward = train_ppo(
                agent_id = id_str,
                df=df_train,
                n_episode=n_episode,
                n_epoch_per_episode=n_epoch,
                batch_size=batch_size,
                gamma=gamma,
                alpha=alpha,
                gae=gae,
                policy_clip=policy_clip,
                checkpoint_step=chckpt
            )

            agent_trained.save_models(episode="_latest")

            plot_folder = "trial_"+id_str
            plot_path   = os.path.join(PLOT_PATH, plot_folder)
            if plot_folder not in os.listdir(PLOT_PATH):
                os.mkdir(plot_path)

            plot_training_path = os.path.join(plot_path, "train.png")
            plot_testing_path  = os.path.join(plot_path, "test.png")

            results_train = test_ppo(agent_id=id_str, df=df_train)
            plot_train = plot_testppo(df_train, results_train['actions_taken'], results_train['probs_history'], results_train['portfolio_values'])
            plot_train.savefig(plot_training_path)

            results_test = test_ppo(agent_id=id_str, df=df_test)
            plot_test  = plot_testppo(df_test, results_test['actions_taken'], results_test['probs_history'], results_test['portfolio_values'])
            plot_test.savefig(plot_testing_path)

            loss_actor_mean  = np.mean(agent_trained.all_actor_losses)
            loss_critic_mean = np.mean(agent_trained.all_critic_losses)

            save_trial(
                trained_reward    = results_train['total_reward'],
                loss_actor_mean   = loss_actor_mean,
                loss_critic_mean  = loss_critic_mean,
                tested_reward     = results_test["total_reward"],
                annual_return     = results_test["annual_return"],
                average_profit    = results_test["average_profit"],
                annual_volatility = results_test["annual_volatility"],
                sharpe_ratio      = results_test["sharpe_ratio"],
                max_drawdown      = results_test["max_drawdown"]
            )

            print(f"Saved trial {id}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStoping training")

if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Training configuration")

    # Training arguments
    parser.add_argument("--epoch"       , type=int, required=True, help="Number of epochs")
    parser.add_argument("--episode"     , type=int, required=True, help="Number of episodes")
    parser.add_argument("--batch_size"  , type=int, required=True, help="Number of batch size")

    # Hyperparameters
    parser.add_argument("--gamma"       , type=float, default=0.99  , help="Discount factor for rewards")
    parser.add_argument("--lr"          , type=float, default=0.0003, help="Learning rate for optimizer")
    parser.add_argument("--gae"         , type=float, default=0.95  , help="GAE (Generalized Advantage Estimation) lambda")
    parser.add_argument("--policy_clip" , type=float, default=0.2   , help="Clipping value for PPO policy update")
    parser.add_argument("--trial"       , type=int  , default=1     , help="Number of unique agent")

    args = parser.parse_args()

    csv_training_info = {
        "n_epoch"       : args.epoch,
        "n_episode"     : args.episode,
        "batch_size"    : args.batch_size,
        "gamma"         : args.gamma,
        "lr"            : args.lr,
        "gae"           : args.gae,
        "policy_clip"   : args.policy_clip
    }

    with open(TRAINING_INFO_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_training_info.keys())
        writer.writeheader()
        writer.writerow(csv_training_info)

    main(
        n_episode   = args.episode,
        n_epoch     = args.epoch,
        batch_size  = args.batch_size,
        gamma       = args.gamma,
        alpha       = args.lr,
        gae         = args.gae,
        policy_clip = args.policy_clip,
        chckpt      = args.episode//5,
        n_trial     = args.trial
    )