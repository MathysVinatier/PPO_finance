import os
import sys
import time
from datetime import datetime

from sklearn.preprocessing import StandardScaler

import sqlite3

import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt

from main_PPO_conf import *

from PPO_Library import DataLoader, TradingEnv, ACAgent, ModelReport, ModelTest

def train_ppo(agent_id, df, n_episode, n_epoch_per_episode, batch_size, gamma, alpha, gae,
                 policy_clip,checkpoint_step=False, threshold = 0.65):

    tick = to_datetime(df.index).to_series().diff().mode()[0]
    print(f"... starting training from {df.index.min()} to {df.index.max()} with {tick} tick (threshold={threshold}) ...")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    create_trial_table(int(agent_id))
    features = df[["Close", "High", "Low", "Open", "Volume"]]

    scaler = StandardScaler()
    print("Scaling Data...")
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    print("Creating DB...")
    env = TradingEnv(df, broker_fee=True)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = 1*24*12
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print("Launching ACAgent...")
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=n_epoch_per_episode, gamma=gamma, alpha=alpha, gae_lambda=gae,
                 policy_clip=policy_clip, chkpt_dir=MODELS_PATH, agent_id=agent_id)

    # -----------------------------
    # Training loop
    # -----------------------------
    action_names = {0: "hold", 1: "buy", 2: "sell"}
    best_reward = 0

    print("Launching Episode...")
    for ep in range(n_episode):
        obs     = env.reset()
        done    = False
        total_reward    = 0
        seq_buffer      = []
        actions_taken   = []

        print("Launching Epoch...")
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
            total_reward = reward

            # Store experience
            print("Agent Remembering...")
            agent.remember(seq_array, action, log_prob, value, reward, done)
            obs = next_obs

        # Update agent after each episode
        agent.learn()
        agent.actor_scheduler.step()
        agent.critic_scheduler.step()

        # Count actions and map to names
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        if checkpoint_step!=False:
            if (ep+1)%checkpoint_step==0:
                agent.save_models(episode=f"{ep:04}")
            if total_reward >= best_reward:
                agent.save_models(episode=f"best_reward")
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

    df_train, df_test = DataLoader().split_train_test(DATASET_PATH, training_size=0.1)

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

            agent_trained.save_models(episode="latest")

            plot_folder = "trial_"+id_str
            plot_path   = os.path.join(PLOT_PATH, plot_folder)
            if plot_folder not in os.listdir(PLOT_PATH):
                os.mkdir(plot_path)

            plot_training_path  = os.path.join(plot_path, "train.png")
            plot_testing_path   = os.path.join(plot_path, "test.png")
            plot_analysis_path  = os.path.join(plot_path, "analysis.png")

            results_train = ModelTest(agent_trained, df_test)
            plot_train = results_train.plot()
            plot_train.savefig(plot_training_path)

            results_test = ModelTest(agent_trained, df_train)
            plot_test  = results_test.plot()
            plot_test.savefig(plot_testing_path)

            plot_analysis = ModelReport(TASK_FOLDER).plot()
            plot_analysis.savefig(plot_analysis_path)

            loss_actor_mean  = np.mean(agent_trained.all_actor_losses)
            loss_critic_mean = np.mean(agent_trained.all_critic_losses)

            save_trial(
                trained_reward    = results_train.info['total_reward'],
                loss_actor_mean   = loss_actor_mean,
                loss_critic_mean  = loss_critic_mean,
                tested_reward     = results_test.info["total_reward"],
                annual_return     = results_test.info["annual_return"],
                average_profit    = results_test.info["average_profit"],
                annual_volatility = results_test.info["annual_volatility"],
                sharpe_ratio      = results_test.info["sharpe_ratio"],
                max_drawdown      = results_test.info["max_drawdown"]
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