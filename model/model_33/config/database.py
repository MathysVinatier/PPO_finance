from datetime import datetime
import sqlite3
import json
import os
from config import *

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            trial_name TEXT UNIQUE,
            params TEXT,
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

def create_trial_table(trial_name):
    """Create a table named 'trial_<trial_name>' to store episode stats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table_name = trial_name
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

def save_episode_to_trial(trial_name, episode_number, mean_actor_loss, mean_critic_loss, actions_dict, final_reward):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table_name = trial_name

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

def save_trial(
    trial_name,
    params,
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
            trial_name,
            params,
            trained_reward,
            loss_actor_mean,
            loss_critic_mean,
            tested_reward,
            annual_return,
            average_profit,
            annual_volatility,
            sharpe_ratio,
            max_drawdown
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            trial_name,
            json.dumps(params),
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
    print(f"[DB] Saved trial {trial_name} (trained={trained_reward:.3f}, tested={tested_reward:.3f})")
