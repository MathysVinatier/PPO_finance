import os
import psutil
from typing import List
import sqlite3

import time
import platform

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns

import numpy as np
import io

from conf_PPO_training import *

# --- UTILS ---
def list_databases(directory:str = ".") -> List[str]:
    return [f for f in os.listdir(directory) if f.endswith(".db")]

def list_task() -> List[str]:
    list_task_folder = []
    for file in os.listdir(TASK_FOLDER):
        if file == "task_dummy":
            continue
        if file.split("_")[0] == "task":
            list_task_folder.append(file)

    return list_task_folder

def list_trial(task: str) -> List[str]:
    db_path = get_db_path(task)

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0].startswith("trial_")]
    conn.close()

    return tables

def list_model(task:str) -> List[str]:
    list_model_name = []
    model_folder = get_models_path(task)
    for fname in os.listdir(model_folder):
        split_name = fname.split("_")
        model_name = f"model_{split_name[-2]}_{split_name[-1]}"
        if model_name not in list_model_name:
            list_model_name.append(model_name)
    return list_model_name

def get_task_info(task:str) -> pd.DataFrame:
    task_path = get_task_path(task)
    info_file = os.path.join(task_path, TRAINING_INFO_PATH)
    return pd.read_csv(info_file)

def get_models_path(task:str) -> str:
    task_path = get_task_path(task)
    return os.path.join(task_path, MODELS_PATH)

def get_task_path(task:str) -> str:
    return os.path.join(TASK_FOLDER, task)

def get_db_path(task:str) -> str:
    return os.path.join(TASK_FOLDER, task, DB_PATH)


def get_df_training(path_db_to_connect:str, trial_name:str) -> pd.DataFrame:
    conn = sqlite3.connect(path_db_to_connect)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({trial_name})")
    columns_info = cursor.fetchall()
    column_names = [info[1] for info in columns_info]
    cursor.execute(f"SELECT * FROM {trial_name}")
    row = cursor.fetchall()
    conn.close()
    df_trial = pd.DataFrame(row, columns=column_names)
    return df_trial

def get_df_result(path_db_to_connect:str) -> pd.DataFrame:
    conn = sqlite3.connect(path_db_to_connect)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trials)")
    columns_info = cursor.fetchall()
    column_names = [info[1] for info in columns_info]
    cursor.execute("SELECT * FROM trials")
    row = cursor.fetchall()
    conn.close()
    df_trial = pd.DataFrame(row, columns=column_names)
    return df_trial

def plot_training(df, title="Training Plot"):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    vmin = np.percentile(df["mean_actor_loss"], 1)
    vmax = np.percentile(df["mean_actor_loss"], 99)

    scatter = sns.scatterplot(
        data=df,
        x="episode_number",
        y="final_reward",
        hue="mean_actor_loss",
        palette="coolwarm",
        hue_norm=(vmin, vmax),
        s=50,
        ax=ax1,
        legend=False
    )

    ax1.set_ylabel("Reward")
    ax1.grid(True)

    ax2 = ax1.twinx()
    sns.histplot(
        data=df,
        x="episode_number",
        weights="actions_buy",
        bins=df["episode_number"].nunique(),
        alpha=0.3,
        color="gray",
        multiple="layer",
        shrink=0.8,
        ax=ax2
    )
    ax2.set_ylabel("Number of Trades")

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Mean Actor Loss")

    plt.xlabel("Episode")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def get_system_stats():
    """
    Return detailed system stats:
    - CPU usage (total and per core)
    - RAM usage
    - Temperatures (if available)
    - System uptime
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    ram = psutil.virtual_memory().percent

    temps = {}
    try:
        if hasattr(psutil, "sensors_temperatures"):
            raw_temps = psutil.sensors_temperatures()
            for name, entries in raw_temps.items():
                temps[name] = [round(t.current, 1) for t in entries if t.current is not None]
    except Exception:
        temps = {}

    # Uptime
    boot_time = psutil.boot_time()
    uptime_seconds = int(time.time() - boot_time)
    uptime_hours = uptime_seconds // 3600
    uptime_minutes = (uptime_seconds % 3600) // 60

    return {
        "cpu_percent": cpu_percent,
        "per_cpu": per_cpu,
        "ram_percent": ram,
        "temps": temps,
        "uptime": f"{uptime_hours}h {uptime_minutes}m",
        "os": platform.system()
    }


def kill_all_main_processes():
    """Kill all main_PPO processes"""
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any("main" in part for part in cmdline):
                proc.kill()
        except Exception:
            continue