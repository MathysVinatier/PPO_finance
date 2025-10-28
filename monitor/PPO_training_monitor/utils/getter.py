import os
import sqlite3

import pandas as pd

from .config import TRAINING_INFO_PATH, MODELS_PATH, TASK_FOLDER, DB_PATH

def get_task_info(task:str) -> pd.DataFrame:
    task_path = get_task_path(task)
    info_file = os.path.join(task_path, TRAINING_INFO_PATH)
    return pd.read_csv(info_file)

def get_models_path(task:str) -> str:
    task_path = get_task_path(task)
    print(f"\n\nHERE{os.path.join(task_path, MODELS_PATH)}\n\n")
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
