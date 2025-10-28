import os
from typing import List
import sqlite3

from .config import TASK_FOLDER
from .getter import get_db_path, get_models_path

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
