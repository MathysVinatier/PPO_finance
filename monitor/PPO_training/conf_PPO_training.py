import os

TASK_FOLDER = "/home/mathys/Documents/PPO_finance/multitask_PPO"

# Base training path
TRAINING_FOLER = "data_training"
TRAINING_PATH  = os.path.join(TRAINING_FOLER)

# Dataset
DATASET_NAME   = "^VIX_2015_2025.csv"
DATASET_FOLDER = "/home/mathys/Documents/PPO_finance/data/General"
DATASET_PATH   = os.path.join(DATASET_FOLDER, DATASET_NAME)

# Database
DB_NAME = "db_training.db"
DB_PATH = os.path.join(TRAINING_PATH, DB_NAME)

# Models
MODELS_FOLDER = "trained_model"
MODELS_PATH   = os.path.join(TRAINING_FOLER, MODELS_FOLDER)

# Training info
TRAINING_INFO_FILE = "training_info.csv"
TRAINING_INFO_PATH = os.path.join(TRAINING_FOLER, TRAINING_INFO_FILE)

# Plot
PLOT_FOLDER = "plot"
PLOT_PATH   = os.path.join(TRAINING_FOLER,PLOT_FOLDER)
