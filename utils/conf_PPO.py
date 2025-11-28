import os

PATH_TRAINING = "tmp/ppo_training"
TASK_PATH = os.curdir
print(TASK_PATH)

# DataSet
DATASET_NAME   = "^VIX_2015_2025.csv"
DATASET_FOLDER = "data/General"
DATASET_PATH   = os.path.join(DATASET_FOLDER, DATASET_NAME)

# DataBase
DB_NAME     = "db_training.db"
DB_PATH     = os.path.join(PATH_TRAINING, DB_NAME)

# Model
MODELS_FOLDER = "trained_model"
MODELS_PATH   = os.path.join(PATH_TRAINING, MODELS_FOLDER)

# Info
TRAINING_INFO_FILE = "training_info.csv"
TRAINING_INFO_PATH = os.path.join(PATH_TRAINING, TRAINING_INFO_FILE)

# Plot
PLOT_FILE = "plot"
PLOT_PATH = os.path.join(PATH_TRAINING, PLOT_FILE)