import os

#--------- Params ---------
N_TRIALS = 100
N_WORKERS = 3
N_JOBS_PER_WORKER = 1


#--------- Folders ---------
TASK_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATASET_NAME   = "^VIX_2015_2025.csv"
DATASET_FOLDER = "/home/mathys/Documents/PPO_finance/data/General"
DATASET_PATH   = os.path.join(DATASET_FOLDER, DATASET_NAME)

# Models
MODELS_FOLDER = "trained_model"
MODELS_PATH   = os.path.join(TASK_FOLDER, MODELS_FOLDER)

# Database
OPTUNA_DB_PATH   = "sqlite:////home/mathys/Documents/PPO_finance/optuna_optimization/PPO_optuna.db"
TRAINING_DB_PATH = "PPO_training.db"
DB_PATH          = os.path.join(MODELS_PATH, TRAINING_DB_PATH)

# Training info
TRAINING_INFO_FILE = "training_info.csv"
TRAINING_INFO_PATH = os.path.join(TASK_FOLDER, TRAINING_INFO_FILE)

for path in [MODELS_PATH]:
    os.makedirs(path, exist_ok=True)