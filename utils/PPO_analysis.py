import os

if __name__ == "__main__":
    from PPO_Training_conf import *
else:
    from .PPO_Training_conf import *

class ModelReport:
    def __init__(self, task_path):
        self._models_path   = os.path.join(task_path, MODELS_PATH)
        self._info_training = os.path.join(task_path, TRAINING_INFO_PATH)
        self._plot_path     = os.path.join(task_path, PLOT_PATH)
        self._dataset_path  = DATASET_PATH

        self._db = os.path.join(task_path, DB_PATH)

    def get_models(self):
        os.path()

if __name__ == "__main__":
    print(ModelReport('/home/mathys/Documents/PPO_finance/multitask_PPO/task_0')._info_training)