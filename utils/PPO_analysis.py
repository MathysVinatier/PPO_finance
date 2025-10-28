import os
import numpy as np
import pandas as pd
import csv

if __name__ == "__main__":
    from PPO_Training_conf import *
    from Environment import TradingEnv, DataLoader
    from PPO import ACAgent
else:
    from utils.PPO_Training_conf import *
    from utils.PPO import ACAgent
    from utils.Environment import TradingEnv, DataLoader


class _ModelFormat:

    def __init__(self):
        pass

    def __get_agent_trial__(self, model_trial):
        if type(model_trial) == int :
            model_trial = f"{model_trial:03}"
        return model_trial

    def __get_agent_episode__(self, model_episode):
        if type(model_episode) == int :
            model_episode = f"{model_episode:04}"
        return model_episode
    
    def __get_agent_id__(self, model_trial, model_episode):
        trial   = self.__get_agent_trial__(model_trial=model_trial)
        episode = self.__get_agent_episode__(model_episode=model_episode)
        id = f"{trial}_{episode}"
        return id

    def __get_actor_format__(self, model_trial, model_episode):
        id = self.__get_agent_id__(model_trial=model_trial, model_episode=model_episode)
        actor_format = f"actor_weight_{id}"
        return actor_format

    def __get_critic_format__(self, model_trial, model_episode):
        id = self.__get_agent_id__(model_trial=model_trial, model_episode=model_episode)
        critic_format = f"critic_weight_{id}"
        return critic_format
    def __make_output_tabular__(self, info_line, tab_title):

        # --- Dynamically compute column widths ---
        key_width   = max(len(k) for k in info_line.keys())
        val_width   = max(len(str(v)) for v in info_line.values())
        total_width = key_width + val_width + 7  # padding + borders

        # --- header ---
        print("+" + "-" * (total_width - 2) + "+")
        print("|" + tab_title.center(total_width - 2) + "|")
        print("+" + "-" * (total_width - 2) + "+")

        # --- Table rows ---
        for key, value in info_line.items():
            print(f"| {key:<{key_width}} | {str(value):<{val_width}} |")

        # --- Footer ---
        print("+" + "-" * (total_width - 2) + "+")

class ModelReport(_ModelFormat):
    def __init__(self, task_path):
        super().__init__()

        # -------- Private --------

        # -------- Public --------
        # DIRECTORIES
        self._models_path   = os.path.join(task_path, MODELS_PATH)
        self._info_training = os.path.join(task_path, TRAINING_INFO_PATH)
        self._plot_path     = os.path.join(task_path, PLOT_PATH)
        self._dataset_path  = DATASET_PATH
        self._model_file    = ""

        # FILE
        self._db = os.path.join(task_path, DB_PATH)
        self._list_models       = os.listdir(self._models_path)
        self._current_agent_id  = self.__get_agent_id__(model_trial=1, model_episode="latest")

        self.agent  = ACAgent
        self.info   = self.get_info()

        self.show_info()

    def get_info(self):
        data_list = list()
        with open(self._info_training, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row not in data_list:
                    data_list.append(row)
        return data_list[0]

    def get_model(self, model_episode="latest", model_trial="001"):
        self.__check_model_dir(model_trial=model_trial, model_episode=model_episode)
        if not self.__check_current_model_version(model_id=model_episode, model_trial=model_trial):
            seq_len = TRAINING_SEQ
            agent_trial = self.__get_agent_trial__(model_trial=model_trial)

            agent = ACAgent(
                n_actions    = ACTION_SPACE,
                num_features = INPUT_SIZE,
                seq_len      = seq_len,
                batch_size   = 1,
                n_epochs     = 1,
                chkpt_dir    = self._models_path,
                agent_id     = agent_trial
            )

            agent_episode = self.__get_agent_episode__(model_episode=model_episode)
            agent.load_models(agent_episode)
            self.agent              = agent
            self._current_agent_id  = self.__get_agent_id__(model_trial=agent_trial, model_episode=agent_episode)
        return self.agent

    def show_info(self):
        info_dict = {
            "Epochs"        : self.info["n_epoch"],
            "Episodes"      : self.info["n_episode"],
            "Batch size"    : self.info["batch_size"],
            "Gamma"         : self.info["gamma"],
            "Learning rate" : self.info["lr"],
            "GAE"           : self.info["gae"],
            "Policy clip"   : self.info["policy_clip"],
            "Dataset path"  : DATASET_PATH,
            "Action space"  : ACTION_SPACE,
            "Input size"    : INPUT_SIZE,
            "Training seq"  : TRAINING_SEQ,
        }

        title = f"TRAINING INFO  (Agent {self._current_agent_id})"

        self.__make_output_tabular__(info_line=info_dict, tab_title=title)

    def __check_model_dir(self, model_trial, model_episode):
        actor_id  = self.__get_actor_format__(model_trial=model_trial, model_episode=model_episode)
        critic_id = self.__get_critic_format__(model_trial=model_trial, model_episode=model_episode)

        if actor_id not in self._list_models:
            raise FileExistsError(f"Agent {actor_id} does not exist")

        if critic_id not in self._list_models:
            raise FileExistsError(f"Agent {critic_id} does not exist")

        return 0

    def __check_current_model_version(self, model_trial, model_id):
        agent_id  = self.__get_agent_id__(model_trial=model_trial, model_episode=model_id)
        print(agent_id)
        print(self._current_agent_id)
        if agent_id == self._current_agent_id:
            return True
        else: return False

class ModelTest:

    def __init__(self, model, df):
        self.model = model
        self.df    = df
        self.env   = TradingEnv(df, broker_fee=True)
        self._launch_test()

    def _launch_test(self):
        return

if __name__ == "__main__":
    model = ModelReport('/home/mathys/Documents/PPO_finance/multitask_PPO/task_0')
    # model.show_info()