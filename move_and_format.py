import os
import shutil

def main(to_copy, to_paste):
    current_path = os.path.abspath(to_copy)
    list_trial = os.listdir(current_path)
    for trial in list_trial:
        try :
            trial_path = os.path.join(current_path, trial)
            for agent in os.listdir(trial_path):
                    number = agent.split("_")[-1]
                    type   = agent.split("_")[0]
                    agent_name = f"{type}_weight_001_trial_{number}"
                    agent_to_copy  = os.path.join(trial_path, agent)
                    agent_to_paste = os.path.join(to_paste, agent_name)
                    shutil.copy2(agent_to_copy, agent_to_paste)
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    from_path = "./optuna_optimization/trained_model/"
    to_path   = "./multitask_PPO/optuna_trial_1/data_training/trained_model"

    main(from_path, to_path)