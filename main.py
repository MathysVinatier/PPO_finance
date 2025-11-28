import os
from PPO_Library import ModelReport, ModelTest, DataLoader

def model_analysis(task_path, episode, seq_size):
    task = ModelReport(task_path, seq_size = seq_size)
    # task.plot(show=True)#, save_path=f"episode_{episode}_analysis")
    df_train, df_test = DataLoader().split_train_test(task._dataset_path, training_size=0.8)

    model = task.get_model(model_episode=episode)
    test  = ModelTest(model, df_test, seq_size)
    train = ModelTest(model, df_train, seq_size)

    train.plot(show=True)#, save_path=f"episode_{episode}_train")
    test.plot(show=True)#, save_path=f"episode_{episode}_test")

def show_best_ep_on_test(task_path, seq_size):
    best_portfolio = 0
    best_episode   = dict()

    task = ModelReport(task_path, seq_size)
    df_train, df_test = DataLoader().split_train_test(task._dataset_path, training_size=0.8)

    for mod in os.listdir(task._models_path):
        episode         = "_".join(mod.split("_")[3:])
        model_test      = ModelTest(task.get_model(episode), df_test)
        final_portfolio = model_test.info["portfolio_values"][-1]

        if final_portfolio > best_portfolio:
            best_episode = episode
            best_model = model_test
            best_portfolio = final_portfolio
            print(f"Best episode is {best_episode} with a {best_portfolio} dollars")

    model_analysis(task_path=task_path, episode=best_episode, seq_size=seq_size)
    return best_model


if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="task name")
    parser.add_argument("-p", "--path", help="task path")
    args = parser.parse_args()

    training_seq_path = os.path.join(args.path, "main_PPO_conf.py")

    spec = importlib.util.spec_from_file_location("training_seq", training_seq_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    task_path = os.path.join(args.path)
    model_analysis(task_path=task_path, episode=args.task, seq_size = module.TRAINING_SEQUENCE)
