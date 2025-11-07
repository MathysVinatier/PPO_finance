import os
from PPO_Library import ModelReport, ModelTest, DataLoader

def model_analysis(task_path, episode):
    task = ModelReport(task_path)
    # task.plot(show=True)#, save_path=f"episode_{episode}_analysis")
    df_train, df_test = DataLoader().split_train_test(task._dataset_path)

    model = task.get_model(model_episode=episode)
    test  = ModelTest(model, df_test)
    train = ModelTest(model, df_train)

    # train.plot(show=True)#, save_path=f"episode_{episode}_train")
    test.plot(show=True)#, save_path=f"episode_{episode}_test")

def show_best_ep_on_test(task_path):
    best_portfolio = 0
    best_episode   = dict()

    task = ModelReport(task_path)
    _, df_test = DataLoader().split_train_test(task._dataset_path)

    for mod in os.listdir(task._models_path):
        episode         = "_".join(mod.split("_")[3:])
        model_test      = ModelTest(task.get_model(episode), df_test)
        final_portfolio = model_test.info["portfolio_values"][-1]

        if final_portfolio > best_portfolio:
            best_episode = episode
            best_model = model_test
            best_portfolio = final_portfolio
            print(f"Best episode is {best_episode} with a {best_portfolio} dollars")

    model_analysis(task_path=task_path, episode=best_episode)
    return best_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="task value")
    args = parser.parse_args()

    task_path = f"multitask_PPO/{args.task}"
    show_best_ep_on_test(task_path=task_path)
    # episode = 19
    # task = model_analysis(task_path, episode)