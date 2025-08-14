from utils.Environment import DataLoader, TradingEnv
from utils.RLMethod import QLearning

import optuna
import numpy as np
from tqdm import tqdm
import multiprocessing

import os

def experiment1():

    start_date = 2010
    stop_date = 2024

    dir = './data/1yearData/'
    company = 'AAPL'
    year_list = [str(i) for i in range(start_date,stop_date)]
    data_list = [company+'_'+year_list[i]+'_'+year_list[i+1]+'.csv' for i in range(len(year_list)-1)]

    data_loader = DataLoader()

    list_dir = os.listdir(dir)

    for df_name in data_list :
        if df_name not in list_dir:
            _, start, stop = df_name[:-4].split('_')
            data_loader.save(df_full_path = dir, df_fname = company, start_date = f"{start}-01-01", end_date = f"{stop}-01-01")

    training_size = 0.8
    episode = 1000

    list_dir = os.listdir(dir)

    for file in list_dir:
        print(f"\n\n File => {file}")
        df = data_loader.read(dir+file)
        env = TradingEnv(df)
        QL = QLearning(env)

        Qtable_trained = QL.train(df, n_training_episodes=episode, train_size=training_size)
        QL.plot(df, Qtable_trained, training_size, save=True, show=True, name=company)

def experiment2():
    start_date = 2016
    stop_date = 2024

    dir = './data/10yearData/'
    company = 'RNO.PA'
    data_loader = DataLoader()
    list_dir = os.listdir(dir)
    fname = f"{company}_{start_date}_{stop_date}.csv"
    if fname not in list_dir:
        data_loader.save(df_full_path = dir, df_fname = company, start_date = f"{start_date}-01-01", end_date = f"{stop_date}-01-01")

    training_size = 0.8
    episode = 1000

    print(f"\n\n File => {fname}")
    print(f'Saving as {company.split('.')[0]}')
    df = data_loader.read(dir+fname)
    env = TradingEnv(df)
    QL = QLearning(env)

    Qtable_trained = QL.train(df, n_training_episodes=episode, train_size=training_size)
    QL.plot(df, Qtable_trained, training_size, save=True, show=True, name=company.split('.')[0])



def optimization1():
    dataloader = DataLoader()
    dataloader.save_companies()

    general_dir = './data/General/'
    list_dir = [os.path.join(general_dir, fname) for fname in os.listdir(general_dir)]

    # Define the objective
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.1, 0.9, step=0.1)
        gamma = trial.suggest_float("gamma", 0.85, 0.99, step=0.01)
        decay_rate = trial.suggest_float("decay_rate", 0.0001, 0.01, log=True)

        scores = []

        for fdir in tqdm(list_dir, desc=f"Trial {trial.number} Training", leave=False):
            df  = dataloader.read(fdir)
            env = TradingEnv(df)
            QL  = QLearning(env, log=False)

            train_size = 0.8
            n_episode = 500

            Qtable = QL.train(
                df, train_size=train_size, n_training_episodes=n_episode,
                learning_rate=learning_rate, gamma=gamma, max_epsilon=1.0,
                min_epsilon=0.05, decay_rate=decay_rate
            )

            df_train, df_test = QL.split_data(df, train_size)
            _, _, _, equity_test = QL.get_actions_and_prices(Qtable, df_test)
            if equity_test:
                scores.append(equity_test[-1])

        return np.mean(scores) if scores else 0

    # Use persistent storage for multi-agent coordination
    storage_url = "sqlite:///Qlearning_optimization.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="QLearning_Study",
        storage=storage_url,
        load_if_exists=True
    )

    # Number of parallel workers (agents)
    n_workers = multiprocessing.cpu_count() - 4  # You can adjust

    # Each worker will request new trials from the same DB
    def run_worker():
        study.optimize(objective, n_trials=50)

    processes = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=run_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    optimization1()