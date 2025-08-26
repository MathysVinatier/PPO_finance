from utils import DataLoader, TradingEnv,  QLearning

import optuna
import multiprocessing

import os

def test_QLearning(fdir, with_broker=False):

    dataloader = DataLoader()
    dataloader.save_companies()
    
    list_dir = os.listdir(fdir)
    
    for data in list_dir :
        df = dataloader.read(fdir+data)
        env = TradingEnv(df, broker_fee=with_broker)
        print(f"TRAINING ON {data}")
        QL = QLearning(env)
        Qtable = QL.train(df=df, train_size=0.8, n_training_episodes=1000)
        QL.plot(df=df, model=Qtable, train_size=0.8, name=data.split('_')[0], save=True, show=True)

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

        # Randomly choose a dataset per trial
        fdir = list_dir[trial.number%6]
        df  = dataloader.read(fdir)
        score_market = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        env = TradingEnv(df)
        QL  = QLearning(env, log=False)

        train_size = 0.8
        n_episode = 500

        Qtable = QL.train(
            df, train_size=train_size, n_training_episodes=n_episode,
            learning_rate=learning_rate, gamma=gamma,
            max_epsilon=1.0, min_epsilon=0.05, decay_rate=decay_rate
        )

        df_train, df_test = QL.split_data(df, train_size)
        _, _, _, equity_test, reward_test = QL.get_actions_and_prices(Qtable, df_test)
        score_equity = (equity_test[-1]-equity_test[0])/equity_test[0]
        score = score_equity - score_market
        print(f"TRIAL #{trial.number} -> market : {score_market} / equity : {score_equity} / score {score}")
        return score

    # Use persistent storage for multi-agent coordination
    storage_url = "sqlite:///Qlearning_optimization.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="QLearning_Study",
        storage=storage_url,
        load_if_exists=True
    )

    # Worker function
    def run_worker():
        study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Run in parallel
    n_workers = 6
    processes = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=run_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def get_best_parameter(storage_url):
    study = optuna.create_study(
            direction="maximize",
            study_name="QLearning_Study",
            storage=storage_url,
            load_if_exists=True
        )
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)



if __name__ == '__main__':
    test_QLearning("data/General/", with_broker=True)
