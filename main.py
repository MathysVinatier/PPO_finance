from utils.Environment import DataLoader, TradingEnv
from utils.RLMethod import QLearning

import os
import matplotlib.pyplot as plt

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
    start_date = 2010
    stop_date = 2021

    dir = './data/10yearData/'
    company = 'BTC-USD'
    data_loader = DataLoader()
    list_dir = os.listdir(dir)
    fname = f"{company}_{start_date}_{stop_date}.csv"
    if fname not in list_dir:
        data_loader.save(df_full_path = dir, df_fname = company, start_date = f"{start_date}-01-01", end_date = f"{stop_date}-01-01")

    training_size = 0.8
    episode = 1000

    print(f"\n\n File => {fname}")
    df = data_loader.read(dir+fname)
    env = TradingEnv(df)
    QL = QLearning(env)

    Qtable_trained = QL.train(df, n_training_episodes=episode, train_size=training_size)
    QL.plot(df, Qtable_trained, training_size, save=True, show=True, name=company)


if __name__ == '__main__':
    experiment2()