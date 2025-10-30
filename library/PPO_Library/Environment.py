import yfinance as yf
import pandas as pd
import numpy as np

import gym
from gym import spaces

import random
import os

if __name__ in ["__main__", "Environment"]:
    from PPO_Training_conf import DATASET_PATH, TODAY, INPUT_SIZE
else:   
    from .PPO_Training_conf import DATASET_PATH, TODAY, INPUT_SIZE

class DataLoader:

    def __init__(self):
        self.DATA_DIR   = str()
        self.DATA_FNAME = str()

    def __check_repo(self, repo_name):
        repo_path = os.path.abspath(repo_name)
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)

        return repo_path+"/"
    
    def __formating(self, company, start, end=None):
        if end == None:
            year_start = start.split('-')[0]
            return f'{company}_{year_start}.csv'
        else:
            year_start = start.split('-')[0]
            year_end   = end.split('-')[0]
            return f'{company}_{year_start}_{year_end}.csv'

    def save_company(self, df_full_path = "./", company_name = "my_extraction", start_date = "2020-01-01", end_date = None):
        if end_date==None:
            year = int(start_date.split('-')[0])
            self.df = yf.download(company_name, start=start_date, end=f'{year+1}-01-01', auto_adjust=True)
        else:
            self.df = yf.download(company_name, start=start_date, end=end_date, auto_adjust=True)
        self.df.to_csv(f'{df_full_path}{self.__formating(company_name, start_date, end_date)}', index=True)

    def save_companies(self, companies=None):

        if companies == None:
            company_dict = {
                "AAPL"    : ["2010-01-01", str(TODAY)+"-01-01"],
                "ATOS"    : ["2017-01-01", str(TODAY)+"-01-01"],
                "BTC-USD" : ["2014-01-01", str(TODAY)+"-01-01"],
                "O"       : ["2016-01-01", str(TODAY)+"-01-01"],
                "RNO.PA"  : ["2016-01-01", str(TODAY)+"-01-01"],
                "TSLA"    : ["2019-01-01", str(TODAY)+"-01-01"]
            }

        else : 
            company_dict = companies

        dir = self.__check_repo(DATASET_PATH)
        list_dir = os.listdir(dir)

        for company in company_dict.keys():
            if len(company_dict[company]) == 2:
                start, end = company_dict[company]
            else :
                start = company_dict[company][0]
                end   = None

            print(self.__formating(company, start, end))
            if self.__formating(company, start, end) not in list_dir:
                self.save_company(df_full_path=dir, company_name=company, start_date=start, end_date=end)

    def read(self,folder_path):
        df = pd.read_csv(folder_path, header=[0, 1], index_col=0)
        df.columns = df.columns.get_level_values(0)
        return df

    def split_train_test(self, folder_path, training_size=0.8):
        df = self.read(folder_path=folder_path)
        df_training = df[:int(training_size*len(df))]
        df_test     = df[int(training_size*len(df))+1:]

        return df_training, df_test


class TradingEnv(gym.Env):
    def __init__(self, df, broker_fee=False, share = 200):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index()
        self.n_steps = len(df)
        self.current_step = 0
        self.initial_balance = 100
        self.current_balance = self.initial_balance
        self.previous_balance = self.initial_balance
        self.position = 0  # 0 = short, 1 = long
        self.share = share
        self.last_price = 0
        self.reward = 0

        if broker_fee :
            self.broker_fee = 50 # commission in dollars
        else :
            self.broker_fee = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(INPUT_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

    def _get_obs(self):
        return np.array([
            self.df.iloc[self.current_step]["Close"],
            self.df.iloc[self.current_step]["Open"],
            self.df.iloc[self.current_step]["High"],
            self.df.iloc[self.current_step]["Low"],
            #self.position,
        ], dtype=np.float32)

    def sample_valid_action(self):
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions)

    def get_valid_actions(self):

        if self.position == 1 :  # last was BUY
            return [0, 2]  # SELL or HOLD
        elif self.position == 0 :  # last was SELL
            return [0, 1]  # BUY or HOLD
        else :
            return [0, 1, 2]

    def step(self, action):

        done  = False
        price = self.df.iloc[self.current_step]["Close"]

        if (action == 1) & (self.position == 0):   # BUY
            self.current_balance -= self.broker_fee
            self.position = 1

        elif (action == 2) & (self.position == 1): # SELL
            self.current_balance += 0
            self.position = 0

        elif (action == 0) & (self.position == 1): # LONG
            market_evolution = (price-self.last_price)/self.last_price
            self.current_balance += market_evolution*self.share

        elif (action == 0) & (self.position == 0): # SHORT
            market_evolution = 0
            self.current_balance += 0

        # Advance time
        self.current_step += 1
        if self.current_step >= self.n_steps-1:
            done = True

        raw_reward = self.current_balance - self.previous_balance
        if raw_reward == 0:
            self.reward += 0
        else:
            self.reward = np.sign(raw_reward)*((np.abs(raw_reward)))

        self.previous_balance = self.current_balance
        self.last_price       = price

        return self._get_obs(), self.reward, done, self.current_balance


    def set_data(self, df):
        self.df = df.reset_index()
        self.n_steps = len(df)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.last_action = 0
        self.reward = 0
        return self._get_obs()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    action_dict = {
        0:'Hold',
        1:'Buy',
        2:'Sell'
    }

    def get_next_step(env, log=True):
        possible_action = env.get_valid_actions()

        if len(possible_action) == 3 :
            action = np.random.choice(possible_action, p=[0.95, 0.025, 0.025])
        else:
            action = np.random.choice(possible_action, p=[0.95, 0.05])

        next_obs, reward, done, current_portfolio = env.step(action)

        if log:
            print(f'Action is {action_dict[action]}')
            print("Next observation:", next_obs)
            print("Reward:", reward)
            print("Done:", done)

        return action, current_portfolio, reward

    data_loader = DataLoader()
    df = data_loader.read('data/General/^VIX_2015_2025.csv')  # This is the DataFrame to use

    # Step 2: Initialize the environment
    env = TradingEnv(df, broker_fee=True)
    obs = env.reset()
    print("Initial observation:", obs)

    total_step = len(df)-1

    list_action    = list()
    list_portfolio = list()
    list_reward    = list()
    list_close = df['Close'][:total_step+1].values

    for i in range(1, total_step+1):
        #print(f'\nSTEP {i}')
        current_action, current_portfolio, current_reward = get_next_step(env, log=False)

        list_action.append(current_action)
        list_portfolio.append(current_portfolio)
        list_reward.append(current_reward)

    array_action = np.array(list_action)
    array_x_buy = np.where(array_action==1)
    array_y_buy = list_close[array_x_buy]

    array_x_sell = np.where(array_action==2)
    array_y_sell = list_close[array_x_sell]

    fig, ax1 = plt.subplots()

    ax1.plot(list_close)
    ax1.scatter(array_x_buy, array_y_buy, marker="^", c='green', s=22, zorder=3)
    ax1.scatter(array_x_sell, array_y_sell, marker="v", c='red', s=22, zorder=3)

    ax2 = ax1.twinx()
    ax2.plot(list_reward, color='red', label='REWARD', alpha=0.4)
    ax2.set_ylabel('REWARD', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Equity vs Reward')
    plt.grid()
    fig.tight_layout()

    plt.show()

    fig, ax1 = plt.subplots()

    ax1.plot(list_portfolio, color='blue', label='EQUITY')
    ax1.set_ylabel('EQUITY', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.grid()
