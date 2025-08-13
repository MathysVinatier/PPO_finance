import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
import random

class DataLoader:

    def __init__(self):
        self.DATA_DIR   = str()
        self.DATA_FNAME = str()

    def save(self, df_full_path = "./data/", df_fname = "my_extraction", start_date = "2020-01-01", end_date = "2023-01-01"):
        self.DATA_DIR   = df_full_path
        self.DATA_FNAME = df_fname
        self.FULL_PATH  = str()
        for element in self.DATA_DIR.split('/'):
            if not element:
                continue
            self.FULL_PATH += element + '/'
        self.FULL_PATH  = self.FULL_PATH+self.DATA_FNAME

        self.start      = start_date
        self.end        = end_date

        self.df = yf.download(self.DATA_FNAME, start=self.start, end=self.end, auto_adjust=True)
        self.df.to_csv(f'{self.FULL_PATH}_{self.start[:4]}_{self.end[:4]}.csv', index=True)

    def read(self,folder_path):
        df = pd.read_csv(folder_path, header=[0, 1], index_col=0)
        df.columns = df.columns.get_level_values(0)
        return df


class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index()
        self.n_steps = len(df)
        self.current_step = 0
        self.initial_balance = 100
        self.balance = self.initial_balance
        self.position = 0  # 0 = flat, 1 = holding
        self.last_action = 0  # Track last valid action (0 = hold, 1 = buy, 2 = sell)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

    def _get_obs(self):
        return np.array([
            self.df.iloc[self.current_step]["Close"],
            self.df.iloc[self.current_step]["Open"],
            self.df.iloc[self.current_step]["High"],
            self.df.iloc[self.current_step]["Low"],
            self.position,
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
        done = False

        price = self.df.loc[self.current_step, "Close"]
        prev_portfolio_value = self.balance + self.position * price

        if action == 1:  # BUY
            if self.position == 0:  # only if not already holding
                self.balance -= price
                self.position = 1
                self.last_action = 1
                penalty = 0
            else:
                penalty = -1  # small penalty for redundant buy
        elif action == 2:  # SELL
            if self.position == 1:  # only if holding
                self.balance += price
                self.position = 0
                self.last_action = 2
                penalty = 0
            else:
                penalty = -1  # small penalty for redundant sell
        else:  # HOLD
            penalty = 0

        # Advance time
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]["Close"]
        current_portfolio_value = self.balance + self.position * new_price

        # Reward: portfolio change + penalty
        reward = (current_portfolio_value - prev_portfolio_value) + penalty

        if self.current_step >= self.n_steps - 1:
            done = True

        return self._get_obs(), reward, done, {
            "portfolio_value": current_portfolio_value,
            "action": action
    }


    def set_data(self, df):
        self.df = df.reset_index()
        self.n_steps = len(df)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.last_action = 0
        return self._get_obs()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    action_dict = {
        0:'Hold',
        1:'Buy',
        2:'Sell'
    }

    def get_next_step(env, log=True):
        action = env.sample_valid_action()
        next_obs, reward, done, info = env.step(action)

        if log:
            print(f'Action is {action_dict[action]}')
            print(info)
            print("Next observation:", next_obs)
            print("Reward:", reward)
            print("Done:", done)

        return info, reward

    data_loader = DataLoader()
    df = data_loader.read('AAPL_2020_2023.csv')  # This is the DataFrame to use

    # Step 2: Initialize the environment
    env = TradingEnv(df)
    obs = env.reset()
    print("Initial observation:", obs)

    total_step = 750

    list_action    = list()
    list_portfolio = list()
    list_reward    = list()
    list_close = df['Close'][:total_step+1].values

    for i in range(1, total_step+1):
        #print(f'\nSTEP {i}')
        current_info, current_reward = get_next_step(env, log=False)

        list_action.append(current_info['action'])
        list_portfolio.append(current_info['portfolio_value'])
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
