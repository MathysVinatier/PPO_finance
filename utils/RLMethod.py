import numpy as np
import sys
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class QLearning:
    def __init__(self, env, log=True):
        self.log = log
        self.env = env

        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        if self.log != True:
            print('===============================================================')
            print("_____OBSERVATION SPACE_____ \n")
            print("Observation Space", self.env.observation_space)
            print("Sample observation", self.env.observation_space.sample())
            print("There are ", self.state_space, " possible states")

            print("\n _____ACTION SPACE_____ \n")
            print("Action Space Shape", self.env.action_space.n)
            print("Action Space Sample", self.env.action_space.sample())
            print("There are ", self.action_space, " possible actions")
            print('===============================================================\n')


    def __get_bins(self, df_training):
        bins = list()
        training_values = df_training.copy()

        for i in range(self.env.observation_space.shape[0]):
            feature_vals = training_values.iloc[:, i]
            bins.append(np.linspace(feature_vals.min(), feature_vals.max(), num=10))
        return bins

    def __bar_graph(self, label, count, max_count):
        max_width=shutil.get_terminal_size().columns//2
        count = int(count)
        max_count = max(1, max_count)  # avoid division by zero
        bar_length = int((count / max_count) * max_width)
        bar = '█' * bar_length
        count_str = f"  {count}"
        line = f"{label:<4}|{bar}{count_str}"

        # total length should be fixed: label + " |" + bar (max_width) + count_str (fixed width)
        total_length = len(label) + 2 + max_width + len(count_str)

        # pad right with spaces so line length == total_length
        return line.ljust(total_length)

    def discretize_state(self, state, bins):
        discrete_state = []
        for i, b in enumerate(bins):
            discrete_state.append(np.digitize(state[i], b))
        return tuple(discrete_state)

    def state_to_index(self, discrete_state, bins):
        sizes = [len(b) + 1 for b in bins]
        index = 0
        for i, val in enumerate(discrete_state):
            prod = np.prod(sizes[i+1:]) if i + 1 < len(sizes) else 1
            index += val * prod
        return int(index)

    def initialize_q_table(self, bins):
        num_bins_per_feature = [len(b) + 1 for b in bins]
        num_states = np.prod(num_bins_per_feature)
        num_actions = self.env.action_space.n
        Qtable = np.zeros((num_states, num_actions))
        return Qtable

    def greedy_policy(self, Qtable, state):
        action = np.argmax(Qtable[state])
        return action

    def epsilon_greedy_policy(self, Qtable, state_idx, epsilon):
        if np.random.rand() > epsilon:
            return np.argmax(Qtable[state_idx])
        else:
            return self.env.sample_valid_action()

    def split_data(self, df, train_size):
        df_train = df[:int(train_size*len(df))]
        df_test = df[int(train_size*len(df)):]
        return df_train, df_test

    def train(self, df,
            train_size=0.8,
            n_training_episodes=1000,       # Total training episodes
            learning_rate=0.7,              # Learning rate
            gamma=0.95,                    # Discounting rate
            max_epsilon=1.0,               # Exploration probability at start
            min_epsilon=0.05,              # Minimum exploration probability
            decay_rate=0.0005):            # Exploration decay rate


        self.df       = df
        self.df_train = self.split_data(self.df.copy(), train_size)[0]

        max_steps = len(self.df_train['Close'])
        bins = self.__get_bins(self.df_train)

        Qtable = self.initialize_q_table(bins)

        pbar = tqdm(range(n_training_episodes))
        print()

        if self.log :
            action_counts = np.zeros(self.env.action_space.n)

        for episode in pbar:
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

                # Reset environment
                state_cont = self.env.reset()
                state_disc = self.discretize_state(state_cont, bins)
                state_idx = self.state_to_index(state_disc, bins)
        
                for step in range(max_steps):
                    # Choose action
                    action = self.epsilon_greedy_policy(Qtable, state_idx, epsilon)

                    # Take step
                    next_state_cont, reward, done, info = self.env.step(action)

                    # Discretize next state
                    next_state_disc = self.discretize_state(next_state_cont, bins)
                    next_state_idx = self.state_to_index(next_state_disc, bins)

                    # Q-learning update
                    Qtable[state_idx][action] += learning_rate * (
                        reward + gamma * np.max(Qtable[next_state_idx]) - Qtable[state_idx][action]
                    )

                    # Move to next state
                    state_idx = next_state_idx

                    if done :
                        break

                    if self.log:
                        sys.stdout.flush()
                        # Move cursor UP 3 lines to overwrite previous bars
                        action_counts[action] += 1
                        sys.stdout.write('\033[E' * 1)
                        sys.stdout.flush()

                        max_count = max(action_counts)

                        print(self.__bar_graph("HOLD", action_counts[0], max_count))
                        print(self.__bar_graph("BUY",  action_counts[1], max_count))
                        print(self.__bar_graph("SELL", action_counts[2], max_count))

                        sys.stdout.write('\033[F' *4)
                        sys.stdout.flush()

                        #pbar.set_postfix({
                        #    'HOLD': int(action_counts[0]),
                        #    'BUY': int(action_counts[1]),
                        #    'SELL': int(action_counts[2])
                        #})
        if self.log:
            print('\n'*3)
        return Qtable

    def get_actions_and_prices(self, Qtable, df, initial_cash=100):
        self.env.set_data(df)
        state_cont = self.env.reset()
        bins = self.__get_bins(df)
        state_disc = self.discretize_state(state_cont, bins)
        state_idx = self.state_to_index(state_disc, bins)

        actions_taken = []
        prices = []
        df_indices = []
        equity_curve = []

        cash = initial_cash
        stock = 0

        done = False
        step = 0
        while not done:
            action = np.argmax(Qtable[state_idx])
            if action not in self.env.get_valid_actions():
                action = 0
            next_obs, reward, done, info = self.env.step(action)
            price = df["Close"].iloc[self.env.current_step-1]

            # Record action and price
            actions_taken.append((step, action))
            prices.append(price)
            df_indices.append(self.env.current_step)

            # Update portfolio based on action
            if action == 1 :  # Buy
                stock += 1
                cash -= price
            elif action == 2 :    # Sell
                stock -= 1
                cash += price

            total_value = cash + stock * price
            equity_curve.append(total_value)

            # Move to next state
            next_state_disc = self.discretize_state(next_obs, bins)
            state_idx = self.state_to_index(next_state_disc, bins)

            step += 1

        return actions_taken, prices, df_indices, equity_curve

    def plot(self, df, Qtable, train_size=0.8, save=False, name="Company", show=True):
        plt.figure(figsize=(14, 6))

        df_train, df_test = self.split_data(df, train_size)

        # Now returns actions, prices, dates, and equity curve directly
        actions_train, prices_train, dates_train, equity_train = self.get_actions_and_prices(Qtable, df_train)
        actions_test, prices_test, dates_test, equity_test = self.get_actions_and_prices(Qtable, df_test, initial_cash=equity_train[-1])

        # --- Subplot 1 : Price + Buy/Sell ---
        plt.subplot(2, 1, 1)

        # TRAIN
        train_value = df_train["Close"]

        list_train_buy = [i for i, (_, action) in enumerate(actions_train) if action == 1]
        list_train_sell = [i for i, (_, action) in enumerate(actions_train) if action == 2]

        buy_dates_train = train_value.index[list_train_buy]
        sell_dates_train = train_value.index[list_train_sell]

        buy_prices_train = train_value.values[list_train_buy]
        sell_prices_train = train_value.values[list_train_sell]

        plt.plot(train_value.index, train_value, label="Training Data", linewidth=1)
        plt.scatter(buy_dates_train, buy_prices_train, marker="^", c='green', s=22, zorder=3)
        plt.scatter(sell_dates_train, sell_prices_train, marker="v", c='red', s=22, zorder=3)

        # TEST
        test_value = df_test["Close"]

        list_test_buy = [i for i, (_, action) in enumerate(actions_test) if action == 1]
        list_test_sell = [i for i, (_, action) in enumerate(actions_test) if action == 2]

        buy_dates_test = test_value.index[list_test_buy]
        sell_dates_test = test_value.index[list_test_sell]

        buy_prices_test = test_value.values[list_test_buy]
        sell_prices_test = test_value.values[list_test_sell]

        plt.plot(test_value.index, test_value, label="Testing Data", linewidth=1)
        plt.scatter(buy_dates_test, buy_prices_test, marker="^", c='green', s=22, zorder=3)
        plt.scatter(sell_dates_test, sell_prices_test, marker="v", c='red', s=22, zorder=3)
        
        plt.title(f"Buy & Sell Signals on {name} in {df.index[0][:4]}")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        plt.xticks(rotation=45)
        plt.xlabel("Date")

        # --- Subplot 2 : Equity Curve ---
        plt.subplot(2, 1, 2)

        # Create x-axis indices for train and test equity separately
        x_train = range(len(equity_train))
        x_test = range(len(equity_train), len(equity_train) + len(equity_test))

        plt.plot(x_train, equity_train, label="Equity Curve Train", linewidth=1)
        plt.plot(x_test, equity_test, label="Equity Curve Test", linewidth=1)

        plt.title("Equity Curve")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        if save:
            plt.savefig(f'./QLearning_{name}_{df.index[0][:4]}')
        if show:
            plt.show()

