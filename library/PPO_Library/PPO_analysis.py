import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from torch import no_grad, tensor, float32

if __name__ in ["__main__", "PPO_analysis"]:
    from PPO_Training_conf import *
    from Environment import TradingEnv, DataLoader
    from PPO import ACAgent
else:
    from .PPO_Training_conf import *
    from .PPO import ACAgent
    from .Environment import TradingEnv, DataLoader


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
            if type(value) == float or type(value) == np.float64 :
                value = f"{value:.3f}"
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
        self._current_agent_id  = ""

        self.agent  = ACAgent
        self.info   = self.get_info()

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
        else:
            print("... models already loaded ...")
        self.show_info()
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
    
    def plot(self, show=False, save_path=None):
        test_dict   = dict()
        train_dict  = dict()
        reward_dict = dict()

        episode_path = self._models_path

        df_path = self._dataset_path
        df_train, df_test = DataLoader().split_train_test(df_path)

        for mod in os.listdir(episode_path):
            episode = "_".join(mod.split("_")[3:])
            if episode not in test_dict.keys():
                model   = self.get_model(model_episode = episode)
                train   = ModelTest(model=model, df=df_train)
                test    = ModelTest(model=model, df=df_test)

                reward        = train.info["total_reward"]
                balance_test  = test.info["portfolio_values"]
                balance_train = train.info["portfolio_values"]

                test_dict[episode]   = balance_test
                train_dict[episode]  = balance_train
                reward_dict[episode] = reward


        fig, axes = plt.subplots(figsize=(16, 6), nrows=2, ncols=1)

        X_train = [x for x in range(len(df_train["Close"]))]
        X_test  = [x for x in range(len(df_test["Close"]))]
        axes[0].plot(X_train, df_train["Close"], color="black", linewidth=0.5, linestyle="--")
        axes[1].plot(X_test, df_test["Close"], color="black", linewidth=0.5, linestyle="--")

        ax_train = axes[0].twinx()
        ax_test  = axes[1].twinx()

        reward_array = np.array(list(reward_dict.values()))
        reward_max   = reward_array.max()

        train_color = "tab:blue"
        test_color  = "tab:green"
        for ep in test_dict.keys():
            if len(reward_array) == 1:
                hue = 1
            else:
                hue = abs(reward_dict[ep]/reward_max)
            ax_train.plot(X_train[:-1], train_dict[ep], color=train_color, alpha = hue, label=f"episode {ep} ({int(reward_dict[ep])})")
            ax_test.plot(X_test[:-1], test_dict[ep], color=test_color, alpha = hue, label=f"episode {ep} ({int(reward_dict[ep])})")

        for ax in [ax_train, ax_test]:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc="center left", bbox_to_anchor=(0, 0.5), frameon=True)

        ax_train.set_title("Training Set")
        ax_test.set_title("Testing Set")

        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path)
        if show:
            plt.show()

        return fig

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
        if agent_id == self._current_agent_id:
            return True
        else: return False

class ModelTest(_ModelFormat):

    def __init__(self, model, df):
        super().__init__()
        self.agent = model
        self.df    = df
        self.env   = TradingEnv(df, broker_fee=True)
        self._launch_test()

    def _launch_test(self, threshold = 0.65):
        tick = pd.to_datetime(self.df.index).to_series().diff().mode()[0]
        print(f"... starting test from {self.df.index.min()} to {self.df.index.max()} with {tick} tick (threshold={threshold}) ...")

        seq_len         = TRAINING_SEQ
        action_names    = ACTION_NAME_DICT

        obs              = self.env.reset()
        done             = False
        total_reward     = 0
        seq_buffer       = []
        actions_taken    = []
        portfolio_values = []
        probs_history    = []

        while not done:
            seq_buffer.append(obs)
            if len(seq_buffer) > seq_len:
                seq_buffer.pop(0)
            seq = seq_buffer if len(seq_buffer) == seq_len else [seq_buffer[0]]*(seq_len-len(seq_buffer)) + seq_buffer

            valid_actions = self.env.get_valid_actions()
            seq_array = np.array(seq, dtype=np.float32)[None, ...]  # batch dimension
            # Forward pass for visualization
            with no_grad():
                state = tensor(seq_array, dtype=float32).to(self.agent.actor.device)
                dist = self.agent.actor(state)
                probs = dist.probs.cpu().numpy().squeeze()
            probs_history.append(probs)

            # Choose action
            action, log_prob, value = self.agent.choose_action(seq_array, valid_actions, threshold=threshold)
            actions_taken.append(action)
            next_obs, reward, done, current_portfolio_value = self.env.step(action)
            total_reward += reward
            portfolio_values.append(current_portfolio_value)

            self.agent.remember(seq_array, action, log_prob, value, reward, done)
            obs = next_obs

        # --- Trading metrics ---
        portfolio_values = np.array(portfolio_values)
        returns          = np.diff(portfolio_values) / portfolio_values[:-1]

        # Annualized return
        total_period  = len(self.df)
        total_return  = portfolio_values[-1] / portfolio_values[0] - 1
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEARS / total_period) - 1

        # Average daily profit
        avg_profit = np.mean(returns)

        # Annualized volatility
        annual_vol = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEARS)

        # Sharpe ratio (risk-free rate = 0)
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(TRADING_DAYS_PER_YEARS) if np.std(returns) != 0 else 0

        # Max drawdown
        cumulative   = portfolio_values / portfolio_values[0]
        peak         = np.maximum.accumulate(cumulative)
        drawdowns    = (cumulative - peak) / peak
        max_drawdown = drawdowns.min()

        # --- Summary ---
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        self.info = {
            "actions_taken"     : actions_taken,
            "probs_history"     : probs_history,
            "portfolio_values"  : portfolio_values,
            "total_reward"      : total_reward,
            "annual_return"     : annual_return,
            "average_profit"    : avg_profit,
            "annual_volatility" : annual_vol,
            "sharpe_ratio"      : sharpe_ratio,
            "max_drawdown"      : max_drawdown,
            "action_summary"    : action_summary
        }

        self.__make_output_tabular__(info_line={k:self.info[k] for k in ["total_reward", "action_summary", "annual_return", "average_profit", "annual_volatility", "sharpe_ratio", "max_drawdown"]}, tab_title=f"Model Testing Result (threshold={threshold})")

        return self.info

    def plot(self, show=False, save_path=None):

        # --- Visualization ---
        prices = self.df["Close"].values[:len(self.info['actions_taken'])]
        probs_history = np.array(self.info["probs_history"])
        steps = np.arange(len(prices))

        fig, (ax_main, ax_probs) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                                gridspec_kw={'height_ratios': [2, 1]})

        # === Price + Portfolio ===
        ax_main.plot(steps, prices, color="black", label="Market Price", linewidth=1.2)
        buy_idx = [i for i, a in enumerate(self.info['actions_taken']) if a == 1]
        sell_idx = [i for i, a in enumerate(self.info['actions_taken']) if a == 2]
        ax_main.plot(buy_idx, prices[buy_idx], "^", color="tab:green", label="Buy", markersize=8)
        ax_main.plot(sell_idx, prices[sell_idx], "v", color="tab:red", label="Sell", markersize=8)
        ax_main.set_ylabel("Price")
        ax_main.set_title("Market Price & Portfolio Value")

        # twin y-axis for portfolio
        ax_portfolio = ax_main.twinx()
        ax_portfolio.plot(steps, self.info['portfolio_values'], color="blue", label="Portfolio Value", linewidth=1.2, alpha=0.7)
        ax_portfolio.set_ylabel("Portfolio Value")

        # combine legends
        lines_1, labels_1 = ax_main.get_legend_handles_labels()
        lines_2, labels_2 = ax_portfolio.get_legend_handles_labels()
        ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        ax_main.grid(True, linestyle='--', alpha=0.6)

        # === Action probability bars ===
        width = 0.25
        #ax_probs.bar(steps - width, probs_history[:, 0], width, label="Hold", color="gray", alpha=0.7)
        ax_probs.bar(steps, probs_history[:, 1], width, label="Buy", color="tab:gray", alpha=0.5)
        ax_probs.bar(steps + width, probs_history[:, 2], width, label="Sell", color="tab:red", alpha=1)
        ax_probs.set_ylim(0, 1)
        ax_probs.set_xlabel("Step")
        ax_probs.set_ylabel("Probability")
        ax_probs.set_title("Action Probabilities per Step")
        ax_probs.legend()
        ax_probs.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path)
        if show:
            plt.show()

        return fig

if __name__ == "__main__":
    from Environment import DataLoader

    model = ModelReport('/home/mathys/Documents/PPO_finance/multitask_PPO/task_0')
    model.plot(show=True, save_path=None)
    # agent = model.get_model()
# 
    # df_train, df_test = DataLoader().split_train_test(model._dataset_path)
    # test_result       = ModelTest(agent, df_test)
    # train_result      = ModelTest(agent, df_train)
# 
    # train_result.plot(show=True)
    # test_result.plot(show=True)
