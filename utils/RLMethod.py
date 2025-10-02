import numpy as np
import sys
import os
import time

import shutil
from tqdm import tqdm
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from transformers import DecisionTransformerConfig, DecisionTransformerGPT2Model

if __name__ == "__main__":
    from Models import MLP, DEVICE, Encoder_Transformer
    from PPO import ACAgent
else:
    from utils.Models import MLP, DEVICE, Encoder_Transformer
    from utils.PPO import ACAgent

class ModelRL:
    def __init__(self, env, log=False):
        self.env = env
        self.log = log
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        if self.log:
            print('================== ENV INFO ==================')
            print("Observation Space:", self.env.observation_space)
            print("Sample observation:", self.env.observation_space.sample())
            print("Number of features:", self.state_space)
            print("\nAction Space:", self.env.action_space.n)
            print("Sample action:", self.env.action_space.sample())
            print('==============================================\n')


    def __bar_graph(self, label, count, max_count):
        max_width    = shutil.get_terminal_size().columns // 2
        count        = int(count)
        max_count    = max(1, max_count)
        bar_length   = int((count / max_count) * max_width)
        bar          = '█' * bar_length
        count_str    = f"  {count}"
        line         = f"{label:<4}|{bar}{count_str}"
        total_length = len(label) + 2 + max_width + len(count_str)
        return line.ljust(total_length)

    def __reload_bar(self):
        self.action_counts = np.zeros(self.env.action_space.n)

    def __log_bar(self,action):
        self.action_counts[action] += 1
        max_count = max(self.action_counts)
        print(self._ModelRL__bar_graph("HOLD", self.action_counts[0], max_count))
        print(self._ModelRL__bar_graph("BUY", self.action_counts[1], max_count))
        print(self._ModelRL__bar_graph("SELL", self.action_counts[2], max_count))
        sys.stdout.write('\033[F' * 3)

    def split_data(self, df, train_size):
        df_train = df[:int(train_size * len(df))]
        df_test = df[int(train_size * len(df)):]
        return df_train, df_test

    def plot(self, df, model, train_size=0.8, save=False, name="Company", show=True):
        plt.figure(figsize=(14, 6))

        df_train, df_test = self.split_data(df, train_size)

        # Now returns actions, prices, dates and equity curve directly
        actions_train, prices_train, dates_train, equity_train, reward_train = self.get_actions_and_prices(model, df_train, reward_type="portfolio_diff", reward_evolution="value")
        actions_test, prices_test, dates_test, equity_test, reward_test = self.get_actions_and_prices(model, df_test, reward_type="portfolio_diff", reward_evolution="value", initial_cash=equity_train[-1])

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
            plt.savefig(f'./{self.__class__.__name__}_{name.split(".")[0]}_{df.index[0][:4]}')

        plt.figure(figsize=(14, 6))

        plt.plot(x_train, reward_train, label="Reward Train", linewidth=1)
        plt.plot(x_test, reward_test, label="Reward Test", linewidth=1)

        plt.title("Reward Evolution")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if show:
            plt.show()


class QLearning(ModelRL):
    def __init__(self, env, log=True):
        super().__init__(env, log)

    def __get_bins(self, df_training):
        bins = []
        training_values = df_training.copy()
        for i in range(self.env.observation_space.shape[0]):
            feature_vals = training_values.iloc[:, i]
            bins.append(np.linspace(feature_vals.min(), feature_vals.max(), num=10))
        return bins

    def discretize_state(self, state, bins):
        return tuple(np.digitize(state[i], b) for i, b in enumerate(bins))

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
        return np.zeros((num_states, self.env.action_space.n))

    def greedy_policy(self, Qtable, state_idx):
        return np.argmax(Qtable[state_idx])

    def epsilon_greedy_policy(self, Qtable, state_idx, epsilon=0):
        if np.random.rand() > epsilon:
            valid_actions = self.env.get_valid_actions()
            action        = np.argmax(Qtable[state_idx])
            if action in valid_actions:
                return action
            else:
                return 0
        else:
            return self.env.sample_valid_action()

    def train(self, df, reward_type, reward_evolution, train_size=0.8, n_training_episodes=1500,
              learning_rate=0.36, gamma=0.55, max_epsilon=0.5,
              min_epsilon=0.2, decay_rate=0.005):

        self.env.reward_type = reward_type
        self.env.reward_evolution = reward_evolution
        self.df = df
        self.df_train, _ = self.split_data(self.df.copy(), train_size)
        max_steps = len(self.df_train['Close'])
        bins = self.__get_bins(self.df_train)
        Qtable = self.initialize_q_table(bins)


        if self.log:
            pbar = tqdm(range(n_training_episodes))
            self._ModelRL__reload_bar()
        else:
            pbar = range(n_training_episodes)

        for episode in pbar:
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state_cont = self.env.reset()
            state_disc = self.discretize_state(state_cont, bins)
            state_idx = self.state_to_index(state_disc, bins)

            for _ in range(max_steps):
                action = self.epsilon_greedy_policy(Qtable, state_idx, epsilon)
                next_state_cont, reward, done, _ = self.env.step(action)

                next_state_disc = self.discretize_state(next_state_cont, bins)
                next_state_idx = self.state_to_index(next_state_disc, bins)

                valid_next_actions = self.env.get_valid_actions()
                max_next_q = np.max(Qtable[next_state_idx, valid_next_actions])
                Qtable[state_idx, action] += learning_rate * (
                    reward + gamma * max_next_q - Qtable[state_idx, action]
                )
                state_idx = next_state_idx

                if done:
                    break

                if self.log:
                    self._ModelRL__log_bar(action)

        if self.log:
            print('\n' * 3)
        return Qtable

    def get_actions_and_prices(self, Qtable, df,reward_type, reward_evolution, initial_cash=100):
        self.env.set_data(df)
        self.env.reward_type = reward_type
        self.env.reward_evolution = reward_evolution
        bins = self.__get_bins(df)
        state_cont = self.env.reset()
        state_disc = self.discretize_state(state_cont, bins)
        state_idx = self.state_to_index(state_disc, bins)

        actions_taken, prices, df_indices, equity_curve, reward_list = [], [], [], [], []
        done, step = False, 0

        while not done:
            if self.env.current_step >= len(df) - 1:
                break

            action = self.epsilon_greedy_policy(Qtable, state_idx)

            next_obs, reward, done, current_portfolio = self.env.step(action)
            reward_list.append(reward)
            equity_curve.append(current_portfolio)

            price = df["Close"].iloc[self.env.current_step - 1]

            actions_taken.append((step, action))
            prices.append(price)
            df_indices.append(self.env.current_step)

            state_disc = self.discretize_state(next_obs, bins)
            state_idx = self.state_to_index(state_disc, bins)
            step += 1

        return actions_taken, prices, df_indices, equity_curve, reward_list


class ReplayBuffer:
    """Simple replay buffer for DQN."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done']
        )

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([b.state for b in batch], dtype=np.float32)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DecisionTransformerQ(nn.Module):
    def __init__(self, input_dim, output_dim, model_name="edbeeching/decision-transformer-gym-hopper-medium"):
        super().__init__()
        # Load config for hidden size, layers, etc.
        config = DecisionTransformerConfig.from_pretrained(model_name)
        self.backbone = DecisionTransformerGPT2Model(config)

        # Our own projection
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        self.q_head = nn.Linear(config.hidden_size, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        # Project states to hidden size
        x = self.input_proj(x)  # (batch, steps, hidden)

        # Pass directly to GPT2 backbone
        outputs = self.backbone(inputs_embeds=x)

        # Take last hidden state
        last_hidden = outputs.last_hidden_state[:, -1]
        return self.q_head(last_hidden)



class DeepQLearning(ModelRL):

    def __init__(self, env, steps=1, gpt=False, log=False, hidden_dims=(128, 128)):
        super().__init__(env=env, log=log)
        self.steps = max(1, int(steps))

        self.state_space = env.observation_space.shape
        self.action_space = env.action_space.n

        # Transformer input is per-step state vector
        input_dim = self.state_space
        output_dim = self.action_space

        if gpt == True:
            self.policy_net = DecisionTransformerQ(input_dim, output_dim).to(DEVICE)
        else:
            self.policy_net = Encoder_Transformer(num_features=5, d_model=512, num_heads=8, num_classes=3).to(DEVICE)

        if self.log:
            print(self.policy_net)
        
        if gpt == True:
            self.target_net = DecisionTransformerQ(input_dim, output_dim).to(DEVICE)
        else:
            self.target_net = Encoder_Transformer(num_features=5, d_model=512, num_heads=8, num_classes=3).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    # ------------------ DQN Core ------------------
    def _stack_state(self, history_deque):
        if len(history_deque) < self.steps:
            pad = [np.zeros(self.state_space, dtype=np.float32)] * (self.steps - len(history_deque))
            arr = np.stack(pad + list(history_deque), axis=0)  # (steps, state_dim)
        else:
            arr = np.stack(list(history_deque), axis=0)  # (steps, state_dim)
        return arr.astype(np.float32)

    @torch.no_grad()
    def _q_values(self, state_batch):
        return self.policy_net(torch.as_tensor(state_batch, dtype=torch.float32, device=DEVICE))

    def _select_action_eps_greedy(self, state_vec, epsilon):
        if np.random.rand() < epsilon:
            return self.env.sample_valid_action()

        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = self.policy_net(state_t)[0].detach().cpu().numpy()

        valid_actions = self.env.get_valid_actions()
        q_valid = q_values[valid_actions]

        return int(valid_actions[np.argmax(q_valid)])

    def _optimize(self, optimizer, buffer, batch_size, gamma):
        if len(buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_sa = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(1, keepdim=True)[0]
            q_target = rewards_t + gamma * q_next * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_sa, q_target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    # ------------------ Training ------------------
    def train(self, df, train_size=0.8, n_training_episodes=500,
              learning_rate=1e-4, gamma=0.9, max_epsilon=1.0,
              min_epsilon=0.07, decay_rate=0.009, buffer_capacity=100_000,
              batch_size=128, target_update_every=250, learn_every=2,
              warmup_steps=10_000):

        self.df_train, _ = self.split_data(df, train_size)
        buffer = ReplayBuffer(buffer_capacity)
        optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        max_steps = len(self.df_train['Close'])

        global_step = 0
        pbar = tqdm(range(n_training_episodes), disable=not self.log)

        if self.log:
            action_counts = np.zeros(self.action_space)

        for episode in pbar:
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state_cont = self.env.reset()
            history = deque(maxlen=self.steps)
            history.append(np.asarray(state_cont, dtype=np.float32))
            state_vec = self._stack_state(history)

            for step in range(max_steps):
                action = self._select_action_eps_greedy(state_vec, epsilon)
                next_state_cont, reward, done, info = self.env.step(action)
                history.append(np.asarray(next_state_cont, dtype=np.float32))
                next_state_vec = self._stack_state(history)
                buffer.push(state_vec, action, reward, next_state_vec, done)

                if global_step % learn_every == 0 and len(buffer) >= max(batch_size, warmup_steps):
                    self._optimize(optimizer, buffer, batch_size, gamma)

                global_step += 1
                if global_step % target_update_every == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                state_vec = next_state_vec
                if done:
                    break

        return self.policy_net

    @torch.no_grad()
    def get_actions_and_prices(self, policy_net, df, reward_type="portfolio", reward_evolution="value", initial_cash=100):
        self.env.set_data(df)
        self.env.reward_type = reward_type
        self.env.reward_evolution = reward_evolution
        obs = self.env.reset()
        history = deque(maxlen=1000)
        history.append(np.asarray(obs, dtype=np.float32))

        actions_taken, prices, df_indices, equity_curve, reward_list = [], [], [], [], []
        cash, stock, step = initial_cash, 0, 0
        done = False

        while not done:
            if self.env.current_step >= len(df) - 1:
                break

            state_t = torch.as_tensor(np.array(history, dtype=np.float32),
                                    dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, seq_len, obs_dim)

            logits, _ = policy_net(state_t)
            logits = logits[0]
            valid_actions = self.env.get_valid_actions()
            masked_logits = torch.full_like(logits, float('-inf'))
            masked_logits[valid_actions] = logits[valid_actions]

            action = int(torch.argmax(masked_logits).item())

            next_obs, reward, done, info = self.env.step(action)
            price = df["Close"].iloc[self.env.current_step - 1]

            actions_taken.append((step, action))
            prices.append(price)
            df_indices.append(self.env.current_step)
            reward_list.append(reward)

            # Portfolio update
            if action == 1:  # Buy
                stock += 1
                cash -= price
            elif action == 2:  # Sell
                stock -= 1
                cash += price
            equity_curve.append(cash + stock * price)

            history.append(next_obs)
            step += 1

        return actions_taken, prices, df_indices, equity_curve, reward_list

class PPOAgent(DeepQLearning):

    def __init__(self, env, log):
        super().__init__(env=env, log=log)

    def train(self, n_games=300, N=20, batch_size=5, n_epochs=4, alpha=0.0003, show=False):
        agent = ACAgent(n_actions  = self.action_space,
                        batch_size = batch_size,
                        alpha      = alpha,
                        n_epochs   = n_epochs,
                        input_dims = self.state_space)

        figure_file = 'tmp/ppo/ppo_agent.png'
        best_score = self.env.initial_balance
        score_history = []

        learn_iters = 0
        avg_score   = 0
        n_steps     = 0

        for i in range(n_games):
            observation = self.env.reset()
            done        = False
            score       = 0

            while not done:
                valid_actions = self.env.get_valid_actions()
                action, probs, val = agent.choose_action(observation, valid_actions=valid_actions)
                observation_, reward, done, info = self.env.step(action)
                n_steps += 1
                score  += reward
                agent.remember(observation, action, probs, val, reward, done)
                #print(f"{observation} -> {action} (out of {valid_actions}) -> reward : {reward:.2f} score : {score:.2f}")

                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history)

            if avg_score > best_score :
                best_score = avg_score
                agent.save_models()

            print(f"episode {i} => score {score:.1f}  | avg score {avg_score:.1f} | time_steps {n_steps} | learning_steps {learn_iters}")

        x = [i+1 for i in range(len(score_history))]
        if show:
            agent.plot_learning_curve(x, score_history, figure_file)
        return agent
    
    def test(self, batch_size=5, n_epochs=4, alpha=0.0003, greedy=False):
        """
        Run a single evaluation episode using the trained PPO agent.
        Returns the sequence of actions and portfolio values.
        """
        agent = ACAgent(
            n_actions=self.action_space,
            batch_size=batch_size,
            alpha=alpha,
            n_epochs=n_epochs,
            input_dims=self.state_space
        )

        # Load trained actor/critic weights
        agent.load_models()

        observation = self.env.reset()
        done = False
        score = 0
        action_history = []
        portfolio_history = []
        score_history = []

        while not done:
            valid_actions = self.env.get_valid_actions()
            dist = agent.actor(torch.tensor(observation, dtype=torch.float).to(agent.actor.device))

            if greedy:
                action = torch.argmax(dist.probs).item()
                if action not in valid_actions:
                    action = np.random.choice(valid_actions)
                value = agent.critic(torch.tensor(observation, dtype=torch.float).to(agent.actor.device))
                value = torch.squeeze(value).item()
            else:
                action, _, value = agent.choose_action(observation, valid_actions=valid_actions)

            observation_, reward, done, info = self.env.step(action)
            score += reward
            score_history.append(score)
            action_history.append(action)
            portfolio_history.append(info)

            observation = observation_

        print(f"Test episode score {score:.1f} | final portfolio {info:.1f}")
        return action_history, portfolio_history, score_history



if __name__ == "__main__":
    from Environment import TradingEnv, DataLoader

    df = DataLoader().read("./data/General/^VIX_2015_2025.csv")
    env = TradingEnv(df)
    """
    model = DeepQLearning(env,gpt=True ,log=True)
    policy_net = model.train(
        df=df,
        train_size=0.8,
        n_training_episodes=500,    # much longer training
        learning_rate=1e-4,         # smaller LR for stability
        gamma=0.99,                 # discount factor (keep)
        max_epsilon=1.0,            # start fully random
        min_epsilon=0.05,           # leave some exploration
        decay_rate=0.005,           # faster decay (converge to exploitation earlier)
        buffer_capacity=100_000,    # big replay memory
        batch_size=128,             # larger batches stabilize updates
        target_update_every=500,    # update target net more frequently
        learn_every=4,              # learn every few steps (stabilizes learning)
        warmup_steps=5_000          # collect random transitions before training
    )
    # Save
    torch.save(policy_net.state_dict(), "gpt_transformer_policy_vix.pth")


    model_loaded = DeepQLearning(env)
    model_loaded.policy_net.load_state_dict(torch.load("gpt_transformer_policy_vix.pth"))
    model_loaded.policy_net.eval()
    """

    model_name = "ppo_transformer_policy_vix.pth"
    '''
    model = PPOAgent(env, log=True)

    policy_net = model.train(
        df=df,
        n_training_epochs=500,
        total_timesteps=100000,
        lr=1e-4,
        batch_size=128,
        save_path=model_name
    )
    '''
    #model_loaded_DL = DeepQLearning(env, log=False)
    #model_loaded_DL.policy_net.load_state_dict(torch.load("model/transformer/encoder_transformer_policy_vix1.pth"))

    model_loaded_PPO = PPOAgent(env, log=False).ACAgent()
    model_loaded_PPO.policy_net.load_state_dict(torch.load(model_name))
    model_loaded_PPO.policy_net.eval()

    model_loaded_PPO.plot(df, model = model_loaded_PPO.policy_net.encoder, name="VIX", save=False)