import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


if __name__ == "__main__" or __name__ == "PPO":
    from Models import PPO_Transformer, Time2Vec
else:
    from utils.Models import PPO_Transformer, Time2Vec

class PPOMemory:
    """
    PPOMemory is a class defined to save the different states
    and variable of the PPO agent while training
    """
    def __init__(self, batch_size):
        self.states     = []
        self.probs      = []
        self.vals       = []
        self.actions    = []
        self.rewards    = []
        self.dones      = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states    = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices     = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches     = [indices[i:i+self.batch_size] for i in batch_start]

        return  np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states     = []
        self.probs      = []
        self.actions    = []
        self.rewards    = []
        self.dones      = []
        self.vals       = []


class ActorNetwork(nn.Module):
    def __init__(self, num_features, seq_len, n_actions=2, d_model=64, num_heads=8, time_dim=8, chkpt_dir='tmp/ppo', agent_id=False):
        super().__init__()
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        if agent_id == False:
            self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_transformer')
        else:
            self.checkpoint_file = os.path.join(chkpt_dir, 'actor_weight_'+agent_id)


        self.seq_len = seq_len
        self.transformer = PPO_Transformer(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_classes=n_actions,
            time_dim=time_dim
        )

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)
        returns: torch.distributions.Categorical
        """
        probs = self.transformer(x)  # shape: (batch, n_actions)
        dist = torch.distributions.Categorical(probs)
        return dist

    def save_checkpoint(self, episode=""):
        torch.save(self.state_dict(), self.checkpoint_file+episode)

    def load_checkpoint(self, episode=""):
        if episode == "":
            self.load_state_dict(torch.load(self.checkpoint_file+episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file+"_"+episode))


class CriticNetwork(nn.Module):
    def __init__(self, num_features, seq_len, alpha, time_dim=8, lstm_hidden=128, fc_dim=128, chkpt_dir='tmp/ppo', agent_id=False):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        
        if agent_id == False:
            self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo_transformer')
        else:
            self.checkpoint_file = os.path.join(chkpt_dir, 'critic_weight_'+agent_id)

        self.seq_len = seq_len
        self.num_features = num_features
        self.time_dim = time_dim

        # --- Time2Vec layer ---
        self.time2vec = Time2Vec(time_dim)

        # --- Feature projection ---
        self.feature_proj = nn.Linear(num_features, fc_dim)

        # --- LSTM layer ---
        self.lstm = nn.LSTM(input_size=fc_dim + time_dim, hidden_size=lstm_hidden, batch_first=True)

        # --- Fully connected layers ---
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)
        returns: (batch, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Create time indices (normalized)
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        t = t / seq_len  # normalize to [0,1]

        # Apply Time2Vec
        t2v = self.time2vec(t)  # (batch, seq_len, time_dim)

        # Project features
        feat_proj = self.feature_proj(x)  # (batch, seq_len, fc_dim)

        # Concatenate along feature dimension
        combined = torch.cat([feat_proj, t2v], dim=-1)  # (batch, seq_len, fc_dim + time_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(combined)  # (batch, seq_len, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]   # take last time step (batch, lstm_hidden)

        # Final dense layers
        value = self.fc(last_hidden)       # (batch, 1)
        return value

    def save_checkpoint(self, episode=""):
        torch.save(self.state_dict(), self.checkpoint_file+episode)

    def load_checkpoint(self, episode=""):
        if episode == "":
            self.load_state_dict(torch.load(self.checkpoint_file+episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file+"_"+episode))

class ACAgent:
    def __init__(self, n_actions, num_features, seq_len, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, chkpt_dir='tmp/ppo', agent_id=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.seq_len = seq_len
        self.num_features = num_features
        self.all_actor_losses = []
        self.all_critic_losses = []

        self.actor  = ActorNetwork(num_features=num_features, seq_len=seq_len, n_actions=n_actions, chkpt_dir=chkpt_dir, agent_id=agent_id)
        self.critic = CriticNetwork(num_features=num_features, seq_len=seq_len, alpha=alpha, chkpt_dir=chkpt_dir, agent_id=agent_id)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, episode=""):
        print('... saving models ...')
        self.actor.save_checkpoint(episode)
        self.critic.save_checkpoint(episode)

    def load_models(self, episode=""):
        print('... loading models ...')
        self.actor.load_checkpoint(episode)
        self.critic.load_checkpoint(episode)

    def choose_action(self, observation, valid_actions, threshold=0.65):
        """
        observation: (seq_len, num_features)
        valid_actions: indices of allowed actions
        threshold: probability threshold for hold action
        """
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)
        # shape: (1, seq_len, num_features)

        dist = self.actor(state)
        value = self.critic(state)

        probs = dist.probs.clone().squeeze(0)

        # Mask invalid actions
        mask = torch.zeros_like(probs)
        mask[valid_actions] = 1
        masked_probs = probs * mask
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            masked_probs = mask / mask.sum()

        # Threshold logic: hold if all probabilities below threshold
        if torch.all(masked_probs < threshold):
            action = 0  # hold
        else:
            action = torch.argmax(masked_probs).item()

        log_prob = torch.log(masked_probs[action] + 1e-8).item()  # add epsilon to avoid log(0)
        value = torch.squeeze(value).item()

        return action, log_prob, value

    def learn(self):
        for epoch in range(self.n_epochs):
            self.all_actor_losses = []
            self.all_critic_losses = []

            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            # Compute advantages using GAE
            values      = vals_arr
            advantage   = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t      = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states      = torch.tensor(state_arr[batch], dtype=torch.float).squeeze(1).to(self.actor.device)
                old_probs   = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions     = torch.tensor(action_arr[batch]).to(self.actor.device)

                # Actor forward
                dist         = self.actor(states)
                # Critic forward (flatten sequence)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs               = dist.log_prob(actions)
                prob_ratio              = new_probs.exp() / old_probs.exp()
                weighted_probs          = advantage[batch] * prob_ratio
                weighted_clipped_probs  = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss              = -torch.min(weighted_probs, weighted_clipped_probs).mean() - 0.1*dist.entropy().mean()

                returns     = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                self.all_actor_losses.append(actor_loss.item())
                self.all_critic_losses.append(critic_loss.item())

            # Print monitoring info per epoch
            print(f"     | Epoch {epoch+1}/{self.n_epochs} - "
                f"Actor loss : {np.mean(self.all_actor_losses):.4f}, "
                f"Critic loss : {np.mean(self.all_critic_losses):.4f}")

        self.memory.clear_memory()

        def plot_learning_curve(self, x, scores, figure_file):
            running_avg = np.zeros(len(scores))
            for i in range(len(running_avg)):
                running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
            plt.plot(x, running_avg)
            plt.title('Running average of previous 100 scores')
            plt.savefig(figure_file)

if __name__ == "__main__":
    import torch
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from Environment import DataLoader, TradingEnv

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    df = DataLoader().read("data/General/TSLA_2019_2024.csv")
    features = df[["Close", "High", "Low", "Open", "Volume"]].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = TradingEnv(df, broker_fee=False)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = 7
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=32, n_epochs=3)

    # -----------------------------
    # Training loop
    # -----------------------------
    n_episodes = 5
    threshold = 0.65
    action_names = {0: "hold", 1: "buy", 2: "sell"}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        seq_buffer = []
        actions_taken = []

        while not done:
            # Build sequence for Transformer
            seq_buffer.append(obs)
            if len(seq_buffer) > seq_len:
                seq_buffer.pop(0)
            if len(seq_buffer) < seq_len:
                pad_len = seq_len - len(seq_buffer)
                seq = [seq_buffer[0]]*pad_len + seq_buffer
            else:
                seq = seq_buffer

            # Get valid actions
            valid_actions = env.get_valid_actions()

            seq_array = np.array(seq, dtype=np.float32)
            seq_array = np.expand_dims(seq_array, axis=0)  # Add batch dimension

            # Choose action
            action, log_prob, value = agent.choose_action(seq_array, valid_actions, threshold=threshold)
            actions_taken.append(action)  # <-- Save action

            # Step environment
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            # Store experience
            agent.remember(seq_array, action, log_prob, value, reward, done)
            obs = next_obs

        # Update agent after each episode
        agent.learn()

        # Count actions and map to names
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        print(f"Episode {ep+1}/{n_episodes} finished, "
            f"total reward: {total_reward:.3f}, "
            f"actions taken: {action_summary}")