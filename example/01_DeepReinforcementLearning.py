from PPO_Library import TradingEnv, DataLoader, DeepQLearning

import torch

def get_env(training_size):
    df_train, df_test = DataLoader().split_train_test("./data/General/^VIX_2015_2025.csv", training_size=training_size)
    env = TradingEnv(df_train)
    return df_train, df_test, env

def train_DeepQLearning(df, env):
    model = DeepQLearning(env, log=True)
    policy_net = model.train(
        df                  = df,
        train_size          = 0.8,
        n_training_episodes = 500,      # much longer training
        learning_rate       = 1e-4,     # smaller LR for stability
        gamma               = 0.99,     # discount factor (keep)
        max_epsilon         = 1.0,      # start fully random
        min_epsilon         = 0.05,     # leave some exploration
        decay_rate          = 0.005,    # faster decay (converge to exploitation earlier)
        buffer_capacity     = 100_000,  # big replay memory
        batch_size          = 128,      # larger batches stabilize updates
        target_update_every = 500,      # update target net more frequently
        learn_every         = 4,        # learn every few steps (stabilizes learning)
        warmup_steps        = 5_000     # collect random transitions before training
    )
    # Save
    torch.save(policy_net.state_dict(), "DL_policy_vix.pth")

def test_DeepQLearning(env):
    model_loaded = DeepQLearning(env)
    model_loaded.policy_net.load_state_dict(torch.load("DL_policy_vix.pth"))
    model_loaded.policy_net.eval()
    return model_loaded


if __name__ == "__main__":
    df_train, df_test, env = get_env(training_size=0.8)
    train_DeepQLearning(df=df_train, env=env)
    model = test_DeepQLearning(df=df_train, env=env)