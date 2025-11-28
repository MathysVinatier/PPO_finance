from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import to_datetime

from PPO_Library import DataLoader, TradingEnv, ACAgent, ModelTest


def train_ppo(df, n_episode, n_epoch_per_episode, batch_size, gamma, alpha, gae,
                 policy_clip, training_seq, checkpoint_step=False, threshold = 0.65):

    tick = to_datetime(df.index).to_series().diff().mode()[0]
    print(f"... starting training from {df_train.index.min()} to {df_train.index.max()} with {tick} tick (threshold={threshold}) ...")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    features = df_train[["Close", "High", "Low", "Open", "Volume"]]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = TradingEnv(df, broker_fee=True)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = training_seq
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=n_epoch_per_episode, gamma=gamma, alpha=alpha, gae_lambda=gae,
                 policy_clip=policy_clip, chkpt_dir="tmp/")

    # -----------------------------
    # Training loop
    # -----------------------------
    action_names = {0: "hold", 1: "buy", 2: "sell"}

    for ep in range(n_episode):
        print(f"Episode {ep+1}/{n_episode} started")
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
            actions_taken.append(action)

            # Step environment
            next_obs, reward, done, _ = env.step(action)
            total_reward = reward

            # Store experience
            agent.remember(seq_array, action, log_prob, value, reward, done)
            obs = next_obs

        # Update agent after each episode
        agent.learn()

        # Count actions and map to names
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_summary = {action_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        print(f"Episode finished, "
                f"total reward  : {total_reward:.3f}, "
                f"actions taken : {action_summary}")

    agent.save_models()

    return agent, total_reward


if __name__ == "__main__":

    df_train, df_test = DataLoader().split_train_test("./data/General/^VIX_2015_2025.csv", training_size=0.8)

    training_sequence = 5

    agent_trained, total_reward = train_ppo(
        df = df_train, 
        n_episode = 10,
        n_epoch_per_episode = 10,
        batch_size = 32,
        gamma = 0.3,
        alpha = 0.3,
        gae = 0.3,
        policy_clip = 0.3,
        training_seq = training_sequence
    )

    results_test  = ModelTest(agent_trained, df_train, training_sequence)
    results_train = ModelTest(agent_trained, df_test, training_sequence)

    results_train.plot(show=True)
    results_test.plot(show=True)
