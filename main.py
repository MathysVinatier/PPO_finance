import os
import numpy as np
import importlib.util

from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import to_datetime

from PPO_Library import DataLoader, TradingEnv, ACAgent, ModelTest, ModelReport

# CONFIGURATION
SEQUENCE_LENGTH = 5
MODEL_NAME      = "001_ppo_transformer"
MODEL_PATH      = "tmp/data_training/trained_model"

# FUNCTIONS

def model_analysis(task_path, episode):
    task = ModelReport(task_path, seq_size = SEQUENCE_LENGTH)
    # task.plot(show=True)#, save_path=f"episode_{episode}_analysis")
    df_train, df_test = DataLoader().split_train_test(task._dataset_path, training_size=0.8)

    model = task.get_model(model_episode=episode)
    test  = ModelTest(model, df_test, SEQUENCE_LENGTH)
    train = ModelTest(model, df_train, SEQUENCE_LENGTH)

    train.plot(show=True)#, save_path=f"episode_{episode}_train")
    test.plot(show=True)#, save_path=f"episode_{episode}_test")

def show_best_ep_on_test(task_path):
    best_portfolio = 0
    best_episode   = dict()

    task = ModelReport(task_path, SEQUENCE_LENGTH)
    df_train, df_test = DataLoader().split_train_test(task._dataset_path, training_size=0.8)

    for mod in os.listdir(task._models_path):
        episode         = "_".join(mod.split("_")[3:])
        model_test      = ModelTest(task.get_model(episode), df_test)
        final_portfolio = model_test.info["portfolio_values"][-1]

        if final_portfolio > best_portfolio:
            best_episode = episode
            best_model = model_test
            best_portfolio = final_portfolio
            print(f"Best episode is {best_episode} with a {best_portfolio} dollars")

    model_analysis(task_path=task_path, episode=best_episode, seq_size=SEQUENCE_LENGTH)
    return best_model

def PPO_training(epoch, episode, batch_size, df_path):

    df_train, df_test = DataLoader().split_train_test(df_path, training_size=0.8)

    tick = to_datetime(df_train.index).to_series().diff().mode()[0]
    print(f"... starting training from {df_train.index.min()} to {df_train.index.max()} with {tick} tick (threshold={0.6}) ...")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    features = df_train[["Close", "High", "Low", "Open", "Volume"]]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = TradingEnv(df_train, broker_fee=True)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = SEQUENCE_LENGTH
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=epoch, chkpt_dir=MODEL_PATH, agent_id=MODEL_NAME)

    # -----------------------------
    # Training loop
    # -----------------------------
    action_names = {0: "hold", 1: "buy", 2: "sell"}

    for ep in range(episode):
        print(f"Episode {ep+1}/{episode} started")
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
            action, log_prob, value = agent.choose_action(seq_array, valid_actions, threshold=0.6)
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
    print(f"model saved in {MODEL_PATH}/actor_weight_{MODEL_NAME} and {MODEL_PATH}/critic_weight_{MODEL_NAME}")

    return agent, total_reward

def main(args):
    if args.mode == "analysis":
        if args.task is None or args.path is None :
            raise AttributeError("By choosing --mode analysis, you should input a --task and --path")
        else:
            task_path = os.path.join(args.path)
            model_analysis(task_path=task_path, episode=args.task)

    elif args.mode == "training":
        if args.epoch is None or args.episode is None or args.batch is None:
            raise AttributeError("By choosing --mode training, you should input a --episode, --epoch and --batch")
        else:
            PPO_training(
                epoch=int(args.epoch),
                episode=int(args.episode),
                batch_size=int(args.batch),
                df_path="./data/General/^VIX_2015_2025.csv"
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PPO Trading Agent Runner",
        epilog="Example : python main.py -m analysis -t task1 -p ./my_task OR python main.py -m training -e 10 -ep 10 -b 32"
    )

    parser.add_argument(
        "-m", "--mode",
        help="Mode of operation. Choose 'training' or 'analysis'.",
        choices=["training", "analysis"],
        required=True
    )

    parser.add_argument(
        "-t", "--task",
        help="Task name. Required for both analysis modes."
    )

    parser.add_argument(
        "-p", "--path",
        help="Path to the task folder. Required for both analysis modes."
    )

    parser.add_argument(
        "-e", "--epoch",
        type=int,
        default=50,
        help="Number of epochs for training (default: 50)."
    )

    parser.add_argument(
        "-ep", "--episode",
        type=int,
        default=10,
        help="Number of episodes for training (default: 10)."
    )

    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=64,
        help="Batch size for training (default: 64)."
    )


    args = parser.parse_args()
    main(args)