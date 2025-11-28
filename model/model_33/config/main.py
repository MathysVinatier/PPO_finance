
import os
import signal
import multiprocessing

from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import to_datetime

from config import *
from database import *

from optimization import OptunaAPI

from PPO_Library import DataLoader, TradingEnv, ACAgent, ModelTest

DF_TRAIN, DF_TEST = DataLoader().split_train_test(DATASET_PATH)

def train_ppo(agent_id, n_episode, n_epoch_per_episode, batch_size, gamma, alpha, gae,
                 policy_clip,checkpoint_step=False, threshold = 0.65):

    df = DF_TRAIN

    tick = to_datetime(df.index).to_series().diff().mode()[0]
    print(f"... starting training from {df.index.min()} to {df.index.max()} with {tick} tick (threshold={threshold}) ...")

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    create_trial_table(agent_id)
    features = df[["Close", "High", "Low", "Open", "Volume"]]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # -----------------------------
    # Create environment
    # -----------------------------
    env = TradingEnv(df, broker_fee=True)

    # -----------------------------
    # Initialize ACAgent
    # -----------------------------
    seq_len = TRAINING_SEQUENCE
    num_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    train_path = os.path.join(MODELS_PATH, f"trial_{agent_id}")
    os.makedirs(train_path, exist_ok=True)
    agent = ACAgent(n_actions=n_actions, num_features=num_features, seq_len=seq_len, batch_size=batch_size, n_epochs=n_epoch_per_episode, gamma=gamma, alpha=alpha, gae_lambda=gae,
                 policy_clip=policy_clip, chkpt_dir=train_path, agent_id=agent_id)

    # -----------------------------
    # Training loop
    # -----------------------------
    action_names = {0: "hold", 1: "buy", 2: "sell"}

    for ep in range(n_episode):
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

        print(f"Episode {ep+1}/{n_episode} finished, "
                f"total reward  : {total_reward:.3f}, "
                f"actions taken : {action_summary}")

    agent.save_models()
    save_episode_to_trial(
        trial_name          = agent_id,
        episode_number      = ep + 1,
        mean_actor_loss     = np.mean(agent.all_actor_losses),
        mean_critic_loss    = np.mean(agent.all_critic_losses),
        actions_dict        = action_summary,
        final_reward        = total_reward
    )

    return agent, total_reward

def objective(trial):
    """
    Optuna objective: suggests hyperparameters, trains once and returns a scalar metric.
    """

    # --- suggestions ---
    params = {
        "n_training_episodes": trial.suggest_int("n_training_episodes", 50, 200),
        "n_epoch":             trial.suggest_int("n_epoch", 10, 200),
        "batch_size":          trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 1.0, step=0.01)
    }

    # Make sure DB exists
    init_db()

    # Name of this trial (matches Optuna’s internal one)
    trial_name = f"trial_{trial.number:05}"

    # Create DB table for this trial
    create_trial_table(trial_name)

    # Train
    agent_trained, reward = train_ppo(
        agent_id            = trial_name,
        n_episode           = params["n_training_episodes"],
        n_epoch_per_episode = params["n_epoch"],
        batch_size          = params["batch_size"],
        gamma               = 0.99,
        alpha               = params["learning_rate"],
        gae                 = 0.95,
        policy_clip         = 0.2,
        checkpoint_step     = params["n_training_episodes"]
    )

    # --- Evaluate ---
    results_train = ModelTest(agent_trained, DF_TEST)
    results_test  = ModelTest(agent_trained, DF_TRAIN)

    loss_actor_mean  = np.mean(agent_trained.all_actor_losses) if len(agent_trained.all_actor_losses)>0 else 0.0
    loss_critic_mean = np.mean(agent_trained.all_critic_losses) if len(agent_trained.all_critic_losses)>0 else 0.0

    save_trial(
        trial_name         = trial_name,
        params             = params,
        trained_reward     = results_train.info.get('total_reward', 0.0),
        loss_actor_mean    = loss_actor_mean,
        loss_critic_mean   = loss_critic_mean,
        tested_reward      = results_test.info.get("total_reward", 0.0),
        annual_return      = results_test.info.get("annual_return", 0.0),
        average_profit     = results_test.info.get("average_profit", 0.0),
        annual_volatility  = results_test.info.get("annual_volatility", 0.0),
        sharpe_ratio       = results_test.info.get("sharpe_ratio", 0.0),
        max_drawdown       = results_test.info.get("max_drawdown", 0.0)
    )

    print(f"[Objective] Trial {trial_name} done, returning average_profit = {results_test.info.get('average_profit', 0.0)}")

    return float(results_test.info.get("average_profit", 0.0))


def main(n_trial, n_worker):
    # Always start fresh, with a clean process group
    try:
        # Create a new session (so all subprocesses are in the same group)
        os.setsid()
    except Exception:
        pass

    # --- Graceful termination handler ---
    def terminate_all(signum, frame):
        print(f"\n[!] Received signal {signum}, terminating all processes...")
        try:
            os.killpg(os.getpgid(0), signal.SIGTERM)
        except Exception as e:
            print(f"[WARN] Could not kill process group: {e}")
        finally:
            os._exit(0)

    # Register termination handlers
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, terminate_all)

    # --- Start multiprocessing ---
    multiprocessing.set_start_method("spawn", force=True)

    from optimization import OptunaAPI
    api = OptunaAPI(objective)

    print(f"[Optuna] Launching optimization with {n_trial} trials and {n_worker} workers")
    api.optimization(
        n_workers   = n_worker,
        n_trials    = n_trial,
        storage_url = OPTUNA_DB_PATH
    )

    print("[Optuna] Optimization finished.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training configuration")

    # Training arguments
    parser.add_argument("--trial" , type=int, required=True, default=N_TRIALS, help="Number of epochs")
    parser.add_argument("--worker", type=int, required=True, default=N_WORKERS, help="Number of episodes")

    args = parser.parse_args()

    main(n_trial=args.trial , n_worker=args.worker)