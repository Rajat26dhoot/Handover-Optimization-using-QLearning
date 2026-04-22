import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sample_indices(length, sample_rate):
    if length <= 0:
        return []

    indices = list(range(0, length, max(1, sample_rate)))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def plot_metrics(Episodes, Rewards, Handovers, steps_taken, success_flags, epsilon_history, save_folder, sample_rate=100):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    sampled_indices = _sample_indices(Episodes, sample_rate)
    sampled_episodes = np.array(sampled_indices) + 1
    sampled_rewards = np.array(Rewards)[sampled_indices]
    sampled_handovers = np.array(Handovers)[sampled_indices]
    sampled_steps = np.array(steps_taken)[sampled_indices]
    sampled_eps_values = np.array(epsilon_history)[sampled_indices]

    cumulative_rewards = np.cumsum(Rewards)
    moving_avg_rewards = [np.mean(Rewards[max(0, i - 10):i + 1]) for i in range(len(Rewards))]
    cumulative_success_rate = np.cumsum(success_flags) / (np.arange(Episodes) + 1)
    sampled_cumulative_rewards = cumulative_rewards[sampled_indices]
    sampled_moving_avg_rewards = np.array(moving_avg_rewards)[sampled_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_rewards, label="Rewards per Episode")
    plt.plot(
        sampled_episodes,
        sampled_moving_avg_rewards,
        label="Moving Average of Rewards (window=10)",
        linestyle="--",
    )
    plt.title("Rewards vs. Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Rewards_vs_Episodes_Sampled.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_handovers, label="Handovers per Episode", color="orange")
    plt.title("Handovers vs. Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Handovers")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Handovers_vs_Episodes_Sampled.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, Episodes + 1), cumulative_success_rate, label="Cumulative Success Rate", color="red")
    plt.title("Success Rate Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Success Ratio")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Cumulative_Success.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_steps, label="Steps per Episode", color="green")
    plt.title("Steps Taken per Episode (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Steps_vs_Episodes_Sampled.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_cumulative_rewards, label="Cumulative Rewards", color="purple")
    plt.title("Cumulative Rewards over Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Cumulative_Rewards_Sampled.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(Handovers, bins=50, color="skyblue", edgecolor="black")
    plt.title("Histogram of Handovers")
    plt.xlabel("Number of Handovers")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_folder, "Histogram_of_Handovers.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    episode_progress = np.linspace(0, 1, len(Rewards))
    scatter = plt.scatter(
        Handovers,
        Rewards,
        c=episode_progress,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Episode Progress")
    plt.title("Reward vs. Handovers Trade-off")
    plt.xlabel("Number of Handovers")
    plt.ylabel("Episode Reward")
    plt.savefig(os.path.join(save_folder, "Reward_vs_Handovers_Tradeoff.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_episodes, sampled_eps_values, label="Epsilon Decay", color="brown")
    plt.title("Epsilon Decay over Episodes (Sampled)")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Epsilon_Decay_Sampled.png"))
    plt.close()
