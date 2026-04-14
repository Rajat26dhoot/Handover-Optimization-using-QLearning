import argparse

import numpy as np

from agent import Agent
from dataset_loader import build_dataset_environment, format_dataset_summary
from gridworld import GridWorld
from plotter import plot_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the handover optimization agent.")
    parser.add_argument("--dataset-root", help="Path to the 5G production dataset root.")
    parser.add_argument("--rows", type=int, help="Number of spatial grid rows for dataset-backed runs.")
    parser.add_argument("--cols", type=int, help="Number of spatial grid columns for dataset-backed runs.")
    parser.add_argument(
        "--mobility-mode",
        default="Driving",
        help="Dataset mobility mode filter, such as Driving or Static.",
    )
    parser.add_argument(
        "--network-mode",
        action="append",
        dest="network_modes",
        help="Repeat to keep specific radio modes, for example --network-mode 5G.",
    )
    parser.add_argument(
        "--app",
        action="append",
        dest="apps",
        help="Repeat to keep specific app folders, such as Netflix or Download.",
    )
    parser.add_argument("--limit-files", type=int, help="Use only the first N dataset files after filtering.")
    parser.add_argument("--top-k-cells", type=int, default=5, help="Keep the top K raw cells per spatial bin.")
    parser.add_argument(
        "--min-cell-observations",
        type=int,
        default=10,
        help="Minimum observations before a raw cell is kept directly in a bin.",
    )
    parser.add_argument("--episodes", type=int, default=6000, help="Training episodes.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.82, help="Initial epsilon.")
    parser.add_argument("--max-steps", type=int, help="Maximum steps per episode.")
    parser.add_argument("--plot-dir", default="./metrics_plots", help="Directory for training plots.")
    return parser.parse_args()


def build_demo_environment():
    rows = 3
    cols = 6
    start = (2, 0)
    goal = (0, 5)
    antennas = {
        "A2": {"position": (2, 1), "Range": [(2, 0), (1, 1), (2, 2), (1, 0), (1, 2), (2, 1)]},
        "A1": {"position": (0, 0), "Range": [(1, 0), (1, 1), (0, 1), (0, 0)]},
        "A3": {"position": (0, 2), "Range": [(0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2)]},
        "A4": {"position": (1, 3), "Range": [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (1, 4), (0, 4), (0, 3), (1, 3)]},
        "A5": {"position": (0, 4), "Range": [(0, 3), (1, 3), (1, 4), (1, 5), (0, 5), (0, 4)]},
        "A6": {"position": (2, 5), "Range": [(1, 4), (2, 4), (1, 5), (2, 5)]},
    }
    antenna_positions = [(0, 0), (2, 1), (0, 2), (1, 3), (0, 4), (2, 5)]

    signal_availability = {
        (row, col): [antenna for antenna in antennas if (row, col) in antennas[antenna]["Range"]]
        for row in range(rows)
        for col in range(cols)
    }

    return {
        "rows": rows,
        "cols": cols,
        "start": start,
        "goal": goal,
        "antennas": antennas,
        "antenna_positions": antenna_positions,
        "signal_availability": signal_availability,
        "summary": "Using the original hand-crafted 3x6 demo environment.",
    }


def build_environment_from_args(args):
    if not args.dataset_root:
        return build_demo_environment()

    dataset_rows = args.rows or 10
    dataset_cols = args.cols or 10
    network_modes = args.network_modes or ["5G"]

    dataset_config = build_dataset_environment(
        dataset_root=args.dataset_root,
        rows=dataset_rows,
        cols=dataset_cols,
        mobility_mode=args.mobility_mode,
        network_modes=network_modes,
        apps=args.apps,
        limit_files=args.limit_files,
        top_k_cells_per_bin=args.top_k_cells,
        min_cell_observations=args.min_cell_observations,
    )

    return {
        "rows": dataset_config.rows,
        "cols": dataset_config.cols,
        "start": dataset_config.start,
        "goal": dataset_config.goal,
        "antennas": dataset_config.antennas,
        "antenna_positions": dataset_config.antenna_positions,
        "signal_availability": dataset_config.signal_availability,
        "summary": format_dataset_summary(dataset_config),
    }


def train_agent(args, environment_config):
    rows = environment_config["rows"]
    cols = environment_config["cols"]
    start = environment_config["start"]
    goal = environment_config["goal"]
    antennas = environment_config["antennas"]
    antenna_positions = environment_config["antenna_positions"]
    signal_availability = environment_config["signal_availability"]

    episodes = args.episodes
    lr = args.lr
    gamma = args.gamma
    epsilon = args.epsilon
    max_steps_per_episode = args.max_steps or max(150, rows * cols * 2)
    eps_decaying_start = 0
    eps_decaying_end = max(1, int(episodes / 1.5))
    eps_decaying = epsilon / max(1, eps_decaying_end - eps_decaying_start)

    env = GridWorld(rows, cols, start, goal, antennas, antenna_positions)
    agent = Agent(rows, cols, env.action_space, signal_availability)
    agent.reset(rows, cols, signal_availability)

    rewards = []
    steps_taken = []
    handovers = []

    print(environment_config["summary"])

    for episode in range(episodes):
        print(f"Episode: {episode}")
        if eps_decaying_start < episode < eps_decaying_end:
            epsilon -= eps_decaying

        counter = 0
        env.reset()
        done = False
        state = env.state
        reward_ep = 0
        handover_ep = 0

        while not done and counter < max_steps_per_episode:
            action = agent.action_selection(state)
            antenna = agent.antenna_selection(state, action, epsilon, signal_availability, agent.Q)
            next_state, reward, handover, done, _ = env.step(action, antenna, epsilon, signal_availability, agent.Q)
            agent.Q_update(state, antenna, next_state, reward, lr, gamma, agent.Q)
            reward_ep += reward
            handover_ep += handover
            print(
                f"Episode: {episode} | State: {state} -> Next State: {next_state}, "
                f"Reward: {reward}, Accumulated Rewards: {reward_ep}, Handovers: {handover_ep}"
            )
            agent.Model_update(state, antenna, next_state, reward)
            state = next_state
            counter += 1

        rewards.append(reward_ep)
        handovers.append(handover_ep)
        steps_taken.append(counter)
        print("")

    plot_metrics(episodes, rewards, handovers, steps_taken, epsilon, eps_decaying, args.plot_dir)

    print(f"Total Episodes: {episodes}")
    print(f"Average Reward per Episode: {np.mean(rewards):.2f}")
    print(f"Total Handovers: {sum(handovers)}")
    print(f"Average Steps per Episode: {np.mean(steps_taken):.2f}")
    print(f"Final Epsilon Value: {epsilon:.4f}\n")


def main():
    args = parse_args()
    environment_config = build_environment_from_args(args)
    train_agent(args, environment_config)


if __name__ == "__main__":
    main()
