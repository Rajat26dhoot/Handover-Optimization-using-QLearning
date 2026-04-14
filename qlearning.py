import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return next_state, reward, done, info


def discretize_state(state, low, high, bins):
    state = np.asarray(state, dtype=np.float32)
    scaled_state = (state - low) * bins
    discrete_state = np.round(scaled_state, 0).astype(int)
    max_indices = np.round((high - low) * bins, 0).astype(int)
    return np.clip(discrete_state, 0, max_indices)


def q_learn(env, learn_rate, discount, epsilon, min_eps, episodes):
    bins = np.array([10, 100], dtype=np.float32)
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    num_states = np.round((state_high - state_low) * bins, 0).astype(int) + 1

    # One Q-value per discretized state-action pair.
    q_table = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    epsilon_decay = (epsilon - min_eps) / episodes
    reward_list = []
    total_reward_list = []
    goal_position = getattr(env.unwrapped, "goal_position", 0.5)

    for i in range(episodes):
        done = False
        total_reward = 0
        state = reset_env(env)
        state_adj = discretize_state(state, state_low, state_high, bins)

        while not done:
            if np.random.random() < 1 - epsilon:
                action = np.argmax(q_table[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            next_state, reward, done, _ = step_env(env, action)
            next_state_adj = discretize_state(next_state, state_low, state_high, bins)

            if done and next_state[0] >= goal_position:
                q_table[state_adj[0], state_adj[1], action] = reward
            else:
                delta = learn_rate * (
                    reward
                    + discount * np.max(q_table[next_state_adj[0], next_state_adj[1]])
                    - q_table[state_adj[0], state_adj[1], action]
                )
                q_table[state_adj[0], state_adj[1], action] += delta

            total_reward += reward
            state_adj = next_state_adj

        if epsilon > min_eps:
            epsilon = max(min_eps, epsilon - epsilon_decay)

        reward_list.append(total_reward)
        print(f"Episode {i + 1}/{episodes} completed.")

        if (i + 1) % 100 == 0:
            total_reward_list.append(np.sum(reward_list))
            reward_list = []

    env.close()
    return total_reward_list


def main():
    episodes = int(os.environ.get("QL_EPISODES", "6000"))
    env = gym.make("MountainCar-v0")
    rewards = q_learn(env, learn_rate=0.2, discount=1, epsilon=0.8, min_eps=0, episodes=episodes)

    plt.figure(figsize=(10, 6))
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward (per 100 episodes)")
    plt.title("Reward vs Episodes")
    plt.grid()

    save_folder = "./qlearning_plots/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.savefig(os.path.join(save_folder, "Plot.png"))
    plt.close()


if __name__ == "__main__":
    main()
