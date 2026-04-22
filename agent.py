import random

import numpy as np


class Agent:
    def __init__(self, rows, cols, actions, signal_availability):
        self.rows = rows
        self.cols = cols
        self.actions = actions
        self.Model = {}
        self.Q = {}
        self.movement_Q = {}
        self.reset(rows, cols, signal_availability)

    def antenna_selection(self, state, eps, signal_availability, Q):
        available_antennas = signal_availability.get(state, [])
        if not available_antennas:
            raise ValueError(f"No antennas available for state {state}.")

        if np.random.rand() < eps:
            return random.choice(available_antennas)

        return max(Q[state], key=Q[state].get)

    def action_selection(self, state, eps, movement_Q):
        if np.random.rand() < eps:
            return random.choice(self.actions)

        return max(movement_Q[state], key=movement_Q[state].get)

    def Q_update(self, state, antenna, next_state, reward, lr, gamma, Q, done=False):
        next_values = list(Q[next_state].values())
        next_max = 0.0 if done or not next_values else max(next_values)
        Q[state][antenna] = (1 - lr) * Q[state][antenna] + lr * (reward + gamma * next_max)

    def movement_Q_update(self, state, action, next_state, reward, lr, gamma, movement_Q, done=False):
        next_max = 0.0 if done else max(movement_Q[next_state].values())
        movement_Q[state][action] = (
            (1 - lr) * movement_Q[state][action]
            + lr * (reward + gamma * next_max)
        )

    def Model_update(self, state, antenna, next_state, reward):
        if state not in self.Model:
            self.Model[state] = {}
        self.Model[state][antenna] = (next_state, reward)

    def n_step_Q_update(self, n, lr, gamma, Q):
        for _ in range(n):
            rand_s = np.random.randint(len(self.Model.keys()))
            random_state = list(self.Model)[rand_s]
            rand_a = np.random.randint(len(self.Model[random_state].keys()))
            random_action = list(self.Model[random_state])[rand_a]
            next_state_r, reward_r = self.Model[random_state][random_action]
            next_max = max(list(Q[random_state].values()))
            Q[random_state][random_action] = (
                (1 - lr) * Q[random_state][random_action]
                + lr * (reward_r + gamma * next_max)
            )

    def reset(self, rows, cols, signal_availability):
        self.Q = {}
        self.movement_Q = {}
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                self.Q[state] = {ant: 0.0 for ant in signal_availability.get(state, [])}
                self.movement_Q[state] = {action: 0.0 for action in self.actions}
        self.Model = {}
