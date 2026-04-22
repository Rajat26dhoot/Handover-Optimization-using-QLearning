import numpy as np


class GridWorld:
    def __init__(self, rows, cols, start, goal, antennas, pos):
        self.action_space = ["r", "u", "l", "d"]
        self.rows = rows
        self.cols = cols
        self.state = start
        self.Start = start
        self.Goal = goal
        self.Antennas = antennas
        self.done = False
        self.goal_reward = 100.0
        self.progress_reward = 3.0
        self.step_penalty = -0.5
        self.handover_penalty = -1.5
        self.stable_link_bonus = 0.75
        self.invalid_move_penalty = -2.5
        self.grid = np.zeros((self.rows, self.cols))
        self.grid[self.Goal] = 2
        for i in pos:
            self.grid[i] = -1

    def reset(self):
        self.state = self.Start
        self.done = False

    def _distance_to_goal(self, state):
        return abs(state[0] - self.Goal[0]) + abs(state[1] - self.Goal[1])

    def _select_next_antenna(self, state, eps, signal_availability, Q):
        available_antennas = signal_availability.get(state, [])
        if not available_antennas:
            return None

        if np.random.rand() < eps:
            return np.random.choice(available_antennas)

        q_values = Q.get(state, {})
        if not q_values:
            return available_antennas[0]

        return max(q_values, key=q_values.get)

    def step(self, action, antenna, eps, signal_availability, Q):
        current_state = self.state
        old_antenna = antenna
        old_distance = self._distance_to_goal(current_state)
        r, c = current_state

        if action == "u":
            r -= 1
        elif action == "d":
            r += 1
        elif action == "r":
            c += 1
        elif action == "l":
            c -= 1

        proposed_state = (r, c)
        invalid_move = not (0 <= r < self.rows and 0 <= c < self.cols)
        if invalid_move:
            next_state = current_state
        else:
            next_state = proposed_state

        self.state = next_state
        new_distance = self._distance_to_goal(self.state)
        new_antenna = self._select_next_antenna(self.state, eps, signal_availability, Q)
        if new_antenna is None:
            new_antenna = old_antenna

        handover = 1 if old_antenna != new_antenna else 0

        if self.state == self.Goal:
            reward = self.goal_reward
            reward += self.handover_penalty if handover else self.stable_link_bonus
            self.done = True
        else:
            reward = self.step_penalty
            reward += self.progress_reward * (old_distance - new_distance)
            reward += self.handover_penalty if handover else self.stable_link_bonus
            if invalid_move:
                reward += self.invalid_move_penalty

        info = {
            "serving_antenna": new_antenna,
            "invalid_move": invalid_move,
            "distance_delta": old_distance - new_distance,
        }
        return self.state, reward, handover, self.done, info

    def show_grid(self):
        self.grid[self.Start] = 1
        for r in range(self.rows):
            print(" -------------------------")
            output = ""
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    value = "M"
                elif self.grid[r, c] == 0:
                    value = "0"
                elif self.grid[r, c] == -1:
                    value = "A"
                elif self.grid[r, c] == 2:
                    value = "G"
                output += " | " + value
            print(output + " | ")
        print(" -------------------------")
