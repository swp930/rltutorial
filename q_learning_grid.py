import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random

# Environment
grid_size = 5
actions = ['up', 'down', 'left', 'right']
action_map = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}
goal = (4, 4)

# Q-table: (row, col, action)
Q = np.zeros((grid_size, grid_size, len(actions)))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, len(actions)-1)
    return np.argmax(Q[state[0], state[1]])

def take_action(state, action_idx):
    action = actions[action_idx]
    move = action_map[action]
    new_row = max(0, min(grid_size - 1, state[0] + move[0]))
    new_col = max(0, min(grid_size - 1, state[1] + move[1]))
    next_state = (new_row, new_col)
    reward = 1 if next_state == goal else 0
    return next_state, reward

# Q-learning training
for ep in range(episodes):
    state = (0, 0)
    while state != goal:
        action = choose_action(state)
        next_state, reward = take_action(state, action)
        best_next_action = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action] += alpha * (reward + gamma * best_next_action - Q[state[0], state[1], action])
        state = next_state

# Visualization function
def visualize_path(Q):
    state = (0, 0)
    path = [state]
    while state != goal:
        action = np.argmax(Q[state[0], state[1]])
        next_state, _ = take_action(state, action)
        if next_state == state:
            break  # prevent infinite loop if stuck
        path.append(next_state)
        state = next_state
    return path

def draw_grid(path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size+1))
    ax.set_yticks(np.arange(0, grid_size+1))
    ax.grid(True)

    # Draw goal
    ax.add_patch(patches.Rectangle((goal[1], grid_size - 1 - goal[0]), 1, 1, facecolor='green'))

    # Draw path
    for idx, (r, c) in enumerate(path):
        ax.add_patch(patches.Circle((c + 0.5, grid_size - 1 - r + 0.5), 0.2, color='blue'))
        if idx > 0:
            prev = path[idx - 1]
            dx = c - prev[1]
            dy = prev[0] - r
            ax.arrow(prev[1] + 0.5, grid_size - 1 - prev[0] + 0.5, dx, dy,
                     head_width=0.1, head_length=0.1, fc='black', ec='black')

    plt.title("Learned Path from (0,0) to Goal (4,4)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Show result
path = visualize_path(Q)
draw_grid(path)