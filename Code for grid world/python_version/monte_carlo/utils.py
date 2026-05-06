import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def state_to_index(state, env_size):
    x, y = state
    return x + y * env_size[0]


def generate_episode(env, policy_matrix):
    episode = []
    state = env.reset()[0]
    max_steps = 200

    for _ in range(max_steps):
        state_index = state_to_index(state, env_size=env.env_size)
        action_index = np.random.choice(len(env.action_space), p=policy_matrix[state_index])
        action = env.action_space[action_index]
        next_state, reward, done, _ = env.step(action)
        episode.append((state_index, action_index, reward))
        state = next_state
        if done:
            break

    return episode
