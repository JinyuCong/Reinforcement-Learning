import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def state_to_index(state, env_size):
    x, y = state
    return x + y * env_size[0]


def epsilon_greedy(epsilon, Q, policy_matrix, state_idx):
    num_actions = policy_matrix.shape[1]
    state_action_probs = epsilon / (np.ones(num_actions) * num_actions)
    state_action_probs[np.argmax(Q[state_idx])] += 1 - epsilon
    return state_action_probs