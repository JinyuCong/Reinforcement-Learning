import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def state_to_index(state, env_size):
    x, y = state
    return x + y * env_size[0]
