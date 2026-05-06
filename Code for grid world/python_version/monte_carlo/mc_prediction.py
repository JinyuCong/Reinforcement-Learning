import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from monte_carlo.utils import state_to_index, generate_episode

env = GridWorld()

# ================================================================
# Every-Visit MC Prediction —— 策略评估
# ================================================================
# 目标：给定均匀随机策略，估计每个状态的 V(s)
#
# 算法思路：
#   对每条 episode，从后往前计算每步的 G（回报）
#   G_t = r_{t+1} + γ * G_{t+1}
#   每次访问某状态时，把 G 累加进去，最后取平均
#
# 数据结构：
#   V[s]          : 状态价值估计，初始全 0
#   returns_sum[s]: 所有 episode 中访问 s 时累积的 G 之和
#   counts[s]     : 状态 s 被访问的总次数
# ================================================================

def mc_prediction(env, num_episodes=5000, gamma=0.9):
    num_states = env.num_states
    num_actions = len(env.action_space)

    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    V = np.zeros(num_states)
    returns_sum = np.zeros(num_states)
    counts = np.zeros(num_states)

    for ep in range(num_episodes):
        episode = generate_episode(env, policy_matrix)
        G = 0
        for state, action_index, reward in reversed(episode):
            G = reward + gamma * G
            returns_sum[state] += G
            counts[state] += 1

    V = returns_sum / (counts + 1e-10)
    return V


if __name__ == "__main__":
    V = mc_prediction(env, num_episodes=5000, gamma=0.9)

    state, _ = env.reset()
    env.render()
    env.add_state_values(V)
    env.render(animation_interval=3)
