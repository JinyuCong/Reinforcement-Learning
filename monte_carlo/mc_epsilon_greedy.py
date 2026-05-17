import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from monte_carlo.utils import state_to_index, generate_episode

env = GridWorld()

# ================================================================
# MC ε-greedy —— On-Policy MC Control
# ================================================================
# 与 Exploring Starts 的核心区别：
#   - Exploring Starts 靠随机起点保证覆盖，策略是确定性的
#   - ε-greedy 靠每步以 ε 概率随机探索保证覆盖，不需要随机起点
#   - 策略始终是软性的（soft policy）
#
# 算法思路：
#   for ep in range(num_episodes):
#     1. 用当前 ε-greedy 策略跑一条完整 episode
#     2. 从后往前算 G，更新 Q[s,a]
#     3. 每步立刻做 ε-greedy 策略改进：
#        policy[s] = ε/num_actions
#        policy[s][argmax Q[s]] += 1 - ε
#     4. ε 随 episode 线性衰减
# ================================================================

def mc_epsilon_greedy(env, num_episodes=5000, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    counts = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
        episode = generate_episode(env, policy_matrix)
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            returns_sum[state, action] += G
            counts[state, action] += 1
            Q[state, action] = returns_sum[state, action] / counts[state, action]

            policy_matrix[state] = epsilon / num_actions
            policy_matrix[state, np.argmax(Q[state])] += 1 - epsilon

    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = mc_epsilon_greedy(env)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        s_idx = state_to_index(state, env.env_size)
        action = env.action_space[np.argmax(best_policy[s_idx])]
        state, reward, done, _ = env.step(action)
        env.render()

    env.add_policy(best_policy)
    env.render(animation_interval=3)
