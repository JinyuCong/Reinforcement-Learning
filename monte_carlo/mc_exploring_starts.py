import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from monte_carlo.utils import state_to_index

env = GridWorld()

# ================================================================
# MC Exploring Starts —— 在线更新，随机起点保证覆盖
# ================================================================
# 与 MC Basic 的核心区别：
#   - MC Basic：跑完 num_episodes 条后做一次策略改进（批量更新）
#   - Exploring Starts：每跑完一条 episode 就立刻做策略改进（在线更新）
#   - Exploring Starts：每条 episode 的起始 (s, a) 随机选取，
#     保证所有 (s, a) 对都有机会被访问到
#
# 算法思路：
#   for ep in range(num_episodes):
#     1. 随机选 start_state 和 start_action_idx
#     2. 强制设置 env.agent_state 和 env.traj，执行第一步
#     3. 用当前策略跑完剩余步骤（最多200步）
#     4. 从后往前算 G，更新 Q，每步立刻做策略改进（one-hot argmax Q[s]）
# ================================================================

def mc_exploring_starts(env, num_episodes=2000, gamma=0.9):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    counts = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    all_states = [
        (x, y)
        for x in range(env.env_size[0])
        for y in range(env.env_size[1])
        if (x, y) != env.target_state and (x, y) not in env.forbidden_states
    ]
    state_p = np.ones(len(all_states)) / len(all_states)

    for ep in range(num_episodes):
        start_state_idx = np.random.choice(len(all_states), p=state_p)
        start_state = all_states[start_state_idx]
        start_s_idx = state_to_index(start_state, env.env_size)
        start_action_idx = np.random.choice(num_actions, p=policy_matrix[start_s_idx])
        start_action = env.action_space[start_action_idx]

        env.agent_state = start_state
        env.traj = [env.agent_state]

        episode = []
        done = False
        while not done and len(episode) < 200:
            next_state, reward, done, _ = env.step(start_action)
            next_state_idx = state_to_index(next_state, env.env_size)
            next_action_idx = np.random.choice(num_actions, p=policy_matrix[next_state_idx])
            next_action = env.action_space[next_action_idx]

            episode.append((start_s_idx, start_action_idx, reward))

            start_s_idx = next_state_idx
            start_action_idx = next_action_idx
            start_state = next_state
            start_action = next_action

        G = 0
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            returns_sum[state, action] += G
            counts[state, action] += 1
            Q[state, action] = returns_sum[state, action] / counts[state, action]

            policy_matrix[state] = np.zeros(num_actions)
            policy_matrix[state, np.argmax(Q[state])] = 1

    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = mc_exploring_starts(env)

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
