import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from td.utils import state_to_index

env = GridWorld()

# ================================================================
# TD(0) —— 策略评估
# ================================================================
# 目标：给定一个固定策略，估计每个状态的 V(s)
#
# 与 MC Prediction 的区别：
#   MC：跑完整条 episode 才更新，用真实回报 G
#   TD：每走一步就更新，用"当前奖励 + 下一状态的估计值"代替 G
#
# 更新公式（每步执行）：
#   V(s) ← V(s) - α * [V(s) - (r + γ * V(s'))]
#   其中 r + γ * V(s') 叫做 TD target
#        V(s) - (r + γ * V(s')) 叫做 TD error (δ)
#
# 算法思路：
#   for ep in range(num_episodes):
#     state = env.reset()
#     while not done（最多200步）:
#       按策略选动作，执行 env.step 得到 r, s'
#       V[s] -= alpha * (V[s] - (r + gamma * V[s']))
#       s ← s'
# ================================================================

def td_zero(env, num_episodes=5, alpha=0.1, gamma=0.9):
    num_states = env.num_states
    num_actions = len(env.action_space)

    # 均匀随机策略
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # V[s]：状态价值，初始全 0
    V = np.zeros(num_states)
    
    env_size = env.env_size

    # 你来写
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            state_idx = state_to_index(state, env_size)
            action_idx = np.random.choice(len(env.action_space), p=policy_matrix[state_idx])
            action = env.action_space[action_idx]
            
            next_state, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_state, env_size)
            
            V[state_idx] -= alpha * (V[state_idx] - (reward + gamma * V[next_state_idx]))
            state = next_state
            
            step += 1
        
        print(f"Episode {ep+1} | State Value : {np.sum(V)}")

    return V


if __name__ == "__main__":
    V = td_zero(env, num_episodes=500)

    state, _ = env.reset()
    env.render()
    env.add_state_values(V)
    env.render(animation_interval=30)
