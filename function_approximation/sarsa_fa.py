import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from function_approximation.utils import get_features, epsilon_greedy_fa, NUM_FEATURES

env = GridWorld()

# ================================================================
# Sarsa with Linear Function Approximation —— On-Policy TD Control
# ================================================================
# 用线性函数近似代替 Q 表：
#   Q(s, a; W) = φ(s)^T W[:, a]
#   φ(s) ∈ R^d 是状态特征向量（多项式特征，d=6）
#   W ∈ R^{d × num_actions} 是待学习的权重矩阵
#
# 与表格型 Sarsa 的对比：
#   表格型：直接更新 Q[s, a]
#   函数近似：更新 W[:, a]（参数共享，可泛化到未见过的状态）
#
# TD 误差（和 Sarsa 完全一样）：
#   δ = r + γ * Q(s', a'; W) - Q(s, a; W)
#
# 参数更新（梯度上升方向 = φ(s)）：
#   W[:, a] += α * δ * φ(s)
#
# 算法思路：
#   for ep in range(num_episodes):
#     s = env.reset()，计算 φ(s)
#     用 ε-greedy（基于 W）在 φ(s) 上选 a
#     while not done（最多 200 步）:
#       执行 a，得到 r, s'，计算 φ(s')
#       用 ε-greedy 在 φ(s') 上选 a'        ← on-policy，先选好再更新
#       δ = r + γ * φ(s')^T W[:,a'] - φ(s)^T W[:,a]
#       W[:, a] += α * δ * φ(s)             ← 只更新动作 a 对应的列
#       s ← s'，a ← a'，φ ← φ(s')
# ===========
def sarsa_fa(env, num_episodes=5000, alpha=0.01, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_actions = len(env.action_space)

    # 权重矩阵 W：shape (NUM_FEATURES, num_actions)，初始全 0
    W = np.zeros((NUM_FEATURES, num_actions))

    # 你来写
    # 提示：
    #   get_features(state, env.env_size)    → φ(s)，shape (NUM_FEATURES,)
    #   epsilon_greedy_fa(epsilon, W, phi, num_actions) → 动作概率
    #   Q(s, a) = phi @ W[:, a]  或  phi @ W  得到所有动作的 Q 值
    #   ε 线性衰减：epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
    for ep in range(num_episodes):
        # ε 线性衰减
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)

        # 初始state
        state, _ = env.reset()

        # 通过state得到特征phi
        phi = get_features(state, env.env_size)

        # 用 ε-greedy（基于 W）在 φ(s) 上选 a
        probs = epsilon_greedy_fa(epsilon, W, phi, num_actions)
        action_idx = np.random.choice(num_actions, p=probs)
        
        done = False
        step = 0
        while not done and step < 200:
            action = env.action_space[action_idx]

            # 执行 a，得到 r, s'，计算 φ(s')
            next_state, reward, done, _ = env.step(action)
            next_phi = get_features(next_state, env.env_size)

            # 用 ε-greedy 在 φ(s') 上选 a'        ← on-policy，先选好再更新
            probs = epsilon_greedy_fa(epsilon, W, next_phi, num_actions)
            next_action_idx = np.random.choice(num_actions, p=probs)
            
            # δ = r + γ * φ(s')^T W[:,a'] - φ(s)^T W[:,a]（终止状态无未来 Q）
            td_error = reward + gamma * (1 - done) * next_phi @ W[:, next_action_idx] - phi @ W[:, action_idx]

            # W[:, a] += α * δ * φ(s)             ← 只更新动作 a 对应的列
            W[:, action_idx] += alpha * td_error * phi

            # s ← s'，a ← a'，φ ← φ(s')
            state = next_state
            action_idx = next_action_idx
            phi = next_phi

            step += 1


    return W


if __name__ == "__main__":
    W = sarsa_fa(env, num_episodes=5000)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        phi = get_features(state, env.env_size)
        action_idx = np.argmax(phi @ W)
        state, reward, done, _ = env.step(env.action_space[action_idx])
        env.render()

    # 从 W 重建 policy_matrix 以便可视化
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy_matrix = np.zeros((num_states, num_actions))
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        phi = get_features((x, y), env.env_size)
        best_a = np.argmax(phi @ W)
        policy_matrix[s, best_a] = 1.0
    env.add_policy(policy_matrix)
    env.render(animation_interval=30)
