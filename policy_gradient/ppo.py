import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from src.grid_world import GridWorld

env = GridWorld()

# ================================================================
# PPO —— Proximal Policy Optimization（Clip 版本）
# ================================================================
# Actor-Critic 每步更新一次，数据效率低，且更新步长难以控制：
#   步长太大 → 策略崩塌（新旧策略差异太大，训练不稳定）
#   步长太小 → 收敛慢
#
# PPO 的核心思想：收集一批数据后，对同一批数据做多次梯度更新，
# 但用 Clip 限制新旧策略的比率，防止更新过大：
#
#   r_t(θ) = π(a_t|s_t; θ) / π(a_t|s_t; θ_old)   ← 重要性采样比率
#
#   L_CLIP = E[ min(r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t) ]
#
#   其中 A_t = δ_t = r + γV(s') - V(s) 是优势函数估计
#   ε（clip_eps）通常取 0.2
#
# 算法流程：
#   for update in range(num_updates):
#     用旧策略 π_old 收集 steps_per_update 步数据：
#       存储 (s, a, r, done, log_prob_old, V(s))
#     计算每步的优势 A_t 和 return G_t
#     对收集的数据做 K 轮（ppo_epochs）mini-batch 更新：
#       r_t = exp(log_π_new(a|s) - log_π_old(a|s))
#       L_clip  = min(r_t·A, clip(r_t, 1-ε, 1+ε)·A)
#       L_value = (V(s) - G_t)²
#       loss = -L_clip + c·L_value
#       反向传播更新
#
# 与 Actor-Critic 的区别：
#   A-C  : 每步更新一次，on-policy，数据用一次就丢
#   PPO  : 收集一批数据后多次更新（ppo_epochs），clip 保证稳定性
#
# 数据结构：
#   actor  : 策略网络（softmax 输出）
#   critic : 值函数网络（scalar 输出）
# ================================================================

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        # 你来写

    def forward(self, x):
        # 你来写
        pass


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # 你来写

    def forward(self, x):
        # 你来写
        pass


def state_to_tensor(state, env_size):
    # 你来写
    pass


def ppo(env, num_updates=500, steps_per_update=256,
        lr=3e-4, gamma=0.99, clip_eps=0.2,
        ppo_epochs=4, batch_size=64, value_coef=0.5):
    # 你来写
    pass


if __name__ == "__main__":
    actor = ppo(env, num_updates=500)
