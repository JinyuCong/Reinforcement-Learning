import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from policy_gradient.utils import LinearActor, LinearCritic

env = gym.make("CartPole-v1", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# A2C —— Advantage Actor-Critic
# ================================================================
# 单步 Actor-Critic 每步更新，方差仍然较大。
# A2C 收集 N 步数据后再统一更新，用 Advantage 函数作为基线：
#
#   A_t = G_t - V(s_t; θ_v)    ← 优势：实际 return 比预期好多少
#
#   G_t = r_t + γ·r_{t+1} + ... + γ^{T-t}·r_T   （从 t 到 episode 结束）
#
# 两个网络：
#   Actor  π(a|s; θ_π)：输出动作概率分布
#   Critic V(s; θ_v)  ：输出状态价值，作为 baseline
#
# 损失函数：
#   Actor  loss : L_π = -mean( A_t · log π(a_t|s_t) )
#   Critic loss : L_v = mean( (G_t - V(s_t))² )
#   Entropy bonus（可选）: H = -mean( π · log π )，鼓励探索
#   Total loss  : L = L_π + c_v · L_v - c_H · H
#
# 与单步 Actor-Critic 的区别：
#   单步 A-C : δ_t = r + γV(s') - V(s)，每步更新，方差大
#   A2C      : G_t 用完整/N步 return，批量更新，方差小，更稳定
#
# 算法流程：
#   for ep in range(num_episodes):
#     s = env.reset()
#     收集完整 episode 的轨迹：
#       trajectory = [(s, a, r, log_prob), ...]
#     从后往前计算每步的 return G_t：
#       G = 0
#       for r, done in reversed:
#           G = r + gamma * G * (1 - done)
#     计算优势 A_t = G_t - V(s_t)
#     计算 L_π, L_v, H，合并成 total loss
#     反向传播一次更新两个网络
#
# 数据结构：
#   actor  : LinearActor，输出 log_softmax
#   critic : LinearCritic，输出 scalar V 值
# ================================================================


def a2c(env, num_episodes=1000, lr=1e-3, gamma=0.99, 
        actor_coef=0.5, value_coef=0.5):
    state_dim = env.observation_space.shape[0]
    num_actions = int(env.action_space.n)
    
    actor = LinearActor(state_dim, num_actions)
    critic = LinearCritic(state_dim)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        
        # 收集完整 episode 的轨迹
        done = False
        trajectory = []
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            log_probs = actor(state_t)  # (1, num_actions)
            action = torch.multinomial(log_probs.exp(), 1).item()
            log_action_p = log_probs[0, action]  # π(a|s, θ_π)
            
            # s', r, done = env.step(a)
            next_state, reward, terminated, truncated, _ = env.step(action)  # next_state : (1, 4)
            done = terminated or truncated
            
            trajectory.append((state_t, action, reward, log_action_p, done))
            state = next_state
        
        G = 0
        actor_loss = 0
        critic_loss = 0
        
        for state_t, action, reward, log_action_p, done in reversed(trajectory):
            # 从后往前计算每步的 return G_t
            G = reward + gamma * G * (1 - done)
            # 计算优势 A_t = G_t - V(s_t)
            V = critic(state_t)
            A = G - V.detach()
            
            actor_loss += -(A * log_action_p)
            critic_loss += (G - V) ** 2
        
        actor_loss /= len(trajectory)
        critic_loss /= len(trajectory)
        
        total_loss = actor_coef * actor_loss + value_coef * critic_loss
        
        # 反向传播一次更新两个网络
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return actor
        
        


if __name__ == "__main__":
    actor = a2c(env, num_episodes=1000)
