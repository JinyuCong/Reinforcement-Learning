import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from collections import deque
import random

env = gym.make("Pendulum-v1", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# DDPG —— Deep Deterministic Policy Gradient
# ================================================================
# 之前的 Actor-Critic / A2C / PPO 都是随机策略：
#   π(a|s) 输出动作的概率分布，然后采样
#
# DDPG 使用确定性策略：
#   μ(s; θ_π) 直接输出一个确定的动作值（连续值）
#   适合连续动作空间，如机器人关节角度、油门大小等
#
# 核心组成（借鉴 DQN 的两个机制）：
#   1. 经验回放（Experience Replay）
#      - 存储 (s, a, r, s', done)，打破时序相关性
#   2. 目标网络（Target Network）
#      - actor_target 和 critic_target 用软更新（Polyak averaging）：
#        θ_target = τ·θ + (1-τ)·θ_target  （τ 很小，如 0.005）
#        比 DQN 的硬更新更平滑
#
# 两个网络：
#   Actor  μ(s; θ_π)  ：输出确定性动作，范围由 tanh 缩放到环境动作范围
#   Critic Q(s,a; θ_v)：输入状态和动作，输出 Q 值（注意同时输入 s 和 a）
#
# 探索方式：
#   确定性策略无法自主探索，需要手动加噪声：
#   a = μ(s) + N(0, σ)   ← 高斯噪声，或 Ornstein-Uhlenbeck 噪声
#
# 更新规则：
#   Critic loss : L = mean( (r + γ·Q_target(s', μ_target(s')) - Q(s,a))² )
#   Actor  loss : L = -mean( Q(s, μ(s)) )   ← 最大化 Q 值，负号变最小化
#
# 算法流程：
#   初始化 actor, critic, actor_target, critic_target, ReplayBuffer
#   for step in range(total_steps):
#     a = μ(s) + noise
#     s', r, done = env.step(a)
#     buffer.push(s, a, r, s', done)
#     if len(buffer) >= batch_size:
#       采样 mini-batch
#       用 actor_target 和 critic_target 计算 TD target：
#         y = r + γ · Q_target(s', μ_target(s'))
#       更新 Critic：minimize (y - Q(s,a))²
#       更新 Actor：maximize Q(s, μ(s))，即 minimize -Q(s, μ(s))
#       软更新目标网络：
#         θ_critic_target = τ·θ_critic + (1-τ)·θ_critic_target
#         θ_actor_target  = τ·θ_actor  + (1-τ)·θ_actor_target
#
# 与随机策略方法的区别：
#   随机 A-C  : π(a|s) 输出分布，用 log_prob 计算梯度
#   DDPG     : μ(s) 直接输出动作，梯度通过 Q 网络反传到 actor
#              即 ∇L_actor = ∇_a Q(s,a) · ∇_θ μ(s;θ)（链式法则）
#
# 适用场景：
#   连续动作空间（CartPole 不适合，Pendulum / MuJoCo / 机器人控制适合）
#
# 数据结构：
#   actor        : 策略网络，输出确定性动作
#   critic       : Q 网络，输入 (s, a)，输出 Q 值
#   actor_target : actor 的目标网络，软更新
#   critic_target: critic 的目标网络，软更新
#   buffer       : ReplayBuffer
# ================================================================


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=256):
        super().__init__()
        # 注意：输出层用 tanh，再乘以 action_bound 缩放到动作范围
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # (B, 1)
        )

    def forward(self, x):
        # 你来写
        return self.action_bound * self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 注意：输入是 cat(state, action)，输出是标量 Q 值
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # (B, 1)
        )

    def forward(self, state, action):
        # 你来写
        # state: (B, 3), action: (B, 1)
        x = torch.concat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        # 你来写
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 你来写
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 你来写
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        # 你来写
        return len(self.buffer)


def soft_update(target, source, tau):
    # θ_target = τ·θ_source + (1-τ)·θ_target
    # 你来写
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def ddpg(env, total_steps=100000, lr_actor=1e-4, lr_critic=1e-3,
         gamma=0.99, tau=0.005, batch_size=64,
         buffer_capacity=100000, noise_std=0.1):
    # 你来写
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]  # 1
    action_bound = env.action_space.high[0]  # 2.0
    
    # 初始化actor, critic
    actor = Actor(state_dim, action_dim, action_bound).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    
    # 初始化actor_target, critic_target
    actor_target = Actor(state_dim, action_dim, action_bound).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_target.eval()
    critic_target.eval()
    
    optim_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    state, _ = env.reset()
    
    for step in range(total_steps):
        state_t = torch.FloatTensor(state).to(device)
        
        # a = μ(s) + noise
        mu = actor(state_t)
        noise = torch.normal(mean=0.0, std=noise_std, size=(action_dim,)).to(device)
        action = mu + noise
        action_np = action.detach().cpu().numpy()
        
        # s', r, done = env.step(a)
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        # buffer.push(s, a, r, s', done)
        replay_buffer.push(state, action_np, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            # 采样 mini-batch
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size=batch_size)
            states_t = torch.FloatTensor(states_b).to(device)
            actions_t = torch.FloatTensor(actions_b).to(device)
            rewards_t = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(next_states_b).to(device)
            dones_t = torch.FloatTensor(dones_b).unsqueeze(1).to(device)
            
            # y = r + γ · Q_target(s', μ_target(s'))
            mu_targets = actor_target(next_states_t)
            q_targets = critic_target(next_states_t, mu_targets)
            td_targets = (rewards_t + gamma * q_targets * (1 - dones_t)).detach()
            
            # 计算 Q(s,a)
            qs = critic(states_t, actions_t)
            # 计算 μ(s)
            mus = actor(states_t)
            
            # 更新 Critic：minimize (y - Q(s,a))²
            loss_critic = ((td_targets - qs) ** 2).mean()
            
            # 更新 Actor：maximize Q(s, μ(s))，即 minimize -Q(s, μ(s))
            loss_actor = -critic(states_t, mus).mean()
            
            optim_actor.zero_grad()
            loss_actor.backward()
            optim_actor.step()
            
            optim_critic.zero_grad()
            loss_critic.backward()
            optim_critic.step()
            
            soft_update(critic_target, critic, tau)
            soft_update(actor_target, actor, tau)
            
        if done:
            state, _ = env.reset()
        else:
            state = next_state
        
    return actor


if __name__ == "__main__":
    actor = ddpg(env, total_steps=100000)
