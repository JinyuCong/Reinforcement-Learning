import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ale_py
import gymnasium as gym
import cv2
from collections import deque

from policy_gradient.utils import CNNActor, CNNCritic, FrameStack

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    
class UpdateDataset(Dataset):
    def __init__(self, collection, returns, advantages):
        super().__init__()
        self.collection = collection
        self.returns = returns
        self.advantages = advantages
        
    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        return {
            'state': self.collection[index][0],
            'action': self.collection[index][1],
            'reward': self.collection[index][2],
            'done': self.collection[index][3],
            'log_action_p': self.collection[index][4],
            'state_value': self.collection[index][5],
            'G': self.returns[index],
            'A': self.advantages[index]
        }


def ppo(env, num_updates=500, steps_per_update=256,
        lr=3e-4, gamma=0.99, clip_eps=0.2,
        ppo_epochs=4, batch_size=64, value_coef=0.5):
    
    num_actions = int(env.action_space.n)
    
    actor_net = CNNActor(num_actions).to(device)
    critic_net = CNNCritic().to(device)
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=lr)
    
    frame_stack = FrameStack(n=4)
    
    state, info = env.reset()
    prev_lives = info['lives']
    state_frames = frame_stack.reset(state)
    
    for update in range(num_updates):
        collection = []  # 用来储存 (s, a, r, done, log_prob_old, V(s))
        epsilon = max(0.05, 1.0 - update / (num_updates * 0.5))
        for step in range(steps_per_update):
            state_t = torch.FloatTensor(state_frames).unsqueeze(0).to(device)  # (1, 4, 84, 84)
            log_probs = actor_net(state_t)  # (1, num_actions)
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.multinomial(log_probs.exp(), 1).item()
                
            log_action_p = log_probs[0, action].detach()
    
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info.get('lives', 0) < prev_lives:
                reward -= 1
            prev_lives = info['lives']
            
            next_state_frames = frame_stack.step(next_state)
            # next_state_t = torch.FloatTensor(next_state_frames).unsqueeze(0)
            
            state_value = critic_net(state_t)  # V(s)
            
            # 存储 (s, a, r, done, log_prob_old, V(s))
            collection.append((state_t.squeeze(0), action, reward, done, log_action_p, state_value.squeeze(0)))
            
            state_frames = next_state_frames
            # 当到达终点后重置env
            if done:
                state, info = env.reset()
                state_frames = frame_stack.reset(state)
        
        # 计算每步的优势 A_t 和 return G_t
        G = 0
        returns = []
        for _, _, reward, done, _, _ in reversed(collection):
            G = reward + gamma * G * (1 - done)
            returns.insert(0, G)
        
        advantages = []
        for (_, _, _, _, _, state_value), G in zip(collection, returns):
            A = G - state_value.detach().squeeze().item()
            advantages.append(A)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.FloatTensor(returns)
            
        dataset = UpdateDataset(collection, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        update_loss = 0
        for epoch in range(ppo_epochs):
            for batch in dataloader:
                state = batch['state'].to(device)  # (batch, frames, H, W)
                action = batch['action'].reshape(-1, 1).to(device)  # (batch, 1)
                done = batch['done'].reshape(-1, 1).to(device)  # (batch, 1)
                log_pi_old = batch['log_action_p'].reshape(-1, 1).to(device)  # (batch, 1)
                V_old = batch['state_value'].to(device)  # (batch, 1)
                G = batch['G'].reshape(-1, 1).to(device)  # (batch, 1)
                A = batch['A'].reshape(-1, 1).to(device)  # (batch, 1)
                
                log_probs: torch.FloatTensor = actor_net(state)
                log_pi_new = log_probs.gather(1, action)
                # r_t = exp(log_π_new(a|s) - log_π_old(a|s))
                ratio = torch.exp(log_pi_new - log_pi_old)  # (batch, 1)
                L_clip = torch.min(ratio * A, torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * A).mean()
                
                V_new = critic_net(state)
                L_value = ((V_new - G) ** 2).mean()
                
                loss = -L_clip + value_coef * L_value
                
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
                
                update_loss += loss.item()
        
        update_loss /= (ppo_epochs * len(dataloader))
        print(f"Update {update+1} | Mean loss : {update_loss}")
                
    return actor_net
            

if __name__ == "__main__":
    actor = ppo(env, num_updates=500, steps_per_update=1024)