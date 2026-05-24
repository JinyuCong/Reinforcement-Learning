import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from src.grid_world import GridWorld

env = GridWorld()

# ================================================================
# REINFORCE —— Monte Carlo Policy Gradient
# ================================================================
# 直接对策略参数 θ 做梯度上升，最大化期望 return：
#   J(θ) = E_π[G_t]
#
# 策略网络 π(a|s; θ)：输入状态，输出每个动作的概率分布（softmax）
#
# 策略梯度定理：
#   ∇J(θ) = E_π[ G_t · ∇log π(a_t|s_t; θ) ]
#
# 算法流程：
#   for ep in range(num_episodes):
#     用当前策略 π(θ) 跑一个完整 episode，收集轨迹
#       trajectory = [(s0, a0, r1), (s1, a1, r2), ..., (sT, aT, rT+1)]
#     从后往前计算每步的 return：
#       G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
#     对每步计算损失并梯度上升：
#       loss = -sum( G_t · log π(a_t|s_t; θ) )
#       （负号是因为 PyTorch 做梯度下降，我们要最大化 J）
#       optimizer.zero_grad(); loss.backward(); optimizer.step()
#
# 与 DQN 的核心区别：
#   - DQN 学的是 Q(s,a)（值函数），间接得到策略
#   - REINFORCE 直接学策略 π(a|s)，不需要 Q 函数
#   - REINFORCE 用完整 episode 的 return，方差较大
#   - DQN 用 TD target，方差小但有 bias
#
# 数据结构：
#   policy_net : 策略网络，输出 softmax 概率
#   optimizer  : Adam
# ================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        # 你来写
        return torch.softmax(self.net(x), dim=-1)


def state_to_tensor(state, env_size):
    # 你来写
    x, y = state
    return torch.FloatTensor([x / (env_size[0] - 1), y / (env_size[1] - 1)])


def reinforce(env, num_episodes=2000, lr=1e-3, gamma=0.99):
    # 你来写
    num_actions = len(env.action_space)
    state_dim = 2
    
    # 初始化net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = PolicyNetwork(state_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    for ep in range(num_episodes):
        state, _ = env.reset()
        
        done = False
        trajectory = []
        
        # 用当前策略 π(θ) 跑一个完整 episode，收集轨迹
        while not done:
            state_t = state_to_tensor(state, env.env_size).to(device)
            probs = policy_net(state_t)
            action_idx = torch.multinomial(probs, 1).item()
            
            # 将action对应的概率挑出来
            action_p = probs[action_idx]
            
            action = env.action_space[action_idx]
            next_state, reward, done, _ = env.step(action)
            
            trajectory.append((state_t, action_idx, action_p, reward))
            
            state = next_state
           
        # 从后往前计算每步的 return
        G = 0
        returns = []
        for _, _, action_p, reward in reversed(trajectory):
            G = reward + gamma * G
            
            # loss = -sum( G_t · log π(a_t|s_t; θ) )
            returns.insert(0, G)
        
        returns_t = torch.FloatTensor(returns).to(device)
        # 归一化稳定训练
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        log_probs = torch.stack([torch.log(action_p) for _, _, action_p, _ in trajectory])
        loss = -(returns_t @ log_probs)
        print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return policy_net


if __name__ == "__main__":
    policy_net = reinforce(env, num_episodes=2000, lr=1e-2)
    policy_net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        state_t = state_to_tensor(state, env.env_size).to(device)
        with torch.no_grad():
            action_idx = policy_net(state_t).argmax().item()
        state, reward, done, _ = env.step(env.action_space[action_idx])
        env.render()
    
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy_matrix = np.zeros((num_states, num_actions))
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        s_t = state_to_tensor((x, y), env.env_size).to(device)
        with torch.no_grad():
            best_a = policy_net(s_t).argmax().item()
        
        policy_matrix[s, best_a] = 1.0
        
    env.add_policy(policy_matrix)
    env.render(animation_interval=30)
