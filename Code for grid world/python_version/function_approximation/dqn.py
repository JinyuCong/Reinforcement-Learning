import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from src.grid_world import GridWorld
from function_approximation.utils import QNetwork, ReplayBuffer, state_to_tensor

env = GridWorld()

# ================================================================
# DQN —— Deep Q-Network（Deep Q-learning with Experience Replay）
# ================================================================
# Q(s, a; θ) 用神经网络近似：输入归一化状态 [x̄, ȳ]，输出所有动作的 Q 值
#
# 两大关键机制：
#   1. 经验回放（Experience Replay）
#      - 将每步 (s, a, r, s', done) 存入 ReplayBuffer
#      - 训练时随机采样 mini-batch，打破时序相关性，稳定训练
#
#   2. 目标网络（Target Network）
#      - 维护两个网络：在线网络 q_net(θ) 和目标网络 target_net(θ⁻)
#      - TD target 由 target_net 计算，避免"追逐移动目标"
#      - 每隔 target_update_freq 步将 θ 复制到 θ⁻（硬更新）
#
# 损失函数（MSE）：
#   y = r + γ * max_a' Q(s', a'; θ⁻)    （未终止）
#   y = r                                 （终止状态）
#   L(θ) = (y - Q(s, a; θ))²
#
# 算法思路：
#   for ep in range(num_episodes):
#     s = env.reset()
#     while not done（最多 200 步）:
#       用 ε-greedy 选 a（基于 q_net）
#       执行 a，得到 r, s', done
#       buffer.push(s, a, r, s', done)
#       if len(buffer) >= batch_size:
#         从 buffer 随机采样一个 mini-batch
#         用 target_net 计算 TD target y（注意 done 掩码）
#         计算 loss = MSE(q_net(s)[a], y)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#       每隔 target_update_freq 步：target_net.load_state_dict(q_net.state_dict())
#       s ← s'
#       step += 1
#
# 数据结构：
#   q_net      : 在线网络，持续被梯度更新
#   target_net : 目标网络，定期从 q_net 复制，计算 TD target 时不参与梯度
#   buffer     : ReplayBuffer，存储经验
#   optimizer  : Adam
# ================================================================

def dqn(env, num_episodes=1000, lr=1e-3, gamma=0.9,
        epsilon_start=1.0, epsilon_end=0.05,
        batch_size=64, buffer_capacity=10000,
        target_update_freq=100):

    num_actions = len(env.action_space)
    state_dim = 2  # 归一化的 (x, y)

    # 在线网络 & 目标网络，初始参数相同
    q_net      = QNetwork(state_dim, num_actions, hidden_dim=64)
    target_net = QNetwork(state_dim, num_actions, hidden_dim=64)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()   # 目标网络不参与梯度计算

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    buffer    = ReplayBuffer(capacity=buffer_capacity)

    step_count = 0  # 全局步数计数器，用于触发 target_net 更新

    # 你来写
    # 提示：
    #   state_to_tensor(state, env.env_size)
    #       → torch.FloatTensor，shape (1, 2)，用于喂给网络
    #
    #   ε-greedy 动作选择：
    #       if np.random.rand() < epsilon:
    #           action_idx = np.random.randint(num_actions)
    #       else:
    #           with torch.no_grad():
    #               action_idx = q_net(state_tensor).argmax().item()
    #
    #   采样 mini-batch 并计算 loss：
    #       states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)
    #       # 转 tensor（states_b 已经是 shape (batch, 2) 的 float32）
    #       states_t     = torch.FloatTensor(states_b)
    #       next_states_t= torch.FloatTensor(next_states_b)
    #       rewards_t    = torch.FloatTensor(rewards_b)
    #       dones_t      = torch.FloatTensor(dones_b)
    #       actions_t    = torch.LongTensor(actions_b)
    #       # 当前 Q 值：q_net(states_t) 形状 (batch, num_actions)，取实际执行的动作
    #       q_pred = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    #       # TD target（target_net，不参与梯度）
    #       with torch.no_grad():
    #           q_next = target_net(next_states_t).max(1).values
    #           y = rewards_t + gamma * q_next * (1 - dones_t)
    #       loss = loss_fn(q_pred, y)
    #
    #   buffer.push 时，state 应存归一化数组，可用：
    #       np.array([x/(W-1), y/(H-1)])

    return q_net


if __name__ == "__main__":
    q_net = dqn(env, num_episodes=1000)
    q_net.eval()

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        s_t = state_to_tensor(state, env.env_size)
        with torch.no_grad():
            action_idx = q_net(s_t).argmax().item()
        state, reward, done, _ = env.step(env.action_space[action_idx])
        env.render()

    # 从 q_net 重建 policy_matrix 以便可视化
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy_matrix = np.zeros((num_states, num_actions))
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        s_t = state_to_tensor((x, y), env.env_size)
        with torch.no_grad():
            best_a = q_net(s_t).argmax().item()
        policy_matrix[s, best_a] = 1.0
    env.add_policy(policy_matrix)
    env.render(animation_interval=30)
