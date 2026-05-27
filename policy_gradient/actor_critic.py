import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

import ale_py
import gymnasium as gym
import cv2
from collections import deque

from policy_gradient.utils import LinearActor, LinearCritic

gym.register_envs(ale_py)
env = gym.make("CartPole-v1", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# Actor-Critic（策略梯度版本单步 TD）
# ================================================================
# REINFORCE 用完整 episode 的 return G_t，方差很大。
# Actor-Critic 用 TD error（优势估计）替代 G_t，降低方差：
#
#   δ_t = r + γ·V(s'; θ_v) - V(s; θ_v)    ← TD error，即优势 A(s,a) 的估计
#
# 两个网络：
#   Actor  π(a|s; θ_π)：策略网络，决定做什么动作
#   Critic V(s; θ_v)  ：值函数网络，评估当前状态有多好
#
# 更新规则（每步更新，不需要等完整 episode）：
#   Critic loss : L_v = δ_t²                           （让 V 更准）
#   Actor  loss : L_π = -δ_t · log π(a_t|s_t; θ_π)   （用 δ 加权策略梯度）
#
# 与 REINFORCE 的区别：
#   REINFORCE : G_t = r1 + γr2 + ...  （完整 episode，高方差）
#   A-C       : δ_t = r + γV(s') - V(s)（单步 TD，低方差，有 bias）
#
# 算法流程：
#   for ep in range(num_episodes):
#     s = env.reset()
#     while not done:
#       a ~ π(·|s; θ_π)
#       s', r, done = env.step(a)
#       δ = r + γ·V(s') - V(s)   （done 时 V(s')=0）
#       loss_critic = δ²
#       loss_actor  = -δ · log π(a|s)
#       反向传播更新 θ_v 和 θ_π
#       s ← s'
#
# 数据结构：
#   actor  : 策略网络（softmax 输出）
#   critic : 值函数网络（scalar 输出）
# ================================================================


def actor_critic(env, num_episodes=2000, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
    state_dim = env.observation_space.shape[0]
    num_actions = int(env.action_space.n)
    
    actor_net = LinearActor(state_dim, num_actions).to(device)
    critic_net = LinearCritic(state_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=lr_actor)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=lr_critic)
    
    for ep in range(num_episodes):
        state, _ = env.reset()  # state : (4,)
        
        done = False
        ep_step = 0
        ep_actor_loss = 0
        ep_critic_loss = 0

        while not done:
            # a ~ π(·|s; θ_π)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, 4)
            log_probs = actor_net(state_t)  # (1, num_actions)

            action = torch.multinomial(log_probs.exp(), 1).item()
            log_action_p = log_probs[0, action]  # π(a|s, θ_π)
            
            # s', r, done = env.step(a)
            next_state, reward, terminated, truncated, _ = env.step(action)  # next_state : (1, 4)
            done = terminated or truncated
            
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)  # (1, 4)
            
            # δ = r + γ·V(s') - V(s)   （done 时 V(s')=0）
            state_value = critic_net(state_t).squeeze()  # V(s)
            next_state_value = critic_net(next_state_t).squeeze() if not done else torch.tensor(0.0).to(device)  # V(s')
            delta = reward + gamma * next_state_value - state_value
            
            loss_critic = delta ** 2 
            loss_actor = -delta.detach() * log_action_p
            ep_actor_loss += loss_actor
            ep_critic_loss += loss_critic
            
            # 反向传播 θ_π
            actor_optimizer.zero_grad()
            loss_actor.backward()
            # torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
            actor_optimizer.step()
            
            # 反向传播 θ_v
            critic_optimizer.zero_grad()
            loss_critic.backward()
            # torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=1.0)
            critic_optimizer.step()
            
            state = next_state
            
            ep_step += 1
        
        print(f"Episode {ep+1} | Actor loss : {ep_actor_loss.item()/ep_step:.3f} | Critic loss : {ep_critic_loss.item()/ep_step:.3f}")
        
    return actor_net
            
            

if __name__ == "__main__":
    actor = actor_critic(env, num_episodes=2000)
    # actor.eval()
    
    # state, _ = env.reset()
    # env.render()
    # done = False
    # while not done:
    #     state_t = state_to_tensor(state, env.env_size).to(device)
    #     with torch.no_grad():
    #         action_idx = actor(state_t).argmax().item()
    #     state, reward, done, _ = env.step(env.action_space[action_idx])
    #     env.render()
    
    # num_states = env.num_states
    # num_actions = len(env.action_space)
    # policy_matrix = np.zeros((num_states, num_actions))
    # for s in range(num_states):
    #     x = s % env.env_size[0]
    #     y = s // env.env_size[0]
    #     s_t = state_to_tensor((x, y), env.env_size).to(device)
    #     with torch.no_grad():
    #         best_a = actor(s_t).argmax().item()
        
    #     policy_matrix[s, best_a] = 1.0
        
    # env.add_policy(policy_matrix)
    # env.render(animation_interval=30)