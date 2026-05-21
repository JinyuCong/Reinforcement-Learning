import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from collections import deque
import random
import cv2

# ----------------------------------------------------------------
# 线性函数近似用：多项式特征
# ----------------------------------------------------------------
# φ(s) = [1, x̄, ȳ, x̄², ȳ², x̄ȳ]，长度 6
# 其中 x̄ = x / (W-1)，ȳ = y / (H-1)（归一化到 [0,1]）

NUM_FEATURES = 6

def state_to_index(state, env_size):
    x, y = state
    return x + y * env_size[0]

def index_to_state(index, env_size):
    x = index % env_size[0]
    y = index // env_size[0]
    return (x, y)

def get_features(state, env_size):
    """
    将状态 (x, y) 转为多项式特征向量 φ(s)，shape (NUM_FEATURES,)
    """
    x, y = state
    x_n = x / (env_size[0] - 1)
    y_n = y / (env_size[1] - 1)
    return np.array([1.0, x_n, y_n, x_n**2, y_n**2, x_n * y_n])


def epsilon_greedy_fa(epsilon, W, phi, num_actions):
    """
    基于权重矩阵 W 的 ε-greedy 策略。
    Q(s, a) = φ(s)^T W[:, a]
    返回动作概率分布，shape (num_actions,)
    """
    q_values = phi @ W          # shape (num_actions,)
    probs = np.ones(num_actions) * epsilon / num_actions
    probs[np.argmax(q_values)] += 1 - epsilon
    return probs


# ----------------------------------------------------------------
# DQN 用：状态 → Tensor、神经网络、经验回放
# ----------------------------------------------------------------

def state_to_tensor(state, env_size):
    """
    将状态 (x, y) 归一化后转为 torch.FloatTensor，shape (1, 2)
    用于 DQN 网络的输入
    """
    x, y = state
    x_n = x / (env_size[0] - 1)
    y_n = y / (env_size[1] - 1)
    return torch.FloatTensor([[x_n, y_n]])


class QNetwork(nn.Module):
    """
    两层全连接网络：
      输入层: state_dim（默认 2，即归一化的 x, y）
      隐藏层: hidden_dim → hidden_dim（ReLU 激活）
      输出层: num_actions（每个动作的 Q 值）
    """
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.net(x)
    

class ReplayBuffer:
    """
    经验回放缓冲区，存储 (s, a, r, s', done) 元组。
    capacity: 最大容量（先进先出）
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样 batch_size 条经验，返回 numpy arrays"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
    

# ----------------------------------------------------------------
# 用于训练DQN玩打砖块的游戏的CNN网络
# ----------------------------------------------------------------
    
class DQN_CNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
    
    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)
    

def preprocess_frame(state):
    state = np.asarray(state, dtype=np.uint8)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # (210, 160, 3) -> (210, 160)
    resized = cv2.resize(gray, (84, 84))  # (210, 160) -> (84, 84)
    return resized.astype(np.float32) / 255.0  # (84, 84) 归一化


class FrameStack:
    def __init__(self, n=4):
        self.n = n
        self.frames = deque(maxlen=n)
    
    def reset(self, state):
        frame = preprocess_frame(state)
        for _ in range(self.n):
            self.frames.append(frame)
        return self._get_state()
    
    def step(self, state):
        self.frames.append(preprocess_frame(state))
        return self._get_state()
    
    def _get_state(self):
        return np.stack(self.frames, axis=0)  # -> (4, 84, 84)
    

if __name__ == "__main__":
    net = DQN_CNN(num_actions=4)
    stk = FrameStack()
    
    frame = np.random.randint(0, 255, (210, 160, 3))
    initial_states = torch.FloatTensor(stk.reset(frame)).unsqueeze(0)
    print(initial_states.shape)
    
    
    out = net(initial_states)
    print(out.shape)
    
    