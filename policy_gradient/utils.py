import numpy as np
import torch
import torch.nn as nn

import cv2
from collections import deque


class LinearActor(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
    
    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=-1)
    

class LinearCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
    def forward(self, x):
        return self.net(x)    
        
        

class CNNActor(nn.Module):
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
        return torch.log_softmax(self.fc(x), dim=-1)
    

class CNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Linear(512, 1),
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