"""DQN agent (PyTorch) for the DrivingEnv."""
from __future__ import annotations
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.98,
                 buffer_size: int = 20000, batch_size: int = 64,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: float = 0.995, target_sync: int = 200,
                 device: str = "cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_sync = target_sync
        self.device = torch.device(device)

        self.qnet = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer: Deque[Tuple] = deque(maxlen=buffer_size)
        self.updates = 0

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.qnet(x)
            return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, float(d)))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.from_numpy(np.array(s)).float().to(self.device)
        s2 = torch.from_numpy(np.array(s2)).float().to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.qnet(s).gather(1, a)
        with torch.no_grad():
            q2 = self.target(s2).max(dim=1, keepdim=True).values
            target = r + self.gamma * q2 * (1 - d)
        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        return float(loss.item())

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path: str):
        torch.save(self.qnet.state_dict(), path)

    def load(self, path: str):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.qnet.state_dict())
