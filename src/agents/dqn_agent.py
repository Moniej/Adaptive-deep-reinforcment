from collections import deque
import random

import numpy as np
import torch
from torch import nn


class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 50000,
        target_update_freq: int = 200,
        hidden_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)
        self.learn_steps = 0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
