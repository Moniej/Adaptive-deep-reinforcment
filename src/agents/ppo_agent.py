import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        hidden_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reset_memory()

    def reset_memory(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            distribution = Categorical(logits=logits)
            action = torch.argmax(distribution.probs, dim=1) if greedy else distribution.sample()
            log_prob = distribution.log_prob(action)

        self._last_state = state
        self._last_action = int(action.item())
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return self._last_action

    def store_transition(self, reward: float, done: bool) -> None:
        self.states.append(self._last_state)
        self.actions.append(self._last_action)
        self.log_probs.append(self._last_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(self._last_value)

    def _compute_advantages(self, next_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        values = self.values + [next_value]

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)

        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = advantages_tensor + torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        return advantages_tensor, returns_tensor

    def learn(self) -> float:
        if not self.states:
            return 0.0

        advantages, returns = self._compute_advantages()
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)

        last_loss = 0.0
        for _ in range(self.update_epochs):
            logits, values = self.model(states)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            last_loss = float(loss.item())

        self.reset_memory()
        return last_loss

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
