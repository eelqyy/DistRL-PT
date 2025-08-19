import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from env_case import Case27

BATCH_SIZE = 32
TAU = 0.05
LR = 1e-4
EPSILON = 0.1
MEMORY_SIZE = 10000
UPDATE_EVERY = 8
hidden_size = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class DQN_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN_Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # (batch_size, hidden_size)
        x = F.relu(self.fc2(x))  # (batch_size, hidden_size)
        x = self.fc3(x)  # (batch_size, action_size * N)
        x = x.view(-1, self.action_size)  # (batch_size, action_size, N)
        return x


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority


class DQN_Agent:
    def __init__(self, case):
        self.state_size = case.state_size
        self.action_size = case.action_size
        self.hidden_size = hidden_size
        self.network = DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.target_network = DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_SIZE)
        self.GAMMA = 0
        self.alpha = 1
        self.beta = 1
        self.t_step = 0
        self.loss_step = 0
        self.Losses = []

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.replay_buffer.buffer) > BATCH_SIZE:
                experiences, indices, weights = self.replay_buffer.sample(BATCH_SIZE, self.alpha, self.beta)
                self.learn(experiences, indices, weights)

    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)

        self.network.train()
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().numpy())

    def learn(self, experiences, indices, weights):
        states, actions, rewards, next_states, dones = experiences
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)
        weights = torch.tensor(weights, dtype=torch.float32)
        q_values = self.network(states)  # (batch_size, action_size, N)
        next_q_values = self.target_network(next_states)  # (batch_size, action_size, N)
        next_q_values = torch.max(next_q_values, dim=1).values.unsqueeze(1)
        target_q_values = rewards + (self.GAMMA * next_q_values * (1 - dones))  # (batch_size, N)
        q_values = q_values.gather(1, actions.squeeze(1).unsqueeze(1))
        loss = F.mse_loss(q_values, target_q_values, reduction='none').mean(dim=-1)  # (batch_size, N, N)
        priorities = loss.detach().numpy()
        self.replay_buffer.update_priorities(indices, priorities)
        loss = weights * loss
        loss = loss.mean()  # scalar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        self.loss_step += 1
        if self.loss_step % (100 * Case27().T / UPDATE_EVERY) == 0:
            self.Losses.append(loss)
        self.soft_update(self.network, self.target_network)

    @staticmethod
    def soft_update(local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def lr_decay(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.98
