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


class DQN_Agent:
    def __init__(self, case):
        self.state_size = case.state_size
        self.action_size = case.action_size
        self.hidden_size = hidden_size
        self.network = DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.target_network = DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.GAMMA = 0
        self.t_step = 0
        self.loss_step = 0
        self.Losses = []

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.sample()
                self.learn(experiences)

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

    def sample(self):
        experiences = random.sample(self.memory, k=BATCH_SIZE)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        q_values = self.network(states)  # (batch_size, action_size, N)
        next_q_actions = self.network(next_states)  # (batch_size, action_size, N)
        next_q_actions = torch.argmax(next_q_actions, dim=1)
        next_q_values = self.target_network(next_states)[torch.arange(BATCH_SIZE), next_q_actions].unsqueeze(1)
        target_q_values = rewards + (self.GAMMA * next_q_values * (1 - dones))  # (batch_size, N)
        q_values = q_values.gather(1, actions.squeeze(1).unsqueeze(1))
        loss = F.mse_loss(q_values, target_q_values, reduction='none')  # (batch_size, N, N)
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
