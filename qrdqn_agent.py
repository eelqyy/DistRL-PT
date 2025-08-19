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
N = 8
K = 32
EPSILON = 0.1
MEMORY_SIZE = 30000
UPDATE_EVERY = 8
hidden_size = 512
w0 = 0.125 * np.ones(N)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class QR_DQN_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QR_DQN_Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_q = N
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size * self.n_q)

    def forward(self, state):
        n_q = self.n_q
        x = F.relu(self.fc1(state))  # (batch_size, hidden_size)
        x = F.relu(self.fc2(x))  # (batch_size, hidden_size)
        x = F.relu(self.fc2(x))  # (batch_size, hidden_size)
        x = F.relu(self.fc2(x))  # (batch_size, hidden_size)
        x = self.fc3(x)  # (batch_size, action_size * N)
        x = x.view(-1, self.action_size, n_q)  # (batch_size, action_size, N)
        return x


class QR_DQN_Agent:
    def __init__(self, case):
        self.state_size = case.state_size
        self.action_size = case.action_size
        self.hidden_size = hidden_size
        self.network = QR_DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.target_network = QR_DQN_Network(self.state_size, self.action_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.memory_done = []
        self.t_step = 0
        self.loss_step = 0
        self.Losses = []
        self.GAMMA = 0
        self.w = w0
        self.N = N

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
            action_values_w = torch.from_numpy(np.average(action_values.cpu().numpy(), axis=2, weights=self.w))
        self.network.train()
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values_w.cpu().numpy())

    def sample(self):
        experiences = random.sample(self.memory, k=BATCH_SIZE)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def percentiles_done(self):
        done_experiences = self.memory_done
        percentiles_action_reward = []
        for i in range(self.action_size):
            done_experiences_action = [exp for exp in done_experiences if exp[1] == i]
            if done_experiences_action == []:
                done_experiences_reward = [0]
            else:
                done_experiences_reward = [exp[2] for exp in done_experiences_action]
            percentiles_reward = np.percentile(done_experiences_reward, np.arange(50/N, 100, 100/N))  # (N)
            percentiles_action_reward.append(percentiles_reward)  # (action_size, N)
        return percentiles_action_reward

    def learn(self, experiences):
        tau = torch.arange(0.5/N, 1, 1/N).repeat(BATCH_SIZE, 1).to(device)
        states, actions, rewards, next_states, dones = experiences
        q_values = self.network(states)  # (batch_size, action_size, N)
        next_q_values = self.target_network(next_states)  # (batch_size, action_size, N)
        next_q_values = next_q_values[torch.arange(BATCH_SIZE), torch.argmax(torch.from_numpy(np.average(next_q_values.detach().numpy(), axis=2, weights=self.w)), dim=1)]  # (batch_size, N)
        target_q_values = rewards + self.GAMMA * next_q_values * (1 - dones)  # (batch_size, N)
        q_values = q_values.gather(1, actions.unsqueeze(-1).repeat(1, 1, N))  # (batch_size, 1, N)
        q_values = q_values.squeeze(1)  # (batch_size, N)
        target_q_values = target_q_values.detach()  # (batch_size, N)
        td_error = target_q_values.unsqueeze(1) - q_values.unsqueeze(2)  # (batch_size, N, N)
        quantile_weight = torch.abs(tau.unsqueeze(-1) - (td_error.detach() < 0).float())  # (batch_size, N, N)
        huber_loss = F.l1_loss(q_values.unsqueeze(2), target_q_values.unsqueeze(1),
                                  reduction='none')  # (batch_size, N, N)
        loss = (quantile_weight * huber_loss).sum(dim=-1).mean(dim=-1)  # (batch_size, N)
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
        i = 0
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            i += 1
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def lr_decay(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.98
