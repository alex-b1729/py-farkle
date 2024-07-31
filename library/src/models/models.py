import random
from collections import deque

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_in_features: int = 2, n_out_features: int = 1):
        """
        2 inputs: current score and number of dice
        remaining if choose a particular possible score
        1 output: expected future score under policy
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_in_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_out_features)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        s = 'memory(\n'
        for i in list(self.memory):
            s += f'\t{i}\n'
        s += ')'
        return s
