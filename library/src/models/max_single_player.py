import math
import random
# import gymnasium
import matplotlib
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from farkle.logic.gameobjects import DiceHand

# Relies heavily on code and examples in:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""
The agent is presented with a dice hand with multiple possible scoring options. 
Each scoring option has an associated cumulative score and number of remaining dice. 
Each scoring option also has an expected score if that option were chosen, the 
dice rolled again and the optimal policy followed in choosing the remaining options. 
For each scoring option the agent tries to estimate the expected score conditional
on following the optimal policy. If at least 1 of the options has E[score] > 0 the 
agent should choose the option with the highest E[score] and roll again. If all 
options are E[score] < 0 the agent should choose the one that has the highest 
cumulative score and choose to not roll again. 

State: 
    Agent observes: 
    1. Cumulative score of the possible scoring option. E.g. if agent has 100 points
       and a scoring option is to choose a triplet of 5's, the cumulative score
       is 600 points. 
    2. The number of remaining dice. 
    
Action: 
    
"""

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    # pass
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions=1):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 2  # re-roll or don't
# Get the number of state observations

# state passed directly into nn
# state = torch.tensor([score, remaining_dice])
# although state is used like a iter of states in select_action function
# info? isn't used in tutorial
state, info = env.reset()
# n_observations = len(state)
n_observations = 2  # score, remaining dice count

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state: torch.tensor) -> torch.tensor:
    """should perform choice of dicehand like select_training_action
    except without the eps_threshold stuff"""
    # expected score if dice hand of each state is rolled
    E_scores = [policy_net(s) for s in state]
    # find max score and its index in E_scores
    max_ind = 0
    max_e_score = E_scores[0]
    for i, s in enumerate(E_scores[1:]):
        if s > max_e_score:
            max_ind = i + 1
            max_e_score = s

    next_action: torch.tensor = None
    if max_e_score > 0:
        # return that scoring option and roll again indicator
        pass
    else:
        # return max[s.score for s in state] and don't roll again
        pass

def select_training_action(state) -> torch.tensor:
    """chooses to use nn or random selection for next action"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:  # use nn
        with torch.no_grad():
            next_action: torch.tensor = select_action(state)
            return next_action
    else:
        return torch.tensor([random.choice(range(n_actions))], device=device, dtype=torch.long)


episode_scores = []


def plot_scores(show_result=False):
    """plots score of dice turns in training"""
    plt.figure(1)
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_e_score_values: torch.tensor = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
