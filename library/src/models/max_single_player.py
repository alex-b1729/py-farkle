import math
import random
import matplotlib
from itertools import count
import matplotlib.pyplot as plt
from dataclasses import dataclass
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
        for i in list(self.memory)[:10]:
            s += f'\t{i}\n'
        s += ')'
        return s


class DQN(nn.Module):
    def __init__(self, n_in_features: int = 2, n_out_features: int = 1):
        """
        2 inputs: current score and number of dice
        remaining if choose a particular possible score
        1 output: expected future ??? score under policy
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


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 1  # not discounting since there's a definite, eventual end to every turn
EPS_PS_START = 0.9  # whether to choose the estimated max possible score
EPS_PS_END = 0.05
EPS_ROLL_START = 0.9  # whether to roll again regardless if recommended by est max ps
EPS_ROLL_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# todo: init state and other stuff here
# state passed directly into nn
# state = torch.tensor([score, remaining_dice])
# although state is used like a iter of states in select_action function
# What's info? isn't used in tutorial
# state, info = env.reset()
# n_observations = len(state)
n_observations = 2  # score, remaining dice count

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())


def dice_remaining_convert(n: int) -> int:
    """convert 0 dice remaining to 6"""
    return int((n - 1) % 6 + 1)


def evaluate_expected_scores(states: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        return policy_net(states)


def select_possible_score(ps_list: list, E_scores: torch.tensor) -> tuple[DiceHand, float]:
    max_score_idx = E_scores.argmax()
    chosen_ps: DiceHand = ps_list[max_score_idx]
    E_score: float = E_scores[max_score_idx]
    return chosen_ps, E_score


def select_training_possible_score(dh: DiceHand) -> tuple[DiceHand, float]:
    """
    uses EPS_PS to determine whether to choose ps with max expected
    future score or choose a random ps
    returns: chosen possible score dicehand and predicted score
    """
    global TURNS_COMPLETE
    sample = random.random()
    eps_threshold = EPS_PS_END + (EPS_PS_START - EPS_PS_END) * \
                    math.exp(-1. * TURNS_COMPLETE / EPS_DECAY)
    # turns_done += 1

    ps_list = dh.possible_scores()
    print(ps_list)
    states = torch.tensor([[int(dh.score + ps.score),
                            dice_remaining_convert(len(dh.dice_values_free()) - ps.num_dice)]
                           for ps in ps_list],
                          device=device, dtype=torch.float32)
    print(states)
    E_scores = evaluate_expected_scores(states)

    if sample > eps_threshold:  # use max possible score
        return select_possible_score(ps_list, E_scores)
    else:
        random_idx = random.choice(range(len(ps_list)))
        return ps_list[random_idx], E_scores[random_idx]


def decide_will_roll_again(E_score: float) -> bool:
    """roll again if positive expected future score"""
    return E_score > 0


def decide_training_will_roll_again(E_score: float) -> bool:
    global TURNS_COMPLETE
    sample = random.random()
    eps_threshold = EPS_PS_END + (EPS_PS_START - EPS_PS_END) * \
                    math.exp(-1. * TURNS_COMPLETE / EPS_DECAY)
    if sample > eps_threshold:  # use roll if E_score > 0
        return decide_will_roll_again(E_score)
    else:
        return random.choice([True, False])


def select_action(dh: DiceHand) -> tuple[DiceHand, float]:
    """chooses possible score and whether to roll again"""
    global TURNS_COMPLETE  # idk where this needs to sit in the logic
    chosen_ps, E_score = select_training_possible_score(dh)
    will_roll_again = decide_training_will_roll_again(E_score)
    TURNS_COMPLETE += 1
    return chosen_ps, will_roll_again


TrainingState = namedtuple('TrainingState',
                           ('score', 'num_dice_remaining'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


@dataclass(frozen=True)
class FarkleAction:
    possible_score: DiceHand
    will_roll_again: bool


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100)
NUM_TRAINING_TURNS = 1
TURNS_COMPLETE = 0

for turn_idx in range(NUM_TRAINING_TURNS):
    dh = DiceHand()  # init as rolled dh
    print(dh)
    will_roll_again = True
    score_post = dh.score
    state_post = TrainingState(score=score_post, num_dice_remaining=len(dh.free_dice))
    while not dh.farkled and will_roll_again:
        # capture initial dh state
        # state_pre = TrainingState(score=dh.score,
        #                           num_dice_remaining=len(dh.free_dice))
        state_pre = state_post
        print(f'state pre: {state_pre}')

        # choose action if not farkled
        # but b/c we're in this loop we've already not farkled
        # policy_net makes decision
        chosen_ps, will_roll_again = select_action(dh)
        print('Chosen ps:', chosen_ps)
        print('Will roll again:', will_roll_again)
        # execute decision
        dh.lock_from_dicehand(chosen_ps)
        # save immediate reward of decision
        reward = chosen_ps.score
        print('reward:', reward)
        score_post = dh.score
        num_dice_remaining_post = len(dh.free_dice)
        state_post = TrainingState(score=score_post,
                                   num_dice_remaining=num_dice_remaining_post)
        print('state post', state_post)

        # save turn transition
        # action is whether to roll or not
        memory.push(
            Transition(
                state=state_pre,
                action=will_roll_again,
                next_state=state_post,
                reward=reward
            )
        )

        if will_roll_again:
            dh.roll()
            print(f'roll again result: {dh}')

    # finally if ended turn by farkle
    # `and will_roll_again` necessary since locking dice then choosing to not
    # roll again can cause dh.farkled == True
    if dh.farkled and will_roll_again:
        # capture final transition to farkle state
        state_pre = state_post
        state_post = None
        reward = -1 * score_post
        memory.push(
            Transition(
                state=state_pre,
                action=True,  # bc if farkled there was the intent to roll again
                next_state=state_post,
                reward=reward
            )
        )

print(memory)

episode_scores = []


def plot_scores(show_result=False):
    """plots score of dice turns in training"""
    plt.figure(1)
    durations_t = torch.tensor(episode_scores, dtype=torch.float32)
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


# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                        if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     # state_action_values = policy_net(state_batch).gather(1, action_batch)
#     state_e_score_values: torch.tensor = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1).values
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     with torch.no_grad():
#         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()
