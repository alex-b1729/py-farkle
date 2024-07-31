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
        for i in list(self.memory):
            s += f'\t{i}\n'
        s += ')'
        return s


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


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 4
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
    if VERBOSE: print(f'choose ps with max E[future score] = {E_score}')
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
    if VERBOSE: print(ps_list)
    states = torch.tensor([[int(dh.score + ps.score),
                            dice_remaining_convert(len(dh.dice_values_free()) - ps.num_dice)]
                           for ps in ps_list],
                          device=device, dtype=torch.float32)
    if VERBOSE:
        print('states:')
        print(states)
    E_scores = evaluate_expected_scores(states)
    if VERBOSE:
        print('with E[future score]:')
        print(E_scores)

    if sample > eps_threshold:  # use max possible score
        return select_possible_score(ps_list, E_scores)
    else:
        if VERBOSE: print('select random ps')
        random_idx = random.choice(range(len(ps_list)))
        return ps_list[random_idx], E_scores[random_idx]


def decide_will_roll_again(E_score: float) -> bool:
    """roll again if positive expected future score"""
    if VERBOSE: print(f'decide to roll based on E[future score] > 0: {E_score > 0}')
    return bool(E_score > 0)


def decide_training_will_roll_again(E_score: float) -> bool:
    global TURNS_COMPLETE
    sample = random.random()
    eps_threshold = EPS_PS_END + (EPS_PS_START - EPS_PS_END) * \
                    math.exp(-1. * TURNS_COMPLETE / EPS_DECAY)
    if sample > eps_threshold:  # use roll if E_score > 0
        return decide_will_roll_again(E_score)
    else:
        if VERBOSE: print('select random roll again')
        return random.random() < 0.9  # 70% of time choose to roll again


def select_action(dh: DiceHand) -> tuple[DiceHand, bool]:
    """chooses possible score and whether to roll again"""
    global TURNS_COMPLETE  # idk where this needs to sit in the logic
    chosen_ps, E_score = select_training_possible_score(dh)
    will_roll_again = decide_training_will_roll_again(E_score)
    TURNS_COMPLETE += 1
    return chosen_ps, will_roll_again


TrainingState = namedtuple('TrainingState',
                           ('score', 'num_dice_remaining'))

# todo: don't think I need to record roll_again for use anywhere
Transition = namedtuple('Transition',
                        ('state', 'roll_again', 'next_state', 'reward'))


@dataclass(frozen=True)
class FarkleAction:
    possible_score: DiceHand
    will_roll_again: bool


def optimize_model():
    if len(MEMORY) < BATCH_SIZE:
        return
    transitions = MEMORY.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    if VERBOSE: print(batch)

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    next_states = torch.tensor(
        [[s.score, s.num_dice_remaining] if s is not None else [0, 0]
         for s in batch.next_state],
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        target_E_scores = target_net(next_states)

    reward_batch = torch.tensor([[r] for r in batch.reward], device=device, dtype=torch.float32)

    y = reward_batch
    # this is where i'd discount with GAMMA
    y[non_final_mask] += target_E_scores[non_final_mask]

    state_batch = torch.tensor([[s.score, s.num_dice_remaining] for s in batch.state],
                               device=device, dtype=torch.float32)

    policy_E_scores = policy_net(state_batch)

    if VERBOSE:
        print(f'state_batch\n{state_batch}')
        print(f'next_states\n{next_states}')
        print(f'target_net_e_scores\n{target_E_scores}')
        print(f'reward batch\n{reward_batch}')
        print(f'y\n{y}')
        print(f'policy_E_scores\n{policy_E_scores}')

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(target_E_scores, policy_E_scores)

    # Optimize the model
    OPTIMIZER.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    OPTIMIZER.step()


EPISODE_SCORES = []
def plot_scores(show_result=False):
    """plots score of dice turns in training"""
    plt.figure(1)
    durations_t = torch.tensor(EPISODE_SCORES, dtype=torch.float32)
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


OPTIMIZER = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
MEMORY = ReplayMemory(300)
NUM_TRAINING_TURNS = 2
TURNS_COMPLETE = 0
VERBOSE = True

for turn_idx in range(NUM_TRAINING_TURNS):
    if VERBOSE: print(f'\n----- turn {turn_idx} -----')
    dh = DiceHand()  # init as rolled dh
    if VERBOSE: print(dh)
    will_roll_again = True
    score_post = dh.score
    state_post = TrainingState(score=score_post, num_dice_remaining=dice_remaining_convert(len(dh.free_dice)))
    while not dh.farkled and will_roll_again:
        # capture initial dh state
        state_pre = state_post
        if VERBOSE: print(f'state pre: {state_pre}')

        # choose action if not farkled
        # but b/c we're in this loop we've already not farkled
        # policy_net makes decision
        chosen_ps, will_roll_again = select_action(dh)
        if VERBOSE:
            print('Chosen ps:', chosen_ps)
            print('Will roll again:', will_roll_again)
        # execute decision
        dh.lock_from_dicehand(chosen_ps)
        # save immediate reward of decision
        reward = chosen_ps.score
        if VERBOSE: print('reward:', reward)
        score_post = dh.score
        num_dice_remaining_post = dice_remaining_convert(len(dh.free_dice))
        state_post = TrainingState(score=score_post,
                                   num_dice_remaining=num_dice_remaining_post)
        if VERBOSE: print('state post', state_post)

        # save turn transition
        MEMORY.push(
            Transition(
                state=state_pre,
                roll_again=will_roll_again,
                next_state=state_post,
                reward=reward
            )
        )

        if will_roll_again:
            if dh.all_locked:
                dh.roll_all_dice()
            else:
                dh.roll()
            if VERBOSE: print(f'roll again result: {dh}')

    # finally if ended turn by farkle
    # `and will_roll_again` necessary since locking dice then choosing to not
    # roll again can cause dh.farkled == True
    turn_score = 0
    if dh.farkled and will_roll_again:
        if VERBOSE: print('farkled')
        # capture final transition to farkle state
        state_pre = state_post
        state_post = None
        reward = -1 * score_post
        MEMORY.push(
            Transition(
                state=state_pre,
                roll_again=True,  # bc if farkled there was the intent to roll again
                next_state=state_post,
                reward=reward
            )
        )
    else:
        turn_score = dh.score

    # train
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    # plot
    EPISODE_SCORES.append(turn_score)
    if not VERBOSE:
        plot_scores()

if VERBOSE:
    print(MEMORY)
    print(EPISODE_SCORES)

print('Complete')
if not VERBOSE:
    plot_scores(show_result=True)
    plt.ioff()
    plt.show()
