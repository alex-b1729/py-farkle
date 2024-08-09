import os
import sys
import json
import random
import numpy as np
from time import perf_counter
from collections import deque, namedtuple, defaultdict

import torch.nn as nn
import torch.nn.functional as F

from farkle import utils
from farkle.logic import scoring
from farkle.logic import gameobjects as go


MODEL_DIR_PATH = '../../../models'


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


TrainingState = namedtuple('TrainingState',
                           ('score', 'num_dice_remaining'))

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward'))


def farkle_counts(num_dice: int = 6, rule_set: str = None) -> dict[int: int]:
    """return dict mapping num dice to number of farkle rolls under rule_set"""
    rs = None
    if rule_set is not None:
        if hasattr(scoring, rule_set):
            rs = getattr(scoring, rule_set)
    rs = rs if rs is not None else scoring.STANDARD_SCORING_HANDS_DICT
    farkle_counts = defaultdict(int)
    roll_outcome_dict = utils.roll_outcome_dict(num_dice=num_dice)
    for nd in roll_outcome_dict.keys():
        for dvals in roll_outcome_dict[nd]:
            if go.DiceHand(dvals, rule_set=rs).farkled:
                farkle_counts[nd] += 1
    return dict(farkle_counts)


def farkle_probs(num_dice: int = 6, rule_set: str = None) -> dict[int: float]:
    """return dict mapping num dice to prob farkling under rule_set"""
    fc = farkle_counts(num_dice, rule_set)
    roll_outcome_dict = utils.roll_outcome_dict(num_dice=num_dice)
    return {nd: fc[nd] / len(roll_outcome_dict[nd]) for nd in range(1, num_dice+1)}


def farkle_stats(num_dice: int = 6, rule_set: str = None) -> dict:
    """
    Returns dict of farkle_counts and farkle_probs.
    Saves/reads as .json from model directory if doesn't/does exist.
    """
    rs = None
    if rule_set is not None:
        if hasattr(scoring, rule_set):
            rs = getattr(scoring, rule_set)
    rs = rs if rs is not None else scoring.STANDARD_SCORING_HANDS_DICT
    filename = (f'farkle_stats'
                f'{"_" + str(num_dice) if num_dice != 6 else ""}'
                f'{"_" + rule_set if rule_set is not None else ""}'
                f'.json')
    stats_path = os.path.join(MODEL_DIR_PATH, filename)
    if os.path.isfile(stats_path):
        with open(stats_path, 'r') as f:
            _stats = json.loads(f.read())
        stats = {}
        for k in ['farkle_counts', 'farkle_probs']:
            stats[k] = {nd: _stats[k][str(nd)] for nd in range(1, num_dice+1)}
    else:
        fc = farkle_counts(num_dice, rule_set)
        fp = farkle_probs(num_dice, rule_set)
        stats = {
            'farkle_counts': fc,
            'farkle_probs': fp
        }
        with open(stats_path, 'w') as f:
            f.write(json.dumps(stats, indent=4))
    return stats


class DicePointsRollExpectation(object):
    def __init__(self, dice_remaining: int, points: int, farkle_prob: float, init_expectation: float = 0):
        self.dice_remaining = dice_remaining
        self.points = points
        self.farkle_prob = farkle_prob

        self.iteration = 0

        self.score_list = []
        self.exp_additional_points_if_not_farkle = init_expectation

    @property
    def E_roll_points(self):
        """E[points you'll end the turn with | roll again]"""
        return (1 - self.farkle_prob) * self.exp_additional_points_if_not_farkle - (self.farkle_prob * self.points)

    def cache_score(self, score):
        """add the score so can later take average"""
        self.score_list.append(score)

    def update_expectation(self):
        self.exp_additional_points_if_not_farkle = sum(self.score_list) / len(self.score_list)
        self.score_list = []
        self.iteration += 1

    def __repr__(self):
        return f'ScoreAndDiceHand(nd={self.dice_remaining}, pts={self.points})'


class ExpectedPointsValueIteration(object):
    def __init__(self,
                 num_dice: int = 6,
                 max_points: int = 10_000,
                 rule_set: str = 'STANDARD_SCORING_HANDS_DICT',
                 init_expectation: float = 0):
        self.num_dice = num_dice
        self.max_points = max_points
        self.rule_set = rule_set

        self.dice_range = range(1, self.num_dice+1)
        self.upper_points_slice = -20
        self.points_range = range(0, self.max_points - (self.upper_points_slice * 50), 50)

        fs = farkle_stats(self.num_dice, self.rule_set)
        self.farkle_counts = fs['farkle_counts']
        self.farkle_probs = fs['farkle_probs']

        # {dice_remaining: {points: DicePointsRollExpectation(), ...}, ...}
        self.dpre_dict = {
            nd: {
                pts: DicePointsRollExpectation(nd, pts, self.farkle_probs[nd], init_expectation)
                for pts in self.points_range
            } for nd in self.dice_range
        }

    def save(self, path: str):
        dpre = {
            nd: {
                pts: self.dpre_dict[nd][pts].E_roll_points
                for pts in self.points_range[:self.upper_points_slice]
            } for nd in self.dice_range
        }
        with open(path, 'w') as f:
            f.write(json.dumps(dpre, indent=4))

    @staticmethod
    def load(path: str) -> dict:
        with open(path, 'r') as f:
            dprej = json.loads(f.read())
        dpre = {
            int(nd): {
                int(pts): dprej[nd][pts] for pts in dprej[nd].keys()
            } for nd in dprej.keys()
        }
        return dpre

    def dpre_array(self, dpre: dict = None) -> np.array:
        """
        Given dict mapping num_dice to dicts mapping points to DicePointsRollExpectation objects,
        return mxn array of expected points where m index is num_points
        and n index is points index starting at 0 and increasing by 50 for each index.
        Assumes dpre dictionary is sorted by num_dice and points
        [
            1_dice[expectation_0_points, expectation_50_points, expectation_100_points, ...],
            2_dice[expectation_0_points, expectation_50_points, expectation_100_points, ...],
            3_dice[expectation_0_points, expectation_50_points, expectation_100_points, ...],
            ...
        ]
        """
        dpre = dpre if dpre is not None else self.dpre_dict
        return np.array([
            [dpre[nd][pts].E_roll_points for pts in dpre[nd].keys()] for nd in dpre.keys()
        ])

    def mse(self, dpre1: np.array, dpre2: np.array) -> np.array:
        return ((dpre1 - dpre2) ** 2).mean(axis=1)

    def iter_expectations(self, num_epochs: int = None, max_mse: float = None) -> list:
        assert num_epochs is not None or max_mse is not None
        if num_epochs is None:
            num_epochs = 100
        if max_mse is None:
            max_mse = 0.0

        roll_outcome_dict = utils.roll_outcome_dict(self.num_dice)

        mse = np.array([100])
        mse_list = []
        prev_dpre_array = self.dpre_array()
        print(f'Iterating roll expected value for {self.num_dice} dice for points up to {self.max_points}')
        t0 = perf_counter()
        for epoch in range(num_epochs):
            if mse.max() < max_mse:
                break

            # plot_expectations(exp_points_dict, title=f'{epoch}')

            for nd in self.dice_range:
                dice_combo_count = len(roll_outcome_dict[nd])

                for i, roll_outcome in enumerate(roll_outcome_dict[nd]):
                    if round(100 * i / dice_combo_count, 0) % 10 == 0:
                        sys.stdout.write(f'\rEpoch {epoch} - {nd} dice: {round(100 * i / dice_combo_count, 2)}%')
                    dh = go.DiceHand(roll_outcome)

                    if not dh.farkled:
                        pss = dh.possible_scores()

                        for pt in self.points_range[:self.upper_points_slice]:
                            E_points_list = []

                            for ps in pss:
                                # how many dice remaining if take this ps?
                                dice_remaining = utils.dice_remaining_convert(nd - ps.num_dice)
                                total_points = pt + ps.score

                                # just ignoring the upper bound for now
                                if total_points < self.max_points - (self.upper_points_slice * 50):
                                    # Get the current estimate of the expected points if roll with that many
                                    # dice_remaining.
                                    # If E[points | roll again] is < points_if_don't_roll_again then we just
                                    # wouldn't roll again
                                    E_roll_points = max(self.dpre_dict[dice_remaining][total_points].E_roll_points, 0)
                                    E_points_list.append(ps.score + E_roll_points)

                            self.dpre_dict[nd][pt].cache_score(max(E_points_list))

            # update all expectations
            for nd in self.dice_range:
                for pt in self.points_range[:self.upper_points_slice]:
                    self.dpre_dict[nd][pt].update_expectation()

            new_dpre_array = self.dpre_array()
            mse = self.mse(prev_dpre_array, new_dpre_array)
            mse_list.append(mse)
            prev_dpre_array = np.copy(new_dpre_array)

            sys.stdout.write(f'\rEpoch {epoch} complete - max mean squared diff from last: {round(mse.max(), 3)}\n')

        t1 = perf_counter()

        print('Complete ', end='')
        print(f'Time: {round((t1 - t0) / 60, 2)} minutes')

        return mse_list
