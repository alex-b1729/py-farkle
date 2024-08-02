import copy
import random
import logging
import numpy as np
from typing import NamedTuple
from time import perf_counter
from dataclasses import dataclass
from functools import cached_property


class Die(object):
    def __init__(self, value: int | None = None):
        if value is not None:
            assert isinstance(value, int)
            assert value in range(1, 7)
            self.value = value
        else:
            self.value = random.randint(1, 6)

    def roll(self):
        self.value = random.randint(1, 6)

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other.value


@dataclass
class GameDie:
    """Die used in farkle game"""
    die: Die
    locked: bool = False


class DiceHand(object):
    def __init__(self, *args, **kwargs):
        """
        When called with no arguments generates DiceHand with 6 random dice.
        When num_dice is not None, generates num_dice random die.
        When called with *args uses those values as values of dice.
        When args is a single list uses those values for dice.
        When args is str of int use those as values for dice.
        If both num_dice is not None and args are passed, num_dice is ignored.
        """
        # TODO: tempted to make self.dice = None unless self.roll() has been called
        # but not sure what this would affect in current code
        if not args:
            self.num_dice = 6 if 'num_dice' not in kwargs else kwargs['num_dice']
            self.dice = {i: GameDie(Die()) for i in range(self.num_dice)}
        elif len(args) == 1 and isinstance(args[0], list):
            self.num_dice = len(args[0])
            self.dice = {i: GameDie(Die(args[0][i]))
                         for i in range(len(args[0]))}
        elif len(args) == 1 and isinstance(args[0], str):
            self.num_dice = len(args[0])
            self.dice = {i: GameDie(Die(int(args[0][i]))) for i in range(self.num_dice)}
        else:  # len(args) > 1
            for i in args:
                if not isinstance(i, int):
                    raise TypeError('args must be all type int or '
                                    'only one arg of type list')
            self.num_dice = len(args)
            self.dice = {i: GameDie(Die(args[i]))
                         for i in range(len(args))}
        self.score = 0 if 'score' not in kwargs else kwargs['score']

    def roll_all_dice(self):
        """rolls and unlocks all dice"""
        for i, gd in self.dice.items():
            gd: GameDie
            gd.die.roll()
            gd.locked = False

    def reset_dice(self):
        """rolls all dice and sets score = 0"""
        self.roll_all_dice()
        self.score = 0

    def roll(self):
        """roll non-locked dice"""
        for i, d in self.dice.items():
            if not d.locked:
                d.die.roll()

    def lock_dice(self, *args):
        """Locks dice at given indexes"""
        for i in args:
            self.dice[i].locked = True

    def lock_from_dicehand(self, other):
        """Lock dice corresponding to unlocked dice from other and add score.
        All dice values in other must be in and unlocked in self"""
        self.score += other.score
        for i, d in other.free_dice.items():
            found = False
            i = 0
            while not found and i < self.num_dice:
                d_s = self.dice[i]
                if not d_s.locked and d.die == d_s.die:
                    d_s.locked = True
                    found = True
                i += 1
            if not found:  # if die in other not found in self raise error
                raise ValueError('Die in other not in and unlocked in self')

    def add_dice(self, *args):
        """Adds dice of given values
        If args is a single list the int values are used as new dice values"""
        max_die_ind = max(self.dice.keys())
        new_dice_vals = None
        if len(args) == 1 and isinstance(args[0], list):
            new_dice_vals = args[0]
        else:  # len(args) > 1
            for i in args:
                if not isinstance(i, int):
                    raise TypeError('args must be all type int or '
                                    'only one arg of type list')
            new_dice_vals = list(args)
        if new_dice_vals:
            for i, val in enumerate(new_dice_vals):
                self.num_dice += 1
                self.dice[i + max_die_ind + 1] = GameDie(Die(val), False)

    def add_from_dicehand(self, other):
        """combines non-locked dice from other into self"""
        self.add_dice(other.dice_values_free())

    def __add__(self, other):
        """combines dice and adds scores"""
        new_score = self.score + other.score
        new_dice = self.copy()
        new_dice.add_from_dicehand(other)
        return DiceHand(new_dice.dice_values(), score=new_score)

    def __eq__(self, other):
        """equal if same scores and same values of locked and unlocked dice"""
        try:
            score_eq = self.score == other.score
            free_eq = self.dice_values_free() == other.dice_values_free()
            locked_eq = self.dice_values_locked() == other.dice_values_locked()
            return score_eq and free_eq and locked_eq
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        s = (self.score,
             ','.join([str(i) for i in self.dice_values_free()]),
             ','.join([str(i) for i in self.dice_values_locked()]))
        return hash(s)

    # id like this to be a @property but want to
    # make it like as dictionary.values()
    def dice_values(self) -> list:
        """returns list of values of dice"""
        gd: GameDie
        return [gd.die.value for gd in self.dice.values()]

    def dice_values_free(self) -> list:
        gd: GameDie
        return sorted([gd.die.value for gd in self.dice.values() if not gd.locked])

    def dice_values_locked(self) -> list:
        gd: GameDie
        return sorted([gd.die.value for gd in self.dice.values() if gd.locked])

    @property
    def str_dice_values(self) -> str:
        return ''.join(str(d) for d in self.dice_values())

    @property
    def str_dice_values_free(self) -> str:
        return ''.join(str(d) for d in self.dice_values_free())

    @property
    def free_dice(self):
        return {i: d for i, d in self.dice.items() if not d.locked}

    @property
    def locked_dice(self):
        return {i: d for i, d in self.dice.items() if d.locked}

    @property
    def all_locked(self):
        return not self.free_dice

    @staticmethod
    def with_dice_values(*args):
        """Accepts dice values as separate args or one list as arg"""
        dh = DiceHand(num_dice=len(args))
        dice_vals = None
        if len(args) == 1 and isinstance(args[0], list):
            dice_vals = [i for i in args[0]]
        else:
            for i in args:
                if not isinstance(i, int):
                    raise TypeError('args must be all type int or '
                                    'only one arg of type list')
            dice_vals = [i for i in args]
        if dice_vals is not None:
            dh.dice = {i: GameDie(Die(dice_vals[i]))
                       for i in range(len(dice_vals))}
        else:
            raise RuntimeError('Why\'d this trip?')
        return dh

    def __repr__(self):
        # s = ''
        # for i, d in self.free_dice.items():
        #     s += f'{i}: {d.die}\n'
        # for i, d in self.locked_dice.items():
        #     s += f'{i}: {d.die} - Locked\n'
        # s += f'Score: {self.score}'
        # return s
        return (f'DiceHand('
                f'free=[{", ".join([str(d) for d in self.dice_values_free()])}], '
                f'locked=[{", ".join([str(d) for d in self.dice_values_locked()])}], '
                f'score={self.score})')

    @staticmethod
    def parse_repr(_repr: str):
        """parses str representation of a DiceHand obj and returns a DiceHand obj"""
        free_loc = _repr.find('free=[')
        locked_loc = _repr.find('], locked=[')
        score_loc = _repr.find('], score=')
        end_start_loc = _repr.find(')')
        free_dice_nums = [int(i) for i in _repr[free_loc + 6: locked_loc].split(', ') if i != '']
        locked_dice_nums = [int(i) for i in _repr[locked_loc + 11: score_loc].split(', ') if i != '']
        score = int(_repr[score_loc + 9: end_start_loc])
        dh = DiceHand(free_dice_nums + locked_dice_nums, score=score)
        num_dice = len(free_dice_nums + locked_dice_nums)
        lock_idxs = [i for i in range(num_dice - len(locked_dice_nums), num_dice)]
        dh.lock_dice(*lock_idxs)
        return dh

    @property
    def json_encode(self):
        return {'__DiceHand__': True,
                'free': self.dice_values_free(),
                'locked': self.dice_values_locked(),
                'score': self.score}

    @staticmethod
    def json_decode(dct):
        dh = DiceHand(dct['free'] + dct['locked'], score=dct['score'])
        num_dice = len(dct['free'] + dct['locked'])
        lock_idxs = [i for i in range(num_dice - len(dct['locked']), num_dice)]
        dh.lock_dice(*lock_idxs)
        return dh

    def __contains__(self, key):
        """checks if all dice in key are included and unlocked in self
        and that scores are equal"""
        is_contained = False
        if isinstance(key, DiceHand) and len(self.free_dice) >= len(key.free_dice):
            dice_self = self.copy()  # without will modify and lock self
            dice_compare: DiceHand = key.copy()
            didnt_find = False
            d_count = 0
            while not didnt_find and d_count < dice_compare.num_dice:
                if not dice_compare.dice[d_count].locked:
                    die_compare: Die = dice_compare.dice[d_count].die
                    found_in = False
                    i = 0
                    while not found_in and i < dice_self.num_dice:
                        d = dice_self.dice[i]
                        found_in = die_compare == d.die and not d.locked
                        i += 1
                        # if the comparison dice found in hand then lock
                        # so can no longer compare to that die
                        if found_in: d.locked = True
                    # if found_in still False
                    # means didn't find the dice (unlocked) in hand
                    if not found_in: didnt_find = True
                    d_count += 1
            # if didnt_find still false
            # means found all comparison dice in self
            if not didnt_find:  # and self.score == dice_compare.score:
                is_contained = True
        return is_contained

    def copy(self):
        return copy.deepcopy(self)

    def count(self, other) -> int:
        """returns number of occurrences of other in self"""
        dice_self = self.copy()
        dice_other = other.copy()
        count = 0
        while dice_other in dice_self:
            count += 1
            dice_self.lock_from_dicehand(dice_other)
        return count

    def _duplicate_possible_scores(self, dice_count_list: list[int]) -> tuple[list, list]:
        """dice_count_list gives count of dice with value of that index.
        E.g. [2,0,1,0,2,0] is a dice had with free dice [1,1,3,5,5]"""
        return_ps_list = []
        return_score_list = []
        for die_idx in range(6):
            dc = dice_count_list[die_idx]
            if dc >= min(SCORING_HANDS_DICT[die_idx + 1].keys()):
                this_ps_list = []
                this_score_list = []
                for dice_score_count in SCORING_HANDS_DICT[die_idx + 1]:
                    if dc >= dice_score_count:
                        this_ps = np.array([0 for _ in range(6)])
                        this_score = SCORING_HANDS_DICT[die_idx + 1][dice_score_count]
                        this_ps[die_idx] = dice_score_count
                        remaining_count = dc - dice_score_count
                        remaining_dice_count_list = [i for i in dice_count_list]
                        remaining_dice_count_list[die_idx] = remaining_count

                        remaining_ps_list, remaining_score_list = self._duplicate_possible_scores(
                            remaining_dice_count_list)

                        combined_ps_list = [this_ps + r_ps for r_ps in remaining_ps_list]
                        combined_score_list = [this_score + s for s in remaining_score_list]

                        combined_ps_list = [this_ps] + remaining_ps_list + combined_ps_list
                        combined_score_list = [this_score] + remaining_score_list + combined_score_list

                        this_ps_list += combined_ps_list
                        this_score_list += combined_score_list

                return_ps_list += this_ps_list
                return_score_list += this_score_list
                return return_ps_list, return_score_list

        return [], []

    def all_possible_scores(self) -> list:
        dice_count_list = [self.str_dice_values_free.count(str(i)) for i in range(1, 7)]
        hands, scores = self._duplicate_possible_scores(dice_count_list)
        # hand_strs = [''.join([str(d + 1) * int(c) for d, c in enumerate(hand)]) for hand in hands]
        ps_set = {''.join(ps.astype(str)) + str(s) for ps, s in list(zip(hands, scores))}
        return [DiceHand(''.join([str(d + 1) * int(c) for d, c in enumerate(s[:5])]),
                         score=int(s[5:])) for s in ps_set]

    def possible_scores(self) -> list:
        dice_count_list = [self.str_dice_values_free.count(str(i)) for i in range(1, 7)]
        hands, scores = self._duplicate_possible_scores(dice_count_list)
        numdie = [a.sum() for a in hands]

        zipped_pss = list(zip(numdie, hands, scores))
        str_pss = [str(nd) + str(s).zfill(4) + ''.join(ps.astype(str)) for nd, ps, s in zipped_pss]
        str_pss.sort()

        nd_points_and_dh_dict = {int(s[0]): s[1:] for s in str_pss}

        return [DiceHand(''.join([str(d + 1) * int(c) for d, c in enumerate(s[-6:])]), score=int(s[:4]))
                for nd, s in nd_points_and_dh_dict.items()]

    @property
    def farkled(self) -> bool:
        """True if no possible scores and all dice are not locked"""
        return self.possible_scores() == [] and not self.all_locked

    def score_from_dicehand(self, dice_hand):
        """add score and locks corresponding dice"""
        assert dice_hand in self
        # self.score += dice_hand.score  # done in self.lock_from_dicehand
        self.lock_from_dicehand(dice_hand)


# TODO: Add ability to define scoring patterns such as 3 pairs
SCORING_HANDS = (

    DiceHand(1, score=100)
    , DiceHand(5, score=50)
    , DiceHand(1, 1, 1, score=1000)
    , DiceHand(2, 2, 2, score=200)
    , DiceHand(3, 3, 3, score=300)
    , DiceHand(4, 4, 4, score=400)
    , DiceHand(5, 5, 5, score=500)
    , DiceHand(6, 6, 6, score=600)

)

SCORING_HANDS_DICT = {1: {1: 100, 3: 1000},
                      2: {3: 200},
                      3: {3: 300},
                      4: {3: 400},
                      5: {1: 50, 3: 500},
                      6: {3: 600}}


@dataclass(frozen=True)
class RollDecision:
    dicehand_pre: DiceHand
    dicehand_post: DiceHand
    will_roll_again: bool


@dataclass(frozen=True)
class GameState:
    scores: dict[str: int]  # player_name: points
    current_player_name: str
    dice_hand: DiceHand
    goal_score: int = 5000

    def __post_init__(self):
        """validation"""
        # TODO: validate current_player_name in scores.keys()

    @cached_property
    def game_over(self) -> bool:
        return any([i >= self.goal_score for i in self.scores.values()])

    @cached_property
    def winner(self) -> str | None:
        """returns name of winning player"""
        if self.game_over:
            for p, s in self.scores.items():
                if s > self.goal_score:
                    return p
        else:
            return None


@dataclass(frozen=True)
class Turn:
    player_name: str
    points_earned: int


@dataclass(frozen=True)
class FarkleAction:
    possible_score: DiceHand
    will_roll_again: bool


if __name__ == '__main__':
    pass
    # dh1 = DiceHand(1, 1, 5)
    # dh1 = DiceHand('123456')
    # print(dh1)
    # print(dh1.str_dice_values_free)
    # print(dh1.possible_scores())
    # dh1 = DiceHand(1,1,1)
    # dh2 = dh1.copy()
    # print(dh1._all_duplicate_possible_scores(dh2), end='\n\n')
    # print(dh1._all_duplicate_possible_scores_updated(dh2))
    # print(dh1.possible_scores(), end='\n\n')
    # print(dh1.possible_scores_updated())
    # for sh in SCORING_HANDS:
    #     print(sh)
