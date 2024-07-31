import abc
import time
import math
import random

import torch

from farkle import utils
from models import models
from farkle.logic.gameobjects import GameState
from farkle.logic.gameobjects import DiceHand, Turn, RollDecision


class Player(metaclass=abc.ABCMeta):
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0

    def __repr__(self):
        return f'{self.name}: {self.score} points'

    @abc.abstractmethod
    def play_dicehand(self, dh: DiceHand) -> tuple[DiceHand, bool]:
        """Handles player's rolling and scoring decisions and returns chosen ps and roll again decision"""


class DQNAgent(Player):
    def __int__(self, name: str) -> None:
        super().__init__(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # todo new or load
        self.policy_net = models.DQN().to(self.device)

        # training
        self.training = True

        self.roll_again_likelihood = 0.7

        self.turns_complete = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000

    def eps_threshold(self):
        """exponentially decays from eps_start to esp_end as turns_complete increases"""
        return (self.eps_end + (self.eps_start - self.eps_end) *
                math.exp(-1. * self.turns_complete / self.eps_decay))

    def play_dicehand(self, dh: DiceHand) -> tuple[DiceHand, bool]:
        """returns chosen ps and roll again decision"""
        ps_list = dh.possible_scores()
        states = torch.tensor([[int(dh.score + ps.score),
                                utils.dice_remaining_convert(len(dh.dice_values_free()) - ps.num_dice)]
                               for ps in ps_list],
                              device=self.device, dtype=torch.float32)
        state_scores = torch.tensor([[int(dh.score + ps.score)] for ps in ps_list])
        E_add_scores = self.expected_additional_score(states)
        E_scores = torch.where(E_add_scores > 0, E_add_scores, 0.0) + state_scores

        if self.training and random.random() < self.eps_threshold():
            # chose random move
            random_idx = random.choice(range(len(ps_list)))
            return ps_list[random_idx], random.random() < self.roll_again_likelihood
        else:
            # use model
            max_score_idx = E_scores.argmax()
            chosen_ps: DiceHand = ps_list[max_score_idx]
            roll_again = bool(E_add_scores[max_score_idx] > 0)
            return chosen_ps, roll_again

    def expected_additional_score(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(states)

class RobotPlayer(Player, metaclass=abc.ABCMeta):
    def __int__(self, name: str, delay_seconds: float = 1) -> None:
        super().__init__(name)
        self.delay_seconds = delay_seconds

    def play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        time.sleep(self.delay_seconds)
        return self.robot_play_dicehand(dice_hand, game_state)

    @abc.abstractmethod
    def robot_play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        """Implement robot turn logic"""


class RandomRobotPlayer(RobotPlayer):
    def __init__(self, name: str, delay_seconds: int):
        """This robot's dumb and makes random decisions"""
        super().__init__(name=name)
        self.delay_seconds = delay_seconds

    def robot_play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        score_decision: DiceHand = random.choice(dice_hand.possible_scores())
        will_roll_again = random.choice([True, False])
        post_dicehand = dice_hand.copy()
        post_dicehand.lock_from_dicehand(score_decision)
        return RollDecision(dice_hand, post_dicehand, will_roll_again)


class HumanPlayer(Player):
    def __int__(self, name: str):
        super().__init__(name)

    def play_dicehand(self, dice_hand, game_state: GameState) -> RollDecision:
        select_ps_index = int(input('Index of possible score to play: ')) - 1
        score_decision: DiceHand = dice_hand.possible_scores()[select_ps_index]
        select_roll_again = input('Do you want to roll again? [y/n] ')
        will_roll_again = select_roll_again.lower() == 'y'
        post_dicehand = dice_hand.copy()
        post_dicehand.lock_from_dicehand(score_decision)
        return RollDecision(dice_hand, post_dicehand, will_roll_again)
