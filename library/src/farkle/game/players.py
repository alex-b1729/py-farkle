import abc
import time
import random

from farkle.logic.models import GameState
from farkle.logic.models import DiceHand, Turn, RollDecision


class Player(metaclass=abc.ABCMeta):
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0

    def __repr__(self):
        return f'{self.name}: {self.score} points'

    @abc.abstractmethod
    def play_dicehand(self, dice_hand, game_state: GameState) -> RollDecision:
        """Handles player's rolling and scoring decisions and returns points earned"""


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
