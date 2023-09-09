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

    def play_turn(self, game_state: GameState) -> GameState:
        """updates player points and returns new game state"""
        dice_hand = DiceHand()  # initial roll

        farkled = dice_hand.farkled
        will_roll_again = not farkled

        while will_roll_again and not farkled:
            roll_decision = self.play_dicehand(dice_hand, game_state)
            dice_hand = roll_decision.dicehand_post
            will_roll_again = roll_decision.will_roll_again
            farkled = dice_hand.farkled

        points = dice_hand.score if not farkled else 0
        self.score += points
        return game_state.update_score(self.name, points)

    @abc.abstractmethod
    def play_dicehand(self, dice_hand, game_state: GameState) -> RollDecision:
        """Handles player's rolling and scoring decisions and returns points earned"""


class RobotPlayer(Player, metaclass=abc.ABCMeta):
    def __int__(self, name: str, delay_seconds: float = 0.5) -> None:
        super().__init__(name)
        self.delay_seconds = delay_seconds

    def play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        time.sleep(self.delay_seconds)
        return self.robot_play_dicehand(dice_hand, game_state)

    @abc.abstractmethod
    def robot_play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        """Implement robot turn logic"""


class RandomRobotPlayer(RobotPlayer):
    """This robot's dumb and makes random decisions"""
    def robot_play_dicehand(self, dice_hand: DiceHand, game_state: GameState) -> RollDecision:
        score_decision: DiceHand = random.choice(dice_hand.possible_scores())
        will_roll_again = random.choice([True, False])
        post_dicehand = dice_hand.copy()
        post_dicehand.lock_from_dicehand(score_decision)
        return RollDecision(dice_hand, post_dicehand, will_roll_again)
