import abc
from dataclasses import dataclass

from farkle.logic.models import GameState


class Renderer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def render(self, game_state: GameState) -> None:
        """renders game state"""
