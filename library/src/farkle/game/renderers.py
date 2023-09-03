from dataclasses import dataclass

from ...farkle.logic.models import GameState


class Renderer:
    """renders game state"""

    def render(self, game_state: GameState) -> str:
        """fill out"""
