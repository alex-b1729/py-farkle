from dataclasses import dataclass

from ...farkle.game.players import Player
from ...farkle.game.renderers import Renderer
from ...farkle.logic.models import GameState

@dataclass(frozen=True)
class Farkle:
    players: list[Player]
    renderer: Renderer

    current_player_index: int = 0
    goal_score: int = 5000

    def __post_init__(self):
        """validate everything"""
        # TODO: validate current_player_index in range(len(players)) etc

    def play(self) -> None:
        game_state = GameState(scores={p.name: 0 for p in self.players},
                               current_player=self.players[self.current_player_index].name,
                               goal_score=self.goal_score)
        while True:
            self.renderer.render(game_state)
            if game_state.game_over:
                break
            player = self.players[self.current_player_index]
            game_state = player.play_turn(game_state)
