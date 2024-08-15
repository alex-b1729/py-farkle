from dataclasses import dataclass

from farkle.game.players import Player, EVMaximizingPlayer, RandomPlayer
from farkle.game.renderers import Renderer, ConsoleRenderer
from farkle.logic.gameobjects import (
    GameState,
    DiceHand,
    Turn,
    RollDecision
)

@dataclass(frozen=True)
class Farkle:
    players: list[Player]
    renderer: Renderer
    goal_score: int = 10_000

    def __post_init__(self):
        """validate everything"""

    def __repr__(self):
        s = 'Farkle(\n'
        s += '\n\t'.join([f'{p}' for p in self.players])
        s += '\n)'
        return s

    def player_sequence(self) -> iter:
        i = 0
        while True:
            yield self.players[i % len(self.players)], i % len(self.players)
            i += 1

    def play(self) -> None:
        print('playing farkle')
        whos_turn = self.player_sequence()
        game_over = False

        while not game_over:
            player, player_idx = next(whos_turn)
            start_of_turn = True
            for turn in player.play_turn():
                turn: RollDecision
                farkled = turn.chosen_ps is None

                game_state = GameState(
                    players=self.players,
                    current_player_idx=player_idx,
                    dh_pre=turn.dh_pre,
                    dh_post=turn.dh_post,
                    chosen_ps=turn.chosen_ps,
                    start_of_turn=start_of_turn,
                    farkled=farkled,
                    will_roll_again=turn.will_roll_again,
                    goal_score=self.goal_score
                )
                self.renderer.render(game_state)

                start_of_turn = False

            if not farkled:
                player.score += turn.dh_post.score

            game_over = game_state.game_over


if __name__ == '__main__':
    ev_player = EVMaximizingPlayer('EVMaximus')
    ev_player.load_roll_evs('py-farkle/models/roll_EV_12000.json')
    rand_player = RandomPlayer('RandoRambo')
    r = ConsoleRenderer()
    f = Farkle([ev_player, rand_player], r, 5_000)
    f.play()
