import abc
from dataclasses import dataclass

from farkle.logic.gameobjects import GameState


class Renderer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def render(self, game_state: GameState) -> None:
        """renders game state"""


class ConsoleRenderer(Renderer):
    def render(self, game_state: GameState) -> None:
        if game_state.start_of_turn:
            scores = {p.name: p.score for p in game_state.players}
            print(f'{scores}')
            print(f'--- Player {game_state.current_player.name} begins turn with {game_state.current_player.score} ---')

        locked_dice = ''.join([str(d) for d in game_state.dh_pre.locked_dice])
        remaining_dice = ''.join([str(d) for d in game_state.dh_post.free_dice])
        if not game_state.farkled:
            chosen_dice = ''.join([str(d) for d in game_state.chosen_ps.free_dice])
            print(f'\tRoll: {" " * len(chosen_dice)}{remaining_dice}\tPlayed: {locked_dice}')
            print(f'\t      {chosen_dice}{" " * len(remaining_dice)}')
        else:
            print(f'\tRoll: {remaining_dice}\tPlayed: {locked_dice}')
            print(f'\t      !! FARKLED !!')

        if game_state.farkled or not game_state.will_roll_again:
            print(f'--- {game_state.current_player.name} gained {game_state.dh_post.score} '
                  f'for {game_state.current_player.score} total points ---')

        if game_state.game_over:
            print(f'Game Over - {game_state.winner} wins!')
