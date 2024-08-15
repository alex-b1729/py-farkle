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
            print(f'\n--- Player {game_state.current_player.name} begins turn '
                  f'with {game_state.current_player.score} ---')
            print(f'Current scores:')
            for n, s in scores.items():
                print(f'\t{n}: {s}')
            print(f'{game_state.current_player.name}\'s rolls:')

        locked_dice = ''.join([str(d) for d in game_state.dh_pre.dice_values_locked()])
        remaining_dice = ''.join([str(d) for d in game_state.dh_post.dice_values_free()])
        if not game_state.farkled:
            chosen_dice = ''.join([str(d) for d in game_state.chosen_ps.dice_values_free()])
            print(f'\tRoll: {" " * len(chosen_dice)}{remaining_dice}'
                  f'\tPlayed: {locked_dice}'
                  f'\tScore: {game_state.dh_pre.score}')
            print(f'\t      {chosen_dice}{" " * len(remaining_dice)}'
                  f'\tWill{" not" if not game_state.will_roll_again else ""} roll again')
        else:
            print(f'\tRoll: {remaining_dice}'
                  f'\tPlayed: {locked_dice}'
                  f'\tScore: {game_state.dh_pre.score}')
            print(f'\t      !! FARKLED !!')

        if game_state.farkled or not game_state.will_roll_again:
            points_gained = game_state.dh_post.score if not game_state.farkled else 0
            print(f'--- {game_state.current_player.name} gained {points_gained} '
                  f'for {game_state.current_player.score + points_gained} total points ---')

        if game_state.game_over:
            print(f'\n-- Game Over - {game_state.winner} wins! --')
