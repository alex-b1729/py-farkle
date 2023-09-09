from dataclasses import dataclass

from farkle.game.players import Player, RandomRobotPlayer
from farkle.game.renderers import Renderer
from farkle.logic.models import GameState, DiceHand, Turn

@dataclass(frozen=True)
class Farkle:
    players: list[Player]
    renderer: Renderer
    goal_score: int = 5000

    def __post_init__(self):
        """validate everything"""

    def __repr__(self):
        return '\n'.join([f'{p}' for p in self.players])

    def player_sequence(self) -> iter:
        i = 0
        while True:
            yield self.players[i % len(self.players)]
            i += 1

    def player_turn(self,
                    player: Player,
                    dice_hand: DiceHand,
                    game_state_pre: GameState,
                    renderer: Renderer) -> Turn:
        dice_hand.reset_dice()
        farkled = False
        will_roll_again = True

        while will_roll_again and not farkled:
            # if player has hot dice and can re-roll all
            if dice_hand.all_locked:
                dice_hand.roll_all_dice()
            else:
                dice_hand.roll()
            farkled = dice_hand.farkled

            # render current state before player makes roll decision
            game_state = GameState(game_state_pre.scores,
                                   player.name,
                                   dice_hand,
                                   game_state_pre.goal_score)
            renderer.render(game_state)

            if not farkled:
                roll_decision = player.play_dicehand(dice_hand, game_state)
                dice_hand = roll_decision.dicehand_post
                will_roll_again = roll_decision.will_roll_again

        points_earned = dice_hand.score if not farkled else 0

        return Turn(player.name, points_earned)

    def play(self) -> None:
        scores = {p.name: 0 for p in self.players}
        dice_hand = DiceHand()

        whos_turn = self.player_sequence()
        player = next(whos_turn)

        game_state = GameState(scores=scores,
                               current_player_name=player.name,
                               dice_hand=dice_hand,
                               goal_score=self.goal_score)

        while True:
            self.renderer.render(game_state)
            # TODO: maybe need a play_turn() attribute for Farkle to render between rolls
            if game_state.game_over:
                break
            self.player_turn(player, dice_hand, game_state, self.renderer)
            player = next(whos_turn)


if __name__ == '__main__':
    rrp = RandomRobotPlayer('rrp')
    f = Farkle([rrp], None)
    print(f)
    p = f.players[0]
    p.score = 10
    print(f)

