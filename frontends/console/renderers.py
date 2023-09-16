import time

from farkle.logic.models import GameState
from farkle.game.renderers import Renderer


class ConsoleRenderer(Renderer):
    def render(self, game_state: GameState) -> None:
        clear_screen()
        # goal score
        print(f'Goal Score: {game_state.goal_score}', end='\n\n')
        # score board
        for p, s in game_state.scores.items():
            p: str
            s: int
            if p == game_state.current_player_name:
                print('\t*\t', end='')
            print(f'{p}: {s} points', end='\n')
        if not game_state.game_over:
            # dice hand
            print('\nDice:\n', game_state.dice_hand, end='')
            # scoring options
            ps: list = game_state.dice_hand.possible_scores()
            for i in range(len(ps)):
                print(f'{i+1}: {ps[i].score} points\n', ps[i], end='\n\n')
        else:
            print(game_over())
        # inp = input('press ')

def clear_screen() -> None:
    print("\033c", end="")

def blink(text: str) -> str:
    return f'\033[5m{text}\033[0m'

def game_over() -> str:
    return """
                                                                                        
    //   ) )                                    //   ) )                         // 
   //         ___      _   __      ___         //   / /        ___      __      //  
  //  ____  //   ) ) // ) )  ) ) //___) )     //   / /|  / / //___) ) //  ) )  //   
 //    / / //   / / // / /  / / //           //   / /|| / / //       //             
((____/ / ((___( ( // / /  / / ((____       ((___/ / ||/ / ((____   //       //  
    """
