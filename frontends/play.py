from farkle.game.engine import Farkle
from farkle.game.players import RandomRobotPlayer

from console.renderers import ConsoleRenderer

player1 = RandomRobotPlayer(name='robot1', delay_seconds=1)
player2 = RandomRobotPlayer(name='ROBOT2', delay_seconds=1)

Farkle(players=[player1, player2],
       renderer=ConsoleRenderer(),
       goal_score=1000).play()
