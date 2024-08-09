import json
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


def dice_remaining_convert(n: int) -> int:
    """convert 0 dice remaining to 6"""
    return int((n - 1) % 6 + 1)


def roll_outcome_dict(num_dice: int = 6) -> dict[int: list]:
    """returns dict mapping number of dice to list of every possible roll outcome"""
    dice_range = range(1, num_dice+1)
    roll_outcome_list = [[i] for i in dice_range]  # init for 1 die
    roll_outcome_dict = {1: roll_outcome_list}
    for i in range(2, 7):
        roll_outcome_list = [
            [i] + j
            for i in dice_range
            for j in roll_outcome_list
        ]
        roll_outcome_dict[i] = roll_outcome_list
    return roll_outcome_dict


def load_roll_ev(path: str, as_numpy: bool = False) -> Union[dict, np.array]:
    with open(path, 'r') as f:
        dprej = json.loads(f.read())
    if not as_numpy:
        dpre = {
            int(nd): {
                int(pts): dprej[nd][pts] for pts in dprej[nd].keys()
            } for nd in dprej.keys()
        }
    else:
        dpre = np.array([[dprej[nd][pts] for pts in dprej[nd].keys()] for nd in dprej.keys()])
    return dpre


def plot_roll_ev(roll_ev,
                 xrange: list = None,
                 yrange: list = None,
                 title: str = None):
    if isinstance(roll_ev, dict):
        ev_array = np.array([[roll_ev[nd][pts] for pts in roll_ev[nd].keys()] for nd in roll_ev.keys()])
    else:
        ev_array = roll_ev

    num_dice = ev_array.shape[0]
    max_points = ev_array.shape[1] * 50

    dice_range = range(1, num_dice+1)
    points_range = range(0, max_points, 50)
    # assume top 20 * 50 points should be cutoff
    upper_pt_range = -20

    if xrange is None: xrange = [0, max_points]
    if yrange is None: yrange = [-1 * max_points, max_points]

    for nd in dice_range:
        E_points = []
        x = []
        for pt in points_range[:upper_pt_range]:
            if xrange[0] <= pt <= xrange[1]:
                ev = ev_array[nd-1][pt // 50]
                ev = ev if yrange[0] <= ev <= yrange[1] else None
                E_points.append(ev)
                x.append(pt)
        plt.plot(x, E_points, label=f'{nd}')

    if title is None:
        plt.title('EV of farkle re-roll for given points and num dice')
    else:
        plt.title(title)
    plt.grid()
    plt.legend(title='Dice to roll')
    plt.ylabel('EV if re-roll')
    plt.xlabel('Current Points')
    plt.show()
