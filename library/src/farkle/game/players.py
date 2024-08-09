import os
import abc
import time
import math
import random
import matplotlib
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from farkle import utils, models
from farkle.logic.gameobjects import GameState
from farkle.logic.gameobjects import DiceHand, Turn, RollDecision

# set up matplotlib
IS_PYTHON = 'inline' in matplotlib.get_backend()
if IS_PYTHON:
    from IPython import display
plt.ion()


class Player(metaclass=abc.ABCMeta):
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0
        # agent won't roll again if their score exceeds the goal_score
        self.goal_score = 1e12

    def __repr__(self):
        return f'{self.name}: {self.score} points'

    @abc.abstractmethod
    def play_dicehand(self, dh: DiceHand, verbose: bool = False) -> tuple[DiceHand, bool]:
        """Handles player's rolling and scoring decisions and returns chosen ps and roll again decision"""


class RandomPlayer(Player):
    def __init__(self, name: str):
        """This robot's dumb and makes random decisions"""
        super().__init__(name=name)
        self.roll_again_likelihood = 0.5

    def play_dicehand(self, dh: DiceHand) -> tuple[DiceHand, bool]:
        ps: DiceHand = random.choice(dh.possible_scores())
        return ps, random.random() < self.roll_again_likelihood


class EVMaximizingPlayer(Player):
    def __init__(self, name: str):
        """plays to maximize EV of each roll ignoring any other players"""
        super().__init__(name=name)

        # dict of EV if roll again for roll_evs[num_dice][current_points]
        self.roll_evs = None
        # highest point with an EV in roll_evs
        self.max_idxd_pt = None

    def load_roll_evs(self, path: str):
        self.roll_evs = utils.load_roll_ev(path)
        self.max_idxd_pt = max(self.roll_evs[1].keys())

    def play_dicehand(self, dh: DiceHand, verbose: bool = False) -> tuple[DiceHand, bool]:
        assert self.roll_evs is not None
        pss = dh.possible_scores()
        evs = []
        evs_given_roll_decision = []
        if verbose: print(f'Player {self.name} playing {dh}')
        for ps in pss:
            nd = utils.dice_remaining_convert(len(dh.free_dice) - ps.num_dice)
            pts = dh.score + ps.score
            if verbose:
                print(f'Considering possible score: {ps}')
                print(f'\tLeaves {nd} dice remaining and {pts} points')
            if pts <= self.max_idxd_pt:
                if verbose: print(f'\tEV of re-roll: {self.roll_evs[nd][pts]}')
                evs.append(self.roll_evs[nd][pts])
                if pts < self.goal_score:
                    evs_given_roll_decision.append(max(pts + self.roll_evs[nd][pts], pts))
                else:
                    evs_given_roll_decision.append(pts)
            else:
                # I've estimated EVs up to where its -EV for all nd, pts
                # so if pts is too large I know I can treat it as -EV
                if verbose: print(f'Do not have EV estimate so assuming it\'s negative')
                evs.append(-1)
                evs_given_roll_decision.append(pts)
        # chose greatest score considering not rolling if ev < 0
        evs_array = np.array(evs_given_roll_decision)
        choice_idx = np.argmax(evs_array)
        if verbose:
            print(f'Choosing {pss[choice_idx]} since maximizes score of {evs_given_roll_decision[choice_idx]}')
        # if score exceeds the goal score agent does not roll again
        if evs_given_roll_decision[choice_idx] < self.goal_score:
            roll_again = evs[choice_idx] > 0
            if verbose:
                print(f"{'R' if roll_again else 'Not r'}olling again since expected points on re-roll: "
                      f"{round(evs[choice_idx], 2)} {'>' if roll_again else '<='} 0")
        else:
            roll_again = False
            if verbose: print(f'Not rolling again since agent has reached the goal score')
        return pss[choice_idx], roll_again


class DQNAgent(Player):
    """
    Haven't had luck getting this DQN to converge
    """
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = models.DQN().to(self.device)
        self.target_net = models.DQN().to(self.device)

        # training
        self.training = False

        self.roll_again_likelihood = 0.7

        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        
        self.tau = 0.005
        self.lr = 1e-4
        self.weight_update_freq = 1000

        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.lr,
            amsgrad=True
        )
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.lr)

        self.memory_size = 300
        self.memory = models.ReplayMemory(self.memory_size)
        self.batch_size = 200
        self.turns_complete = 0
        self.episode_scores = []

        self.verbose = False
        self.show_plot = False
        self.plot_freq = 1_000

    def eps_threshold(self):
        """exponentially decays from eps_start to esp_end as turns_complete increases"""
        return (self.eps_end + (self.eps_start - self.eps_end) *
                math.exp(-1. * self.turns_complete / self.eps_decay))

    def play_dicehand(self, dh: DiceHand) -> tuple[DiceHand, bool]:
        """returns chosen ps and roll again decision"""
        ps_list = dh.possible_scores()
        states = torch.tensor([[int(dh.score + ps.score),
                                utils.dice_remaining_convert(len(dh.dice_values_free()) - ps.num_dice)]
                               for ps in ps_list],
                              device=self.device, dtype=torch.float32)
        state_scores = torch.tensor([[int(dh.score + ps.score)] for ps in ps_list])
        E_add_scores = self.expected_additional_score(states)
        E_scores = torch.where(E_add_scores > 0, E_add_scores, 0.0) + state_scores

        if self.verbose:
            print(f'states\n{states}')
            print(f'state_scores\n{state_scores}')
            print(f'E_add_scores\n{E_add_scores}')
            print(f'E_scores\n{E_scores}')

        if self.training and random.random() < self.eps_threshold():
            # chose random move
            if self.verbose: print('chose random ps')
            random_idx = random.choice(range(len(ps_list)))
            return ps_list[random_idx], random.random() < self.roll_again_likelihood
        else:
            # use model
            if self.verbose: print('chose model ps')
            max_score_idx = E_scores.argmax()
            chosen_ps: DiceHand = ps_list[max_score_idx]
            roll_again = bool(E_add_scores[max_score_idx] > 0)
            return chosen_ps, roll_again

    def expected_additional_score(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(states)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = models.Transition(*zip(*transitions))
        if self.verbose: print(f'optimizing with batch: \n{batch}')

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )

        state_batch = torch.tensor(
            [[s.score, s.num_dice_remaining] for s in batch.state],
            device=self.device,
            dtype=torch.float32
        )
        predicted_additional_scores = self.policy_net(state_batch)

        next_state_batch = torch.tensor(
            [[s.score, s.num_dice_remaining] if s is not None else [0, 0]
             for s in batch.next_state],
            device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            future_reward_estimates = self.target_net(next_state_batch)

        reward_batch = torch.tensor(
            [[r] for r in batch.reward],
            device=self.device,
            dtype=torch.float32
        )
        targeted_additional_scores = reward_batch.detach().clone()
        targeted_additional_scores[non_final_mask] += future_reward_estimates[non_final_mask]

        loss = self.loss_func(predicted_additional_scores, targeted_additional_scores)

        if self.verbose:
            print(f'state_batch\n{state_batch}')
            print(f'predicted_additional_scores\n{predicted_additional_scores}')

            print(f'next_state_batch\n{next_state_batch}')
            print(f'future_reward_estimates\n{future_reward_estimates}')

            print(f'reward_batch\n{reward_batch}')
            print(f'targeted_additional_scores\n{targeted_additional_scores}')

            print(f'loss: {loss}')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot(self, show_result=False, freq: int = 1, avg_freq: int = 1000):
        """plots score of dice turns in training"""
        if self.turns_complete % freq != 0:
            return
        plt.figure(1)
        points = torch.tensor(self.episode_scores, dtype=torch.float32)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(points.numpy())
        # Take episode averages and plot them too
        if len(points) >= avg_freq:
            means = points.unfold(0, avg_freq, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg_freq - 1), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if IS_PYTHON:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def play_turn(self) -> int:
        """plays hand according to current policy and returns turn's score"""
        dh = DiceHand()  # init as rolled dh
        will_roll_again = True
        score_post = dh.score
        state_post = models.TrainingState(
            score=score_post,
            num_dice_remaining=utils.dice_remaining_convert(len(dh.free_dice))
        )

        if self.verbose: print(dh)

        while not dh.farkled and will_roll_again:
            # capture initial dh state
            state_pre = state_post

            # choose action if not farkled
            # but b/c we're in this loop we've already not farkled
            # policy_net makes decision
            chosen_ps, will_roll_again = self.play_dicehand(dh)
            # execute decision
            dh.lock_from_dicehand(chosen_ps)
            # save immediate reward of decision
            reward = chosen_ps.score

            score_post = dh.score
            num_dice_remaining_post = utils.dice_remaining_convert(len(dh.free_dice))
            state_post = models.TrainingState(
                score=score_post,
                num_dice_remaining=num_dice_remaining_post
            )

            if self.verbose:
                print(f'state pre: {state_pre}')
                print('Chosen ps:', chosen_ps)
                print('Will roll again:', will_roll_again)
                print('reward:', reward)
                print('state post', state_post)

            # save turn transition
            self.memory.push(
                models.Transition(
                    state=state_pre,
                    next_state=state_post,
                    reward=reward
                )
            )

            if will_roll_again:
                if dh.all_locked:
                    dh.roll_all_dice()
                else:
                    dh.roll()
                if self.verbose: print(f'roll again result: {dh}')

        # finally if ended turn by farkle
        # `and will_roll_again` necessary since locking dice then choosing to not
        # roll again can cause dh.farkled == True
        turn_score = 0
        if dh.farkled and will_roll_again:
            if self.verbose: print(f'farkled - reward: {-1 * score_post}')
            # capture final transition to farkle state
            state_pre = state_post
            state_post = None
            reward = -1 * score_post
            self.memory.push(
                models.Transition(
                    state=state_pre,
                    next_state=state_post,
                    reward=reward
                )
            )
        else:
            turn_score = dh.score

        return turn_score
        
    def train(self, num_turns: int):
        self.training = True
        self.turns_complete = 0

        self.target_net.load_state_dict(self.policy_net.state_dict())

        t0 = perf_counter()
        for turn_idx in range(num_turns):
            if self.verbose: print(f'\n----- turn {turn_idx} -----')

            turn_score = self.play_turn()

            # train
            self.optimize_model()

            self.turns_complete += 1

            # soft target net weight update
            # target_net_state_dict = self.target_net.state_dict()
            # policy_net_state_dict = self.policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau +
            #                                   target_net_state_dict[key] * (1 - self.tau))
            # self.target_net.load_state_dict(target_net_state_dict)

            # hard target net weight update every weight_update_freq turns
            if self.turns_complete % self.weight_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # plot
            self.episode_scores.append(turn_score)
            if self.show_plot:
                self.plot(freq=self.plot_freq)

        if self.verbose:
            print(self.memory)
            print(f'episode_scores: {self.episode_scores}')

        t1 = perf_counter()

        print('Training complete')
        print(f'Time: {round((t1-t0) / 60, 2)} min')

        if self.show_plot:
            self.plot(show_result=True)
            plt.ioff()
            plt.show()


class HumanPlayer(Player):
    def __int__(self, name: str):
        super().__init__(name)

    def play_dicehand(self, dice_hand, game_state: GameState) -> RollDecision:
        select_ps_index = int(input('Index of possible score to play: ')) - 1
        score_decision: DiceHand = dice_hand.possible_scores()[select_ps_index]
        select_roll_again = input('Do you want to roll again? [y/n] ')
        will_roll_again = select_roll_again.lower() == 'y'
        post_dicehand = dice_hand.copy()
        post_dicehand.lock_from_dicehand(score_decision)
        return RollDecision(dice_hand, post_dicehand, will_roll_again)


if __name__ == '__main__':
    agent = EVMaximizingPlayer('agent007')
    agent.goal_score = 5_000
    agent.load_roll_evs('/Users/abrefeld/ab/Scripts/py-farkle/models/roll_EV_12000.json')
    dh = DiceHand('522234')
    dh.score = 1000
    print(agent.play_dicehand(dh, verbose=True))
