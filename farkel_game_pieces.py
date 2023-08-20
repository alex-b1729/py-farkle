import copyimport randomimport loggingfrom typing import NamedTuplefrom dataclasses import dataclassclass Die():    def __init__(self, value: int = None):        if value is not None:            assert isinstance(value, int)            assert value in range(1, 7)            self.value = value        else:             self.value = random.randint(1, 6)            def roll(self):        self.value = random.randint(1, 6)            def __repr__(self):        return str(self.value)        def __eq__(self, other):        return self.value == other.value    @dataclassclass GameDie:    '''Die used in Fakel game'''    die: Die    locked: bool = False    class DiceHand():    def __init__(self, num_die: int = 6):        self.num_die = num_die        self.all_dice = {i:GameDie(Die()) for i in range(self.num_die)}        self.score = 0            def roll_all_dice(self):        for i, d in self.all_dice.items(): d.die.roll()            def roll(self):        '''roll non-locked dice'''        for i, d in self.all_dice.items():             if not d.locked:                d.die.roll()            def lock_dice(self, *args):        """Locks dice at given indexes"""        for i in args:            self.all_dice[i].locked = True                def lock_from_dicehand(self, other):        """Lock dice corresponding to unlocked dice from other        All dice values in other must be in and unlocked in self"""        for i, d in other.free_dice.items():            found = False            i = 0            while not found and i < self.num_die:                d_s = self.all_dice[i]                if not d_s.locked and d.die == d_s.die:                    d_s.locked = True                    found = True                i += 1            if not found:  # if die in other not found in self raise error                raise ValueError('Die in other not in and unlocked in self')                @property    def free_dice(self):        return {i:d for i, d in self.all_dice.items() if not d.locked}        @property    def locked_dice(self):        return {i:d for i, d in self.all_dice.items() if d.locked}        @staticmethod    def with_dice_values(*args):        dh = DiceHand(num_die=len(args))        dh.all_dice = {i: GameDie(Die(args[i])) for i in range(len(args))}        return dh            def __repr__(self):        s = ''        for i, d in self.free_dice.items():            s += f'{i}: {d.die}\n'        for i, d in self.locked_dice.items():            s += f'{i}: {d.die} - Locked\n'        return s                def __contains__(self, key):        '''checks if a dicehand combination included in self'''        is_contained = False        if isinstance(key, DiceHand):            dice_self = self.copy()  # without will modify and lock self            dice_compare: DiceHand = copy.deepcopy(key)            didnt_find = False            d_count = 0            while not didnt_find and d_count < dice_compare.num_die:                if not dice_compare.all_dice[d_count].locked:                    die_compare: Die = dice_compare.all_dice[d_count].die                    found_in = False                    i = 0                    while not found_in and i < dice_self.num_die:                        d = dice_self.all_dice[i]                        found_in = die_compare == d.die and not d.locked                        i += 1                        # if the comparison dice found in hand then lock                         # so can no longer compare to that die                        if found_in: d.locked = True                    # if found_in still False                    # means didn't find the dice (unlocked) in hand                    if not found_in: didnt_find = True                    d_count += 1            # if didnt_find still false            # means found all comparison dice in self            if not didnt_find: is_contained = True        return is_contained        def copy(self):        return copy.deepcopy(self)        class Player():    def __init__(self, name: str):        self.name = name        self.dice_hand = DiceHand()        self.score = 0                class ScoringCombo(NamedTuple):    dice: DiceHand    score: int    # TODO: Add ability to define scorring patterns such as 3 pairs    SCORING_HANDS = [        ScoringCombo(DiceHand().with_dice_values(1), 100)    ,ScoringCombo(DiceHand().with_dice_values(5), 50)        ,ScoringCombo(DiceHand().with_dice_values(1, 1, 1), 1000)    ,ScoringCombo(DiceHand().with_dice_values(2, 2, 2), 200)    ,ScoringCombo(DiceHand().with_dice_values(3, 3, 3), 300)    ,ScoringCombo(DiceHand().with_dice_values(4, 4, 4), 400)    ,ScoringCombo(DiceHand().with_dice_values(5, 5, 5), 500)    ,ScoringCombo(DiceHand().with_dice_values(6, 6, 6), 600)        ,ScoringCombo(DiceHand().with_dice_values(1, 2, 3, 4, 5, 6), 1500)        ]        class ScoreKeeper():    def __init__(self):        pass        def possible_scores(DiceHand):    pos_scores = {}        for start_i in range(len(SCORING_HANDS)):        # start with this scoring hand                 # itterate through the rest of scoring hands        total_score = 0        dh = DiceHand.copy()        for i in range(len(SCORING_HANDS)):            sh = SCORING_HANDS[(start_i + i) % len(SCORING_HANDS)]            if sh.dice in dh:                print(f'{sh.dice} Score: {sh.score}')                total_score += sh.score                dh.lock_from_dicehand(sh.dice)        print(dh)                if __name__ == '__main__':    # d = Die()    # d.roll()    # print(d)        # dh1 = DiceHand()    # dh1.roll_all_dice()    # dh2 = copy.deepcopy(dh1)    # print(dh1)    # print(dh2)        # # returns false?     # print(dh2 in dh1)        # dh1 = DiceHand().with_dice_values(5)    # print(dh1)    # dh2 = DiceHand()    # dh2.lock_dice(0,1,2)    # print(dh2)    # print(dh1 in dh2)        dh1 = DiceHand().with_dice_values(1,1,1,1,2,3)    dh1.lock_dice(4)    print(dh1, '\n')    dh2 = DiceHand().with_dice_values(1,2)    dh1.lock_from_dicehand(dh2)    print(dh1)        # possible_scores(dh1)        # for s in SCORING_HANDS:    #     # print(s)    #     if s.dice in dh1:    #         print('yes\n', s.dice, 'Points: ', s.score, end='\n')