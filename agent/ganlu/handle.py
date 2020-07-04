import random

import numpy as np

from util.other_util import ID, POS_X, POS_Y, HIT_TARGET
from util.env_util import get_old_strikes

class InfsHandler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.enemy_coordinate_dict = {}                                 # { enemy_id: (id, x, y) }
        self.last_strikes = np.zeros((0, 3), dtype=int)

    def update_coordinates(self, enemy_infs):
        for enemy_inf in enemy_infs:                                        # (id, x, y)
            self.enemy_coordinate_dict[enemy_inf[ID]] = enemy_inf

    def update_coordinates_from_die_friend(self, die_coordinates):
        for die_coordinate in die_coordinates:
            ran = random.randint(3, 12)
            self.enemy_coordinate_dict[ran] = (ran, die_coordinate[0], die_coordinate[1])

    def remove_enemy_coordinates(self, ids):
        for enemy_id in ids:
            try: self.enemy_coordinate_dict.pop(enemy_id)
            except: pass

    def get_enemy_infs(self): # [ (id, x, y) ]
        return np.array(list(self.enemy_coordinate_dict.values())).reshape((-1, 3))

    def disappeared_strikes(self, now_strikes):
        if len(self.last_strikes == 0): return self.last_strikes
        old_strikes = get_old_strikes(now_strikes)
        disappeared_indexes = []
        for index, strike in enumerate(self.last_strikes):
            exist = (strike == old_strikes).all(axis=1).any()
            if not exist: disappeared_indexes.append(index)
        disappeared_strikes = self.last_strikes[disappeared_indexes]
        return disappeared_strikes

    # strike (att_id, tar_id, step)
    def destoryed_enemys(self, disappeared_strikes, destorys):
        if len(disappeared_strikes) == 0: return np.array([])
        indexes = np.where(destorys != 0)[0]
        destroy_ids = indexes + 1
        destroyed_enemy_ids = []
        for disappeared_strike in disappeared_strikes:
            if disappeared_strike[0] in destroy_ids:
                destroyed_enemy_ids.append(disappeared_strike[1])
        destroyed_enemy_ids = np.unique(np.array(destroyed_enemy_ids))
        return destroyed_enemy_ids

    # 每个时刻都会调用一次
    def get_destroyed_enemy_ids(self, now_strikes, destroys):
        disappeared_strikes = self.disappeared_strikes(now_strikes)
        destroyed_enemy_ids = self.destoryed_enemys(disappeared_strikes, destroys)
        self.last_strikes = now_strikes
        return destroyed_enemy_ids