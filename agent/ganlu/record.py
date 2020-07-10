import numpy as np

from util.env_util import ID, POS_X, POS_Y, LAST_REWARD, get_alive_inf
from util.ganlu_util import INTERVAL, RL_GAMMA
from util.other_util import TOTAL_UNIT_NUM

class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):                                                        #保存时都是用最简形式
        self.step = 0
        self.game_reward = 0
        self.last_alive_ids = np.array([])
        self.accumulate_rewards = np.zeros((TOTAL_UNIT_NUM,))
        self.obs_maps = []                                     #  [ (d, d) ]
        self.respect_states = {}                                  # { plane_id: [(d, d)] }
        self.respect_actions = {}                               # { plane_id: [(d, d)]}
        self.respect_rewards = {}                             # { plane_id: [float] }
        
        for i in range(TOTAL_UNIT_NUM):
            self.respect_states[i+1] = []
            self.respect_actions[i+1] = []
            self.respect_rewards[i+1] = []

    def record_to_dqn_training_data(self, plane_id):                        # 用真实的敌方信息训练
        r = self.respect_rewards[plane_id]                                                  # (n,)

        len_traj = len(r)
        obs_maps = self.obs_maps[0: len_traj]                                        # (n, 4, d, d)
        # enemy_maps = self.true_enemys_maps[0: len_traj]              # (n, d, d)
        respect_states = self.respect_states[plane_id]                          # (n, d, d)

        s = np.concatenate([
            obs_maps, np.expand_dims(respect_states, axis=1)], axis=1)     # (n, 5, d, d)

        a = np.expand_dims(self.respect_actions[plane_id], axis=1)      # (n, 1, d, d)

        global_rewards = self.global_rewards[0: len_traj]
        r += global_rewards

        s_prime = np.concatenate([s[1:], s[0][None]])                             # (n, 5, d, d)

        is_last = np.zeros_like(r, dtype=int)
        is_last[-1] = 1

        return s, a, r, s_prime, is_last

    def post_process_record(self):
        self.obs_maps = np.array(self.obs_maps)
        len_total_traj = len(self.obs_maps)

        # 把全局回报加到每一步上
        global_rewards = np.zeros((len_total_traj,))
        global_rewards[-1] = self.game_reward
        for i in reversed(range(0, len_total_traj - 1)):
            global_rewards[i] = global_rewards[i+1] * RL_GAMMA
        self.global_rewards = global_rewards
        
        for i in range(1, 11):
            self.respect_states[i] = np.array(self.respect_states[i])
            self.respect_actions[i] = np.array(self.respect_actions[i])
            self.respect_rewards[i] = np.array(self.respect_rewards[i])

    # rewards已经移位
    def add_record(self, obs_map, alive_friend_ids, 
        alive_friend_states, alive_friend_actions, rewards, game_reward=None):
        self.accumulate_rewards += rewards

        if game_reward is not None:
            self.game_reward = game_reward
            for friend_id in self.last_alive_ids:
                friend_index = friend_id - 1
                self.respect_rewards[friend_id].append(self.accumulate_rewards[friend_index])
            return

        if (self.step % INTERVAL == 0):
            self.obs_maps.append(obs_map)

            for i, alive_plane_id in enumerate(alive_friend_ids):
                self.respect_states[alive_plane_id].append(alive_friend_states[i])
                self.respect_actions[alive_plane_id].append(alive_friend_actions[i])

            for friend_id in self.last_alive_ids:
                friend_index = friend_id - 1
                self.respect_rewards[friend_id].append(self.accumulate_rewards[friend_index])

            self.accumulate_reward = np.zeros((TOTAL_UNIT_NUM,))
            self.last_alive_ids = alive_friend_ids

        self.step += 1