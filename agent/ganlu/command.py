import os

import torch
import torch.nn as nn

from agent.ganlu.dqn import DQN
from agent.ganlu.cnn import BlockNet
from util.ganlu_util import *
from util.other_util import DEVICE

class Commander:
    def __init__(self, name):
        # 输入：己方统计、敌方统计、当前块，输出：目标块
        torch.cuda.set_device(DEVICE)
        self.name = name
        self.common = BlockNet(in_channel=4, out_channels=COMMON_OUT_CHANNELS, kernel_sizes=COMMON_OUT_KERNEL_SIZES)
        self.dqnd = DQN(name+'-dqnd', in_channel=COMMON_OUT_CHANNELS[-1] + 1, preprocess_net=self.common)
        self.dqnf = DQN(name+'-dqnf', in_channel=COMMON_OUT_CHANNELS[-1] + 1, preprocess_net=self.common)
        self.load_model()
        
    def save_model(self):
        if not os.path.exists(MODEL_FOLDER):
            os.mkdir(MODEL_FOLDER)
        common_net_folder = os.path.join(MODEL_FOLDER, 'common')
        if not os.path.exists(common_net_folder):
            os.mkdir(common_net_folder)
        torch.save(self.common.state_dict(), os.path.join(common_net_folder, self.name))
        self.dqnd.save_model()
        self.dqnf.save_model()

    def load_model(self):
        common_net_path = os.path.join(MODEL_FOLDER, 'common', self.name)
        if os.path.exists(common_net_path):
            self.common.load_state_dict(torch.load(common_net_path))
        self.common.to(DEVICE)
        self.dqnd.load_model()
        self.dqnf.load_model()

    def act(self, obs_map, alive_friend_maps, alive_friend_ids): # 每个env_step都会调用
        action_maps = []
        action_block_indexes = []
        if self.step % INTERVAL == 0: # 每INTERVAL更换一次action
            if MODIFY_MAP:
                modified_map = self.get_modified_map(obs_map)
            else:
                modified_map = obs_map
            for i, friend_map in enumerate(alive_friend_maps):
                state = torch.Tensor(np.concatenate([
                    modified_map, np.expand_dims(friend_map, axis=0)])).unsqueeze_(0).cuda()
                if alive_friend_ids[i] in [1, 2]:
                    q_map = self.dqnd(state)[0, 0].detach().cpu().numpy()
                else:
                    q_map = self.dqnf(state)[0, 0].detach().cpu().numpy()
                action_map, action_block_index = q_to_action(q_map)
                action_maps.append(action_map)
                action_block_indexes.append(action_block_index)
                self.action_buffer[alive_friend_ids[i]] = (action_map, action_block_index)
        else:
            for alive_friend_id in alive_friend_ids:
                action_map, action_block_index = self.action_buffer[alive_friend_id]
                action_maps.append(action_map)
                action_block_indexes.append(action_block_index)

        action_maps = np.array(action_maps)
        action_block_indexes = np.array(action_block_indexes)
        self.step += 1
        return action_maps, action_block_indexes

    def get_modified_map(self, obs_map):                                                                          # (4, D, D)
        origin_enemy_detectors_map = obs_map[2]
        origin_enemy_fighters_map = obs_map[3]
        appeared_enemy_detector_num = np.sum(origin_enemy_detectors_map, dtype=int)
        appeared_enemy_fighter_num = np.sum(origin_enemy_fighters_map, dtype=int)
        remained_enemy_detector_num = self.enemy_detector_num - appeared_enemy_detector_num
        remained_enemy_fighter_num = self.enemy_fighter_num - appeared_enemy_fighter_num
        predicted_enemy_detectors_map = np.reshape(origin_enemy_detectors_map, (-1,))
        predicted_enemy_fighters_map = np.reshape(origin_enemy_fighters_map, (-1,))
        for _ in range(remained_enemy_detector_num):
            predicted_enemy_detectors_map[random.randint(0, DIVIDE * DIVIDE - 1)] += 1
        for _ in range(remained_enemy_fighter_num):
            predicted_enemy_fighters_map[random.randint(0, DIVIDE * DIVIDE - 1)] += 1
        predicted_enemy_detectors_map = np.reshape(predicted_enemy_detectors_map, origin_enemy_detectors_map.shape)
        predicted_enemy_fighters_map = np.reshape(predicted_enemy_fighters_map, origin_enemy_fighters_map.shape)
        modified_map = obs_map.copy()
        modified_map[2] = predicted_enemy_detectors_map
        modified_map[3] = predicted_enemy_fighters_map
        return modified_map

    def reset(self):
        self.step = 0
        self.enemy_fighter_num = 10
        self.enemy_detector_num = 2
        self.action_buffer = {}

    def train(self):
        self.common.train()
        self.dqnd.train()
        self.dqnf.train()
    
    def eval(self):
        self.common.eval()
        self.dqnd.eval()
        self.dqnf.eval()

    # 原本是probability，后来用了mean_square训练，就不再是probability了
    def get_predicted_enemys_map(self, origin_map, probability_map):
        origin_detector_map = origin_map[0].copy()
        origin_fighter_map = origin_map[1].copy()
        appeared_enemy_detector_num = np.sum(origin_detector_map)
        appeared_enemy_fighter_num = np.sum(origin_fighter_map)
        remained_enemy_detector_num = self.enemy_detector_num - appeared_enemy_detector_num
        remained_enemy_fighter_num = self.enemy_fighter_num - appeared_enemy_fighter_num

        flattened_origin_detector_map = np.reshape(origin_detector_map, (-1,))
        flattened_origin_fighter_map = np.reshape(origin_fighter_map, (-1,))
        flattened_probability_detector_map = np.reshape(probability_map[0], (-1,))
        flattened_probability_fighter_map = np.reshape(probability_map[1], (-1,))

        sorted_detector_indexes = np.argsort(flattened_probability_detector_map)[::-1]
        sorted_fighter_indexes = np.argsort(flattened_probability_fighter_map)[::-1]

        for index in sorted_detector_indexes:
            if remained_enemy_detector_num == 0:
                break
            if flattened_origin_detector_map[index] == 0:
                flattened_origin_detector_map[index] += 1
                remained_enemy_detector_num -= 1

        for index in sorted_fighter_indexes:
            if remained_enemy_fighter_num == 0:
                break
            if flattened_origin_fighter_map[index] == 0:
                flattened_origin_fighter_map[index] += 1
                remained_enemy_fighter_num -= 1

        return np.stack([flattened_origin_detector_map.reshape(origin_detector_map.shape),
            flattened_origin_fighter_map.reshape(origin_fighter_map.shape)])