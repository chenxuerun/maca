import random

import torch
import torch.nn as nn

from agent.ganlu.handle import InfsHandler
from agent.base_agent import BaseAgent
from agent.ganlu.record import Recorder
from agent.ganlu.dqn import DQN
from agent.ganlu.command import Commander
from util.env_util import *
from util.dl_util import *
from util.ganlu_util import *

'''
    智能体内部有一个维护敌方信息的仓库r
    智能体每时刻收到的敌方信息为s1，
    更新仓库后返回的综合敌方信息为 s2 = f1(r, s1)，
    随机添加后，获得的预测敌方信息为 s3 = f2(s2)
'''

class Agent(BaseAgent):
    def __init__(self, name, alert_range=180):
        self.name = name
        self.alert_range = alert_range
        BaseAgent.__init__(self)
        self.recorder = Recorder()
        self.handler = InfsHandler()
        self.commander = Commander(name)

    def reset(self, record=True):
        self.record = record
        self.handler.reset()
        self.recorder.reset()
        self.commander.reset()
        self.commander.eval()

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    # 每个时刻都要追踪所有敌机的位置，用于判断是否进攻，
    # 每INTERVAL个时刻Commander调用一次网络，更新目标位置。
    def get_action(self, obs_dict, step_cnt, **kwargs):
        friend_detector_infs, friend_fighter_infs, enemy_infs = get_inf_from_one_dict(obs_dict)
        # (id, x, y, alive, last_reward)
        friend_infs = np.concatenate([friend_detector_infs, friend_fighter_infs[:, 0:5]])
        alive_friend_detector_infs = get_alive_inf(friend_detector_infs)
        alive_friend_fighter_infs = get_alive_inf(friend_fighter_infs)
        alive_friend_detector_ids = alive_friend_detector_infs[:, ID]
        alive_friend_fighter_ids = alive_friend_fighter_infs[:, ID]
        alive_friend_ids = np.concatenate([alive_friend_detector_ids, alive_friend_fighter_ids])
        alive_friend_detector_coordinates = alive_friend_detector_infs[:, [POS_X, POS_Y]]
        alive_friend_fighter_coordinates = alive_friend_fighter_infs[:, [POS_X, POS_Y]]
        alive_friend_detector_num = len(alive_friend_detector_infs)
        alive_friend_fighter_num = len(alive_friend_fighter_infs)

        now_strikes = strike_list_to_array(obs_dict['joint_obs_dict']['strike_list'])
        enemy_striked_num = get_enemy_striked_num(now_strikes)
        enemy_infs_from_strike = get_enemy_inf_from_strike(obs_dict)
        detect_nums, dies, destroy_nums = get_inf_from_signal_rewards(friend_infs)

        # 在remove enemy之前统计，detector reward
        destroyed_enemy_ids = self.handler.get_destroyed_enemy_ids(now_strikes, destroy_nums)
        detector_assists = np.zeros((TOTAL_UNIT_NUM,)) 
        if (len(destroyed_enemy_ids) != 0)  and (len(alive_friend_detector_infs) != 0):
            last_buffered_enemy_infs = self.handler.get_enemy_infs()
            destroyed_enemy_infs = np.array(
                [inf for inf in last_buffered_enemy_infs if inf[0] in destroyed_enemy_ids]).reshape((-1, 3))
            if len(destroyed_enemy_infs) == 0:
                print(destroyed_enemy_ids)
                print(last_buffered_enemy_infs)
            destoryed_enemy_coordinates = destroyed_enemy_infs[:, [POS_X, POS_Y]]
            for detector_inf in alive_friend_detector_infs:
                difference = detector_inf[[POS_X, POS_Y]] - destoryed_enemy_coordinates
                dis = np.linalg.norm(difference, ord=2, axis=1)
                num = np.sum(dis < DETECTOR_DETECT_RANGE)
                detector_assists[detector_inf[ID] - 1] = num
        
        self.handler.remove_enemy_coordinates(destroyed_enemy_ids)
        buffered_enemy_infs = self.handler.get_enemy_infs()
        disappeared_enemy_ids = get_disappeared_enemy(
            friend_fighter_infs, buffered_enemy_infs, enemy_infs)
        self.handler.remove_enemy_coordinates(disappeared_enemy_ids)
        self.handler.update_coordinates(enemy_infs_from_strike)
        self.handler.update_coordinates(enemy_infs)

        for enemy_id in destroyed_enemy_ids:
            if enemy_id in [1, 2]:
                self.commander.enemy_detector_num -= 1
            else:
                self.commander.enemy_fighter_num -= 1
            
        new_enemy_infs = self.handler.get_enemy_infs()
        new_enemy_coordinates = new_enemy_infs[:, [POS_X, POS_Y]]
        new_enemy_detector_infs, new_enemy_fighter_infs = divide_infs(new_enemy_infs)
        new_enemy_detector_coordinates = new_enemy_detector_infs[:, [POS_X, POS_Y]]
        new_enemy_fighter_coordinates = new_enemy_fighter_infs[:, [POS_X, POS_Y]]

        # 以下全都只包括活着的数据，各数组下标第一维都是对应的。
        friend_detectors_map = to_map(alive_friend_detector_coordinates)
        friend_fighters_map = to_map(alive_friend_fighter_coordinates)
        see_enemy_detectors_map = to_map(new_enemy_detector_coordinates)
        see_enemy_fighters_map = to_map(new_enemy_fighter_coordinates)
        obs_map = np.stack([friend_detectors_map, friend_fighters_map,
            see_enemy_detectors_map, see_enemy_fighters_map])
        friend_detector_maps, _ = to_divided_maps(alive_friend_detector_coordinates)
        friend_fighter_maps, _ = to_divided_maps(alive_friend_fighter_coordinates)
        friend_maps = np.concatenate([friend_detector_maps, friend_fighter_maps])

        # 先输出一个行为，根据后面进攻的情况再改
        action_maps, action_block_indexes = self.commander.act(
            obs_map, friend_maps, alive_friend_ids)
        detector_action_block_indexes = action_block_indexes[0: alive_friend_detector_num]
        fighter_action_block_indexes = action_block_indexes[alive_friend_detector_num:]

        detector_actions = np.zeros((DETECTOR_NUM, 2), dtype=int)
        fighter_actions = np.zeros((FIGHTER_NUM, 4), dtype=int)
        alive_detector_actions = []          # [course, r]
        alive_fighter_actions = []              # [course, r, j, attack]
        detector_r = random.randint(1, 20)
        fighter_r = random.randint(1, 10)
        j = 11
        d_enemy_dis_dict = {}                   # {enemy_index: (alive_friend_index, dis)} 视野范围以内

        for i, alive_friend_detector_inf in enumerate(alive_friend_detector_infs):
            detector_coordinate = alive_friend_detector_inf[[POS_X, POS_Y]]
            detector_goal = block_min_coordinate(
                number_to_block(detector_action_block_indexes[i]))
            detector_goal[0] += random.randint(0, BLOCK_WIDTH)
            detector_goal[1] += random.randint(0, BLOCK_HEIGHT)
            detector_course = course_start_to_goal(start=detector_coordinate, goal=detector_goal)
            detector_action = np.array([detector_course, detector_r])
            alive_detector_actions.append(detector_action)
        for i, detector_action in enumerate(alive_detector_actions):
            detector_id = alive_friend_detector_infs[i, ID]
            detector_actions[detector_id - 1] = detector_action

        for i, alive_friend_fighter_inf in enumerate(alive_friend_fighter_infs):
            alive_friend_fighter_coordinate = alive_friend_fighter_inf[[POS_X, POS_Y]]
            fighter_goal = None
            attack = 0

            if len(new_enemy_coordinates) != 0: # 所有index都是围绕new_enemy_coordinates展开的
                difference = alive_friend_fighter_coordinate - new_enemy_coordinates
                enemy_distances = np.linalg.norm(difference, ord=2, axis=1)

                sorted_enemy_index = np.argsort(enemy_distances)
                sorted_enemy_distances = np.sort(enemy_distances)
                near_enemy_distances = sorted_enemy_distances[sorted_enemy_distances < self.alert_range]
                near_enemy_indexes = sorted_enemy_index[0: len(near_enemy_distances)]
                near_enemy_infs = new_enemy_infs[near_enemy_indexes]
                near_enemy_ids = near_enemy_infs[:, ID]
                near_enemy_coordinates = near_enemy_infs[:, [POS_X, POS_Y]]
            else:
                near_enemy_indexes = []

            if len(near_enemy_indexes) != 0:  # near包括ALERT_RANGE以内有敌人
                for enemy_index in near_enemy_indexes:
                    enemy_distance = enemy_distances[enemy_index]
                    if enemy_distance < FIGHTER_DETECT_RANGE:
                        if enemy_index not in d_enemy_dis_dict.keys():
                            d_enemy_dis_dict[enemy_index] = []
                        d_enemy_dis_dict[enemy_index].append((i, enemy_distance))

                for j, near_enemy_id in enumerate(near_enemy_ids):
                    if enemy_striked_num[near_enemy_id - 1] < MAX_STRIKE_NUM: # 最近的那个敌人没被进攻
                        near_enemy_distance = near_enemy_distances[j]
                        attack = try_attack(alive_friend_fighter_inf, near_enemy_id, near_enemy_distance)
                        print(attack)
                        if attack == VOID: # 身上没子弹，跑
                            attack = 0
                            fighter_goal = escape(alive_friend_fighter_inf[[POS_X, POS_Y]], near_enemy_coordinates)
                        else: # 身上有子弹，打
                            fighter_goal = near_enemy_coordinates[0]
                        break
                    elif j == len(near_enemy_ids) - 1: # 全都已经被进攻了，撤吧
                        fighter_goal = escape(alive_friend_fighter_inf[[POS_X, POS_Y]], near_enemy_coordinates)
                    else:
                        continue

            if fighter_goal is None:
                if NINE_ACTION:
                    fighter_goal = get_goal_from_action(alive_friend_fighter_coordinate, action_block_indexes[i])
                else:
                    fighter_goal = block_min_coordinate((number_to_block(action_block_indexes[i])))
                    fighter_goal[0] += random.randint(0, BLOCK_WIDTH)
                    fighter_goal[1] += random.randint(0, BLOCK_HEIGHT)
                swing = random.randint(-SWING, SWING)
            else: 
                new_action_map, _ = get_action_from_goal(fighter_goal, alive_friend_fighter_coordinate)
                action_maps[i] = new_action_map
                swing = 0

            fighter_course = course_start_to_goal(start=alive_friend_fighter_coordinate, goal=fighter_goal)
            fighter_course += swing
            fighter_action = np.array([fighter_course, fighter_r, j, attack])
            alive_fighter_actions.append(fighter_action)

        # 留一个最远的干扰（所有记录都是180以内）
        for enemy_index in list(d_enemy_dis_dict.keys()):
            enemy_dis = np.array(d_enemy_dis_dict[enemy_index]) # (alive_friend_fighter_index, dis)
            max_dis_index = np.argmax(enemy_dis[:, 1])
            alive_friend_fighter_index = enemy_dis[max_dis_index, 0].astype(int)
            alive_friend_fighter_coordinate = alive_friend_fighter_infs[alive_friend_fighter_index, [POS_X, POS_Y]]
            enemy_coordinate = new_enemy_infs[enemy_index, [POS_X, POS_Y]]
            course = course_start_to_goal(start=alive_friend_fighter_coordinate, goal=enemy_coordinate)
            alive_fighter_actions[alive_friend_fighter_index][0] = course

        # 得到所有行动
        for i, alive_fighter_action in enumerate(alive_fighter_actions):
            plane_id = alive_friend_fighter_infs[i, ID]
            fighter_index = plane_id - 1 - DETECTOR_NUM
            fighter_actions[fighter_index] = alive_fighter_action

        if self.record: # 记录 
            other_penalties = np.zeros((TOTAL_UNIT_NUM,))
            # for i, friend_map in enumerate(friend_maps):
            #     other_penalty = get_dense_penalty(friends_map, friend_map)
            #     friend_index = alive_friend_fighter_ids[i] - 1
            #     other_penalties[friend_index] = other_penalty
            alives = friend_infs[:, ALIVE]
            reward_dict = REWARD[CHOOSE_REWARD]
            rewards = reward_dict['detect_reward'] * detect_nums + \
                reward_dict['destroy_reward'] * destroy_nums + \
                    reward_dict['die_reward'] * dies + \
                        reward_dict['alive_reward'] * alives + other_penalties + \
                            reward_dict['assist_reward'] * detector_assists
            if 'game_reward' in kwargs.keys():
                game_reward = kwargs['game_reward']
            else: game_reward = None

            self.recorder.add_record(obs_map=obs_map, 
                alive_friend_ids=alive_friend_ids, 
                alive_friend_states=friend_maps,  alive_friend_actions=action_maps, 
                rewards=rewards, game_reward=game_reward)

        return detector_actions, fighter_actions