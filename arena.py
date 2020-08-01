"""
@author: Xuerun Chen
@contact: chenxuerun18@nudt.edu.cn
@software: vscode
@time: 2020/5/27 15:52
"""

import importlib
import os
import time
import random
from interface import Environment

import numpy as np

from train.ganlu.train_agent import train_ganlu
from util.env_util import *

DEBUG = False

if __name__ == '__main__':
    num_round = 10
    num_step = 2000
    render = True
    train = True
    map_path = 'maps/' + '1000_1000_fighter10v10' + '.map'

    # alert_ranges = [375, 400, 410, 420, 425, 450, 500]
    alert_ranges = [375, 400, 425]
    agent_types = ['ganlu' for _ in alert_ranges]
    agent_names = [str(x) for x in alert_ranges]

    if not DEBUG:
        agent_types.append('fix_rule')   # 5
        agent_names.append('fix_rule')
        alert_ranges.append(-1)

        # agent_types.append('MidSolo')    # 6
        # agent_names.append('MidSolo')
        # alert_ranges.append(-1)

        # agent_types.append('HMSL_PFR')    # 7
        # agent_names.append('HMSL_PFR')
        # alert_ranges.append(-1)

    agents = []

    for i, agent_type in enumerate(agent_types):
        agent_import_path = 'agent.' + agent_types[i] + '.agent'
        agent_module = importlib.import_module(agent_import_path)
        try: agent = agent_module.Agent(name=agent_names[i], alert_range=alert_ranges[i])
        except: 
            if DEBUG: raise
            agent = agent_module.Agent()
        agents.append(agent)
    
    env = Environment(map_path,  'raw',  'raw',
                    max_step=num_step, render=render, random_pos=False, log=False)
    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()
    for agent in agents:
        agent.set_map_info(size_x, size_y, side1_detector_num, side1_fighter_num)
    agents_num = len(agents)

    prob_last = 0.4;    prob_other = (1 - prob_last) / (agents_num - 1)
    prob = prob_other * np.ones((agents_num,)); prob[-1] = prob_last

    total_round_num = np.ones((agents_num,))
    total_win_num = np.zeros((agents_num,))

    vs = []
    for i in range(agents_num - 1):
        for j in range(i+1, agents_num):
            vs.append((i, j))
            vs.append((j, i))
    len_vs = len(vs)

    vs_index = 0
    vs_num = 0
    while(True):
        # choosed_agent_indexes = np.random.choice(a=range(agents_num), size=2, replace=False, p=prob)
        choosed_agent_indexes = [1, 2]
        SIDE1_INDEX = choosed_agent_indexes[0]
        SIDE2_INDEX = choosed_agent_indexes[1]
        # SIDE1_INDEX = vs[vs_index][0]
        # SIDE2_INDEX = vs[vs_index][1]
        agent1 = agents[SIDE1_INDEX]
        agent2 = agents[SIDE2_INDEX]
        print(agent_names[SIDE1_INDEX], ' vs ', agent_names[SIDE2_INDEX])

        for x in range(num_round):
            env.reset()
            step_cnt = 0

            try: agent1.reset(record=train)
            except: 
                if DEBUG: raise
            try: agent2.reset(record=train)
            except: 
                if DEBUG: raise

            side1_obs_dict, side2_obs_dict = env.get_obs()
            while True:
                # time.sleep(0.02)
                step_cnt += 1

                try: side1_detector_action, side1_fighter_action = agent1.get_action(side1_obs_dict, step_cnt, enemy_obs_dict=side2_obs_dict)
                except: 
                    if DEBUG: raise
                    side1_detector_action, side1_fighter_action = agent1.get_action(side1_obs_dict, step_cnt)
                try: side2_detector_action, side2_fighter_action = agent2.get_action(side2_obs_dict, step_cnt, enemy_obs_dict=side1_obs_dict)
                except: 
                    if DEBUG: raise
                    side2_detector_action, side2_fighter_action = agent2.get_action(side2_obs_dict, step_cnt)

                env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)
                side1_obs_dict, side2_obs_dict = env.get_obs()

                if env.get_done():
                    env_reward = env.get_reward()
                    agent1_reward = env_reward[2]
                    agent2_reward = env_reward[5]
                    total_round_num[SIDE1_INDEX] += 1;  total_round_num[SIDE2_INDEX] += 1
                    if agent1_reward > 0: total_win_num[SIDE1_INDEX] += 1
                    if agent2_reward > 0: total_win_num[SIDE2_INDEX] += 1
                    try: agent1.get_action(side1_obs_dict, step_cnt, enemy_obs_dict=side2_obs_dict, game_reward=agent1_reward)
                    except: 
                        if DEBUG: raise
                    try: agent2.get_action(side2_obs_dict, step_cnt, enemy_obs_dict=side1_obs_dict, game_reward=agent2_reward)
                    except: 
                        if DEBUG: raise
                    break

            try: train_ganlu(agent1)
            except: 
                if DEBUG: raise
            try: train_ganlu(agent2)
            except: 
                if DEBUG: raise

        vs_index = (vs_index + 1) % len_vs
        vs_num += 1
        if vs_num % 10 == 0:
            print(f'win_rate: {total_win_num / total_round_num}')

        if train:
            try: agent1.commander.save_model()
            except:
                if DEBUG: raise
            try: agent2.commander.save_model()
            except:
                if DEBUG: raise