"""
@author: Xuerun Chen
@contact: chenxuerun18@nudt.edu.cn
@software: vscode
@time: 2020/5/27 15:52
"""

import argparse
import importlib
import os
import time
import numpy as np
import datetime
from interface import Environment

from train.ganlu.train_agent import train_ganlu
from util.env_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='1000_1000_fighter10v10')
    parser.add_argument('--agent1', type=str, default='ganlu')
    # parser.add_argument('--agent2', type=str, default='ganlu')
    parser.add_argument('--agent2', type=str, default='fix_rule')
    parser.add_argument('--round', type=int, default=5000)
    parser.add_argument('--fps', type=float, default=0)
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--random_pos', action='store_true') # 带这个参数就是True
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--log_path', type=str, default='default_log')
    args = parser.parse_args()

    map_path = 'maps/' + args.map + '.map'
    agent1_path = 'agent/' + args.agent1 + '/agent.py'
    agent2_path = 'agent/' + args.agent2 + '/agent.py'
    agent1_import_path = 'agent.' + args.agent1 + '.agent'
    agent2_import_path = 'agent.' + args.agent2 + '.agent'

    if not os.path.exists(map_path):
        print('Error: map file not exist!')
        exit(1)
    if not os.path.exists(agent1_path):
        print('Error: agent1 file not exist!')
        exit(1)
    if not os.path.exists(agent2_path):
        print('Error: agent2 file not exist!')
        exit(1)

    if args.fps == 0:
        step_delay = 0
    else:
        step_delay = 1 / args.fps

    agent1_module = importlib.import_module(agent1_import_path)
    agent2_module = importlib.import_module(agent2_import_path)
    try: agent1 = agent1_module.Agent(name='baseline')
    except: 
        raise
        agent1 = agent1_module.Agent()
    try: agent2 = agent2_module.Agent(name='baseline2')
    except: agent2 = agent2_module.Agent()

    agent1_obs_ind = agent1.get_obs_ind()
    agent2_obs_ind = agent2.get_obs_ind()

    if args.log:
        if args.log_path == 'default_log':
            log_flag = args.agent1 + '_vs_' + args.agent2
        else:
            log_flag = args.log_path
    else:
        log_flag = False
    env = Environment(map_path, agent1_obs_ind, agent2_obs_ind,
                    max_step=args.max_step, render=True,
                    random_pos=args.random_pos, log=log_flag)

    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()
    agent1.set_map_info(size_x, size_y, side1_detector_num, side1_fighter_num)
    agent2.set_map_info(size_x, size_y, side2_detector_num, side2_fighter_num)

    step_cnt = 0
    round_cnt = 0

    for x in range(args.round):
        if x != 0:
            env.reset()
        step_cnt = 0
        round_cnt += 1
        print('round ', round_cnt)

        try: agent1.reset(record=True)
        except: pass
        try: agent2.reset()
        except: pass

        side1_obs_dict, side2_obs_dict = env.get_obs()
        while True:
            time.sleep(step_delay)
            step_cnt += 1

            try: side1_detector_action, side1_fighter_action = agent1.get_action(side1_obs_dict, step_cnt, enemy_obs_dict=side2_obs_dict)
            except: side1_detector_action, side1_fighter_action = agent1.get_action(side1_obs_dict, step_cnt)
            try: side2_detector_action, side2_fighter_action = agent2.get_action(side2_obs_dict, step_cnt, enemy_obs_dict=side1_obs_dict)
            except: side2_detector_action, side2_fighter_action = agent2.get_action(side2_obs_dict, step_cnt)

            env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)
            side1_obs_dict, side2_obs_dict = env.get_obs()

            if env.get_done():
                env_reward = env.get_reward()
                agent1_reward = env_reward[2]
                agent2_reward = env_reward[5]
                try: agent1.get_action(side1_obs_dict, step_cnt, enemy_obs_dict=side2_obs_dict, game_reward=agent1_reward)
                except: pass
                try: agent2.get_action(side2_obs_dict, step_cnt, enemy_obs_dict=side1_obs_dict, game_reward=agent2_reward)
                except: pass
                break

        try: train_ganlu(agent1)
        except: pass
        try: train_ganlu(agent2)
        except: pass