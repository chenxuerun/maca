#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: fight.py
@time: 2018/3/9 0009 16:41
@desc: execution battle between two agents
"""
import argparse
import os
import time
from interface import Environment
from common.agent_process import AgentCtrl

from train.xuance.train_agent import train_xuance

# 1000_1000_fighter10v10
# 1000_1000_2_10_vs_2_10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="1000_1000_fighter10v10", help='map name, only name, not file path')
    parser.add_argument("--agent1", type=str, default="xuance", help='agent 1 name, only name, not path')
    parser.add_argument("--agent2", type=str, default="fix_rule", help='agent 2 name, only name, not path')
    parser.add_argument("--round", type=int, default=2, help='play rounds')
    parser.add_argument("--fps", type=float, default=0, help='display fps')
    parser.add_argument("--max_step", type=int, default=5000, help='max step in a round')
    parser.add_argument("--random_pos", action="store_true", help='if the initial positions are random or fix')
    parser.add_argument("--log", action="store_true", help='saving log')
    parser.add_argument("--log_path", type=str, default="default_log", help='log folder name')
    args = parser.parse_args()

    print('Map:', args.map)
    print('Side1 agent:', args.agent1)
    print('Side2 agent:', args.agent2)
    print('Round number:', args.round)

    side1_win_times = 0
    side2_win_times = 0
    draw_times = 0

    # file path constructing
    map_path = 'maps/' + args.map + '.map'
    agent1_path = 'agent/' + args.agent1 + '/agent.py'
    agent2_path = 'agent/' + args.agent2 + '/agent.py'

    if not os.path.exists(map_path):
        print('Error: map file not exist!')
        exit(-1)
    if not os.path.exists(agent1_path):
        print('Error: agent1 file not exist!')
        exit(-1)
    if not os.path.exists(agent2_path):
        print('Error: agent2 file not exist!')
        exit(-1)
    # delay calc
    if args.fps == 0:
        step_delay = 0
    else:
        step_delay = 1 / args.fps

    # environment initiation
    if args.log:
        if args.log_path == 'default_log':
            log_flag = args.agent1 + '_vs_' + args.agent2
        else:
            log_flag = args.log_path
    else:
        log_flag = False
    env = Environment(map_path, 'raw', 'raw', max_step=args.max_step, render=True,
                      random_pos=args.random_pos, log=log_flag)
    # get map info
    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()

    # create agent
    agent1 = AgentCtrl(args.agent1, size_x, size_y, side1_detector_num, side1_fighter_num)
    agent2 = AgentCtrl(args.agent2, size_x, size_y, side2_detector_num, side2_fighter_num)
    if not agent1.agent_init():
        print('ERROR: Agent1 init failed!')
        agent1.terminate()
        agent2.terminate()
        exit(-1)
    else:
        print('Agent1 init success!')
    if not agent2.agent_init():
        print('ERROR: Agent2 init failed!')
        agent1.terminate()
        agent2.terminate()
        exit(-1)
    else:
        print('Agent2 init success!')

    # execution
    # input("Press the <ENTER> key to continue...")
    for x in range(args.round):
        if x != 0:
            env.reset()
        step_cnt = 0

        while True:
            time.sleep(step_delay)
            step_cnt += 1
            # get obs
            side1_obs_dict, side2_obs_dict = env.get_obs()
            # get action
            agent1_action, agent1_result = agent1.get_action(side1_obs_dict, step_cnt)
            if agent1_result == 0:
                side1_detector_action = agent1_action['detector_action']
                side1_fighter_action = agent1_action['fighter_action']
            agent2_action, agent2_result = agent2.get_action(side2_obs_dict, step_cnt)
            if agent2_result == 0:
                side2_detector_action = agent2_action['detector_action']
                side2_fighter_action = agent2_action['fighter_action']

            # execution
            if agent1_result == 0 and agent2_result == 0:
                env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)
            elif agent1_result != 0 and agent2_result != 0:
                env.set_surrender(2)
            elif agent1_result != 0:
                env.set_surrender(0)
            else:
                env.set_surrender(1)

            if env.get_done():
                time.sleep(2)
                break

        train_xuance(agent1)
        # train_xuance(agent2, agent2.collection)

    agent1.terminate()
    agent2.terminate()
    input("Press the <ENTER> key to continue...")
