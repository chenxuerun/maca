import random

import numpy as np

from util.env_util import ENV_HEIGHT, ENV_WIDTH
from util.other_util import DIVIDE

# 计数从左上角开始，同坐标

BLOCK_WIDTH = ENV_WIDTH // DIVIDE
BLOCK_HEIGHT = ENV_HEIGHT // DIVIDE

RL_GAMMA = 0.98
EXPLORE_PROB = 1
EXPLORE_NUM_CHOICE = 5
SWING = 60  # 摆动幅度

COMMON_NET_FOLDER = 'model/ganlu/common'
DQN_MODEL_FOLDER = 'model/ganlu/dqn'
INTERVAL = 30

COMMON_OUT_CHANNEL = 4
NINE_ACTION = False

def to_map(coordinates): # 坐标的最小值是0，最大值是1000
    out_map = np.zeros((DIVIDE, DIVIDE))
    if len(coordinates) == 0:
        return out_map
    dim = len(coordinates.shape)
    if dim == 1:
        x_index = coordinates[0] // BLOCK_WIDTH
        y_index = coordinates[1] // BLOCK_HEIGHT
        if x_index >= DIVIDE: x_index = DIVIDE - 1
        if y_index >= DIVIDE: y_index = DIVIDE - 1
        out_map[x_index, y_index] += 1
        return out_map, block_to_number((x_index, y_index))
    elif dim == 2:
        x_indexes = coordinates[:, 0] // BLOCK_WIDTH
        y_indexes = coordinates[:, 1] // BLOCK_HEIGHT
        x_indexes[x_indexes >= DIVIDE] = DIVIDE - 1
        y_indexes[y_indexes >= DIVIDE] = DIVIDE - 1
        for i in range(len(coordinates)):
            out_map[x_indexes[i], y_indexes[i]] += 1
        return out_map
    else: raise Exception()

def to_divided_maps(coordinates):
    maps = []
    block_indexes = []
    for coordinate in coordinates:
        map, block_index = to_map(coordinate)
        maps.append(map)
        block_indexes.append(block_index)
    maps = np.array(maps).reshape((-1, DIVIDE, DIVIDE))
    block_indexes = np.array(block_indexes)
    return maps, block_indexes

def random_action(q_map, num_choice=EXPLORE_NUM_CHOICE):
    flattened_map = np.reshape(q_map, (-1,))
    largest_indexes = np.argsort(flattened_map)[::-1][0:num_choice]
    choice = np.random.choice(largest_indexes)
    return choice

def q_to_action(q_map):
    # print(q_map)
    if random.random() < EXPLORE_PROB:
        block_index = random_action(q_map)
    else:
        block_index = np.argmax(q_map)
    temp = np.zeros_like(q_map).reshape((-1,))
    temp[block_index] = 1
    action_map = temp.reshape(q_map.shape)
    return action_map, block_index

'''
def to_big_map(coordinates): # (x, y) ~ [0, 1000]
    out_map = np.zeros((5, 5))
    dim = len(coordinates.shape)
    block_len = 200
    if dim == 1:
        x_index = coordinates[0] // block_len
        y_index = coordinates[1] // block_len
        out_map[x_index, y_index] += 1
        return out_map, (x_index, y_index)
    elif dim == 2:
        x_indexes = coordinates[:, 0] // block_len
        y_indexes = coordinates[:, 1] // block_len
        for i in range(len(coordinates)):
            out_map[x_indexes[i], y_indexes[i]] += 1
        return out_map
    else:
        raise Exception()

def to_small_map(coordinates, big_map_block):
    out_map = np.zeros((5, 5))
    dim = len(coordinates.shape)
    block_len = 40
    x_min = 200 * big_map_block[0]
    y_min = 200 * big_map_block[1]
    if dim == 1:
        x_index = (coordinates[0] - x_min) // block_len
        y_index = (coordinates[1] - y_min) // block_len
        out_map[x_index, y_index] += 1
        return out_map, (x_index, y_index)
    elif dim == 2:
        x_indexes = (coordinates[:, 0] - x_min) // block_len
        y_indexes = (coordinates[:, 1] - y_min) // block_len
        for i in range(len(coordinates)):
            if (x_indexes[i] in range(0, 5)) and (y_indexes[i] in range(0,5)):
                out_map[x_indexes[i], y_indexes[i]] += 1
        return out_map
    else:
        raise Exception()
'''

def block_center_coordinate(block):
    x = BLOCK_WIDTH * block[0] + BLOCK_WIDTH / 2
    y = BLOCK_HEIGHT * block[1] + BLOCK_HEIGHT / 2
    return np.array([x, y], dtype=int)

def block_min_coordinate(block):
    x = BLOCK_WIDTH * block[0]
    y = BLOCK_HEIGHT * block[1]
    return np.array([x, y], dtype=int)

def get_goal_from_action(coordinate, action_block_index): # action: 0~8
    delta_x, delta_y = (action_block_index % 3 - 1, action_block_index // 3 - 1)
    goal = coordinate.copy()
    goal[0] += delta_x * BLOCK_WIDTH
    goal[1] += delta_y * BLOCK_HEIGHT
    return goal

def get_action_from_goal(goal, coordinate=None):
    if NINE_ACTION:
        action_map = np.zeros((9,))
        delta = goal - coordinate
        delta_x = min(max(round(delta[0] / BLOCK_WIDTH), 1), -1) + 1
        delta_y = min(max(round(delta[1] / BLOCK_HEIGHT), 1), -1) + 1
        action_index = int(round(3 * delta_y + delta_x))
        action_map[action_index] += 1
        action_map = action_map.reshape((3,3))
    else:
        action_map, action_index = to_map(goal)
    return action_map, action_index

def get_dqn_name(friend_id):
    if friend_id == 1: return 'dqnd1' 
    elif friend_id == 2: return 'dqnd2'
    elif friend_id in [3,4,5,6,7]: return 'dqnf1'
    elif friend_id in [8,9,10,11,12]: return 'dqnf2'

def number_to_block(number):
    return (number % DIVIDE, number // DIVIDE)

def block_to_number(block):
    return block[1] * DIVIDE + block[0]

def equal(block, number):
    return block[1] * DIVIDE + block[0] == number