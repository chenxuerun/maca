import random

import numpy as np

from util.env_util import ENV_HEIGHT, ENV_WIDTH, get_distances, dis_to_edge
from util.other_util import DIVIDE, TOTAL_UNIT_NUM, FIGHTER_NUM, POS_X, POS_Y, ID

# 计数从左上角开始，同坐标

BLOCK_WIDTH = ENV_WIDTH // DIVIDE
BLOCK_HEIGHT = ENV_HEIGHT // DIVIDE

RL_GAMMA = 0.98
EXPLORE_PROB = 1
EXPLORE_NUM_CHOICE = 10
SWING = 60  # 摆动幅度
MODIFY_MAP = False

MODEL_FOLDER = 'model/ganlu'
INTERVAL = 30

COMMON_OUT_CHANNELS = [4, 8]
COMMON_OUT_KERNEL_SIZES = [7, 7]
DQN_OUT_CHANNELS = [4, 2, 1]
DQN_OUT_KERNEL_SIZES = [5, 3, 3]

FIGHTER_TO_DETECTOR_TH = 200       # 太远了不行, 最短距离
FIGHTER_TO_FIGHTER_TH = 200
EDGE_TH = 150
K_DIS_PENALTY = 3e-6

def to_map(coordinates): # 坐标的最小值是0，最大值是1000
    out_map = np.zeros((DIVIDE, DIVIDE))
    if len(coordinates) == 0:
        return out_map
    dim = len(coordinates.shape)
    if dim == 1:
        x_index = int(coordinates[0] // BLOCK_WIDTH)
        y_index = int(coordinates[1] // BLOCK_HEIGHT)
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

def get_action_from_goal(goal):
    action_map, action_index = to_map(goal)
    return action_map, action_index

# 对飞机之间的距离有些的要求
def get_dis_penalties(alive_friend_fighter_infs):
    dis_penalties = np.zeros((TOTAL_UNIT_NUM,))
    alive_fighter_num = len(alive_friend_fighter_infs)

    for alive_friend_fighter_inf in alive_friend_fighter_infs:
        index = alive_friend_fighter_inf[ID] - 1
        alive_friend_fighter_coordinate = alive_friend_fighter_inf[[POS_X, POS_Y]]

        # 不能离队友太远
        temp_dis = get_distances(alive_friend_fighter_coordinate, alive_friend_fighter_infs[:, [POS_X, POS_Y]])
        if len(temp_dis) > 1:
            temp_dis = np.sum(temp_dis) / (len(temp_dis) - 1)
        else: temp_dis = temp_dis[0]
        dis_to_fighter_penalty = - max(temp_dis - FIGHTER_TO_FIGHTER_TH, 0) * K_DIS_PENALTY
        dis_penalties[index] += dis_to_fighter_penalty

        # 不能太靠近边上
        temp_dis = dis_to_edge(alive_friend_fighter_coordinate)
        edge_penalty = - max(EDGE_TH - temp_dis, 0) * K_DIS_PENALTY
        dis_penalties[index] += edge_penalty

    return dis_penalties

def number_to_block(number):
    return (number % DIVIDE, number // DIVIDE)

def block_to_number(block):
    return block[1] * DIVIDE + block[0]

def equal(block, number):
    return block[1] * DIVIDE + block[0] == number