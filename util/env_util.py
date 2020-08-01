import math

import numpy as np
import torch
from util.other_util import *

K_DENSE_PENALTY = 1e-3
K_CORNER_PENALTY = 0.05

def get_inf_from_two_dict(side1_obs_dict, side2_obs_dict):
    side1_infs = []
    side2_infs = []
    for fighter_obs in side1_obs_dict['fighter_obs_list']:
        side1_inf = []
        for inf_str in INF_STRS:
            if inf_str in ['hit_target', 'missile_type']:
                side1_inf.append(fighter_obs['last_action'][inf_str])
            else:
                side1_inf.append(fighter_obs[inf_str])
        side1_infs.append(side1_inf)
    for fighter_obs in side2_obs_dict['fighter_obs_list']:
        side2_inf = []
        for inf_str in INF_STRS:
            if inf_str in ['hit_target', 'missile_type']:
                side2_inf.append(fighter_obs['last_action'][inf_str])
            else:
                side2_inf.append(fighter_obs[inf_str])
        side2_infs.append(side2_inf)

    side1_infs = np.array(side1_infs)
    side2_infs = np.array(side2_infs)
    
    return side1_infs, side2_infs

def get_inf_from_one_dict(obs_dict): # 对战时使用
    enemy_dict = {}
    fighter_infs = []
    detector_infs = []
    for fighter_obs in obs_dict['fighter_obs_list']:
        fighter_inf = []
        for inf_str in INF_STRS:
            if inf_str in ['hit_target', 'missile_type']:
                fighter_inf.append(fighter_obs['last_action'][inf_str])
            else:
                fighter_inf.append(fighter_obs[inf_str])
        fighter_infs.append(fighter_inf)
        if fighter_obs['alive']:
            for enemy_inf in fighter_obs['r_visible_list']:
                enemy_dict[enemy_inf['id']] = (enemy_inf['id'], enemy_inf['pos_x'], enemy_inf['pos_y'])
                
    fighter_infs = np.array(fighter_infs, dtype=int)
    enemy_infs = np.array(list(enemy_dict.values()), dtype=int).reshape((-1, 3))

    # 友方；敌方(id, x, y)
    return detector_infs, fighter_infs, enemy_infs

def get_alive_inf(infs):
    alive = infs[:, ALIVE] == 1
    infs = infs[alive]
    return infs

# 所有友方被标记为1，敌方被标记为2
def to_dataset_data(friend_coordinates, enemy_coordinates): # (x, y)
    friend_coordinates = np.concatenate(
        [friend_coordinates, np.ones((friend_coordinates.shape[0], 1))], axis=1)
    enemy_coordinates = np.concatenate(
        [enemy_coordinates, -np.ones((enemy_coordinates.shape[0], 1))], axis=1)

    con = np.concatenate([friend_coordinates, enemy_coordinates], axis=0)
    return con # (x, y, 我方/敌方) 可能返回一个(0,3)的数组

def standardize(coordinates): # 转换坐标
    coordinates = coordinates.astype(np.float32)
    if len(coordinates.shape) == 2:
        coordinates[:, 0] /= ENV_WIDTH
        coordinates[:, 0] = 2 * coordinates[:, 0] - 1
        coordinates[:, 1] /= ENV_HEIGHT
        coordinates[:, 1] = 2 * coordinates[:, 1] - 1
    else:
        coordinates[0] /= ENV_WIDTH
        coordinates[1] /= ENV_HEIGHT
    return coordinates

def rearrange(coordinates): # 重排
    square_dis = coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2
    seq = np.argsort(square_dis) # 从小到大的编号
    coordinates = coordinates[seq]
    return coordinates

# 转换：左上坐标为(-1,-1)，右下坐标为(1,1)，正中间坐标为(0,0)
# 按照离中心的距离，从小到大排序
# 暂不区分异构飞机
def to_standardized_data(coordinates): # (x, y, 我方/敌方)
    coordinates = rearrange(standardize(coordinates))
    return coordinates # (x, y, 我方/敌方)

def convert_dict_action_to_array(actions):
    output_actions = []
    for action in actions:
        hit_tar = action['hit_target']
        if action['missile_type'] == 2:
            hit_tar += TOTAL_UNIT_NUM
        output_action = (action['course'], action['r_fre_point'] if action['r_iswork'] else 0,
            action['j_fre_point'] if action['j_iswork'] else 0, hit_tar)
        output_actions.append(output_action)
    output_actions = np.array(output_actions)
    return output_actions

def get_enemy_inf_from_strike(obs_dict):
    enemy_dict = {}
    fighter_obs_list = obs_dict['fighter_obs_list']
    for fighter_obs in fighter_obs_list:
        striking_dict_list = fighter_obs['striking_dict_list']
        for striking_dict in striking_dict_list:
            enemy_dict[striking_dict['target_id']] = (striking_dict['target_id'],
                striking_dict['pos_x'], striking_dict['pos_y'])
    enemy_infs = np.array(list(enemy_dict.values()), dtype=int).reshape((-1, 3))
    return enemy_infs

def strike_list_to_array(strike_list):
    strikes = []
    for strike in strike_list:
        attacker_id = strike['attacker_id']
        target_id = strike['target_id']
        step_count = strike['step_count']
        strikes.append(np.array([attacker_id, target_id, step_count]))
    return np.array(strikes, dtype=int).reshape(-1, 3)

def get_enemy_striked_num(strikes):
    enemy_striked_num = np.zeros((TOTAL_UNIT_NUM,))
    for strike in strikes:
        target_id = strike[1]
        enemy_index = target_id - 1
        enemy_striked_num[enemy_index] += 1
    return enemy_striked_num

def get_old_strikes(now_strikes):
    if len(now_strikes) == 0: return now_strikes
    old_strikes = now_strikes[now_strikes[:, 2] != 1].copy()
    old_strikes[:, 2] -= 1
    return old_strikes

def get_invalid_enemy_ids(attack_invalids, friend_infs):
    attack_invalid_indexes = np.where(attack_invalids == 1)[0]
    temp_friend_infs = friend_infs[attack_invalid_indexes]
    attack_invalid_enemy_ids = temp_friend_infs[:, HIT_TARGET]
    return attack_invalid_enemy_ids

def get_disappeared_enemy(friend_infs, buffered_enemy_infs, enemy_infs):
    disappeared_enemy_ids = []
    for enemy_inf in buffered_enemy_infs:
        if enemy_inf[ID] not in enemy_infs[:, ID]:
            enemy_coordinate = enemy_inf[[POS_X, POS_Y]]
            friend_coordinates = friend_infs[:, [POS_X, POS_Y]]
            difference = enemy_coordinate - friend_coordinates
            distance = np.linalg.norm(difference, ord=2, axis=1)
            if (distance < 100).any():
                disappeared_enemy_ids.append(enemy_inf[ID])
    return np.array(disappeared_enemy_ids)

def divide_infs(infs):
    ids = infs[: ID]
    detector_indexes = np.where(ids<=2)[0]
    fighter_indexes = np.where(ids>2)[0]
    detector_infs = infs[detector_indexes]
    fighter_infs = infs[fighter_indexes]
    return detector_infs, fighter_infs

# 传进来的是所有的信息，返回的长度是12
def get_inf_from_signal_rewards(friend_infs):
    signal_rewards = friend_infs[:, LAST_REWARD]
    detect_nums = get_inf_from_signal_reward(signal_rewards, DETECT_INDEX)
    dies = get_inf_from_signal_reward(signal_rewards, DIE_INDEX)
    destroy_nums = get_inf_from_signal_reward(signal_rewards, DESTROY_INDEX)
    return detect_nums, dies, destroy_nums

def get_die_coordinates(friend_infs, dies):
    die_index = np.where(dies == 1)[0]
    return friend_infs[die_index][:, [POS_X, POS_Y]]

def get_corner_penalty(friend_map):
    x, y = np.where(friend_map == 1)
    x = x[0];   y = y[0]
    x_len = min(x, DIVIDE - x)
    y_len = min(y, DIVIDE - y)
    lenn = min(x_len, y_len)
    penalty = min(3 - lenn, 0)
    return penalty * K_CORNER_PENALTY

def dis_to_edge(friend_coordinate):
    x = friend_coordinate[0]
    y = friend_coordinate[1]
    x_len = min(x, 1000 - x)
    y_len = min(y, 1000 - y)
    min_len = min(x_len, y_len)
    return min_len

def get_dense_penalty(friends_map, friend_map):
    sigma = 0
    all_friend_num = np.sum(friends_map) - 1
    friends_map = friends_map - friend_map
    x_hat, y_hat = np.where(friend_map == 1)
    x_hat = x_hat[0];   y_hat = y_hat[0]
    x, y = np.where(friends_map != 0)

    for (x_i, y_i) in zip(x, y):
        num_friend = friends_map[x_i, y_i]
        dis = abs(x_hat - x_i) + abs(y_hat - y_i)
        sigma += dis * num_friend
    penalty = min(sigma - 4 * all_friend_num, 0) # 如果平均超过为3个格子，那么损失为0
    return penalty * K_DENSE_PENALTY

def get_distances(source_coordinate, target_coordinates):
    if len(target_coordinates) == 0: return 0
    differences = source_coordinate - target_coordinates
    dim = len(differences.shape)
    if dim == 1: axis = 0
    elif dim == 2: axis = 1
    distances = np.linalg.norm(differences, ord=2, axis=axis)
    return distances

def get_medium_coordinate(friend_coordinates):
    if len(friend_coordinates.shape) == 2:
        return np.mean(friend_coordinates, axis=0)
    else:
        return friend_coordinates

# 为了两个探测机而设计的
def update_cluster_center(old_center, friend_coordinates):
    cluster_num = math.ceil(len(friend_coordinates) / 2)
    distances = get_distances(old_center, friend_coordinates)
    sorted_indexes = np.argsort(distances)
    choosed_index = sorted_indexes[int(cluster_num / 2)] # 取中位数
    choosed_coordinate = friend_coordinates[choosed_index]
    return choosed_coordinate

def farest_friend_coordinate(coordinate, friend_coordinates):
    distances = get_distances(coordinate, friend_coordinates)
    max_index = np.argmax(distances)
    choosed_coordinate = friend_coordinates[max_index]
    return choosed_coordinate

def escape(friend_coordinate, enemy_coordinates):
    direction_vector = friend_coordinate - enemy_coordinates
    if len(direction_vector.shape) == 2:
        direction_vector = np.mean(direction_vector, axis=0)
    goal = friend_coordinate + direction_vector
    return goal.astype(int)

def try_attack(friend_inf, enemy_id, distance): 
    # 调用的前提是敌方还没被打击
    # 返回attack=VOID意味着逃跑
    attack = VOID
    if distance < SHORT_MISSLE_RANGE:
        if friend_inf[S_MISSILE_LEFT]:
            attack = enemy_id + TOTAL_UNIT_NUM
        elif friend_inf[L_MISSILE_LEFT]:
            attack = enemy_id
    elif distance < LONG_MISSLE_RANGE:
        if friend_inf[L_MISSILE_LEFT]:
            attack = enemy_id
        elif friend_inf[S_MISSILE_LEFT]:
            attack = enemy_id + TOTAL_UNIT_NUM
    elif distance < FIGHTER_DETECT_RANGE:
        if friend_inf[L_MISSILE_LEFT] or friend_inf[S_MISSILE_LEFT]:
            attack = 0
    else: attack = 0
    return attack

def test_missile_range(side1_infs, side2_infs, side1_attacks):
    side1_attack_out = side1_attacks[:, FIGHTER_MISSILE]
    side1_attack_indexes = np.where(side1_attack_out != 0)[0]
    if len(side1_attack_indexes) == 0:
        return
    temp = side1_infs[side1_attack_indexes]
    side1_coordinates = temp[:, [POS_X, POS_Y]]

    temp = side1_attack_out > TOTAL_UNIT_NUM
    side2_hitted_id = np.copy(side1_attack_out)
    missile_types = np.copy(side1_attack_out)
    side2_hitted_id[temp] -= TOTAL_UNIT_NUM
    side2_hitted_id = side2_hitted_id[side2_hitted_id != 0]
    missile_types[missile_types > 0] = 1
    missile_types[temp] = 2
    missile_types = missile_types[missile_types != 0]
    side2_hitted_index = side2_hitted_id - 1
    temp = side2_infs[side2_hitted_index]
    side2_coordinates = temp[:, [POS_X, POS_Y]]
    difference = side1_coordinates - side2_coordinates
    L1_distances = np.linalg.norm(difference, ord=1, axis=1)
    L2_distances = np.linalg.norm(difference, ord=2, axis=1)
    for i, missile_type in enumerate(missile_types):
        print('type: ', missile_type, ' l1: ', L1_distances[i], ' l2: ', L2_distances[i])

def convert_angle(angle): # angle: 0~1
    # 角度：0~1->0~180; 0~-1->360~180
    # 为了避免分段，1和-1比较相似，比0和1相似多了
    if INPUT_COURSE_DIM == 1:
        if angle < 0.5:
            return 2 * angle
        else:
            return 2 * (angle - 1)
    # 多一个特征：sin值，使0和1有一半的连续
    elif INPUT_COURSE_DIM == 2: # 默认
        sin = math.sin(2 * math.pi * angle)
        return angle, sin

def convert_to_input_course(angles): # angle: 0~1
    if type(angles) != np.ndarray:
        sin_value = math.sin(2 * math.pi * angles)
        out = np.array([angles, sin_value])
    else:
        angles = angles.reshape((-1, 1))
        sin_values = np.sin(2 * math.pi * angles)
        out = np.concatenate([angles, sin_values], axis=1)
    return out

def course_start_to_goal(start, goal):
    relative_coordinate = goal - start
    course = math.atan2(relative_coordinate[1], relative_coordinate[0]) / math.pi * 180
    return course

def tanh_derivative_loss(x, k, t, l):
    print(k * (math.log(math.cosh(t * x))/t - l * x + 1 - math.log(math.cosh(l * t))/t))

def exp_derivative_loss(x, offset, k, t, l):
    c = -(1 + 1/t) * math.exp(-t * (1 + offset))
    loss = k * math.exp(-l * t * (x + l * offset)) / t + l * x * math.exp(-t * (1 + offset)) + c
    derivative = k * (-l * math.exp(-l * t * (x + l * offset)) + l * math.exp(-t * (1 + offset)))
    print('loss: ', loss)
    print('derivative: ', derivative)

if __name__ == '__main__':
    exp_derivative_loss(x=0, offset=-0.5, k=0.5, t=2, l=-1)