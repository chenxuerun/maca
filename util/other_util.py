import torch

VOID = -99999

DEVICE_STR = 'cuda:0'
DEVICE = torch.device(DEVICE_STR)

ENV_WIDTH = 1000
ENV_HEIGHT = 1000

FIGHTER_NUM = 10
TOTAL_UNIT_NUM = 10

DIVIDE = 20

FIGHTER_DETECT_RANGE = 180
LONG_MISSLE_RANGE = 121
SHORT_MISSLE_RANGE = 51

MAX_STRIKE_NUM = 2

INF_STRS = ['id', 'pos_x', 'pos_y', 'alive', 'last_reward', 'course', 'l_missile_left', 's_missile_left'
, 'hit_target'
#, 'missile_type'
]
ID = 0
POS_X = 1
POS_Y = 2
ALIVE = 3
LAST_REWARD = 4
COURSE = 5
L_MISSILE_LEFT = 6
S_MISSILE_LEFT = 7
HIT_TARGET = 8
# MISSILE_TYPE = 9

FIGHTER_ACTION_COURSE = 0
FIGHTER_ACTION_R = 1
FIGHTER_ACTION_J = 2
FIGHTER_ACTION_MISSILE = 3

HEX = 10
BIT = 3
DETECT_INDEX = 1
DIE_INDEX = 2
DESTROY_INDEX = 3
DETECT_SIGNAL_REWARD = HEX ** (BIT - DETECT_INDEX)
DESTROY_SIGNAL_REWARD = HEX ** (BIT - DESTROY_INDEX)
DIE_SIGNAL_REWARD = HEX ** (BIT - DIE_INDEX)

# 总共BIT位，第一位在最左边
# 每一位要么是1，要么是0
def get_inf_from_signal_reward(rewards, inf_index):
    assert inf_index <= BIT
    for index in range(1, inf_index):
        base = HEX ** (BIT - index)
        rewards = rewards % base
    base = HEX ** (BIT - inf_index)
    infs = rewards // base           # 1或0
    return infs.astype(int)

CHOOSE_REWARD = 'baseline'
REWARD = {
    'baseline': {
        'detect_reward': 0,
        'destroy_reward': 0.3,
        'die_reward': -0.3,
        'alive_reward': 0,
        'total_win_reward': 1,
        'win_reward': 1,
        'draw_reward': 0,
        'lose_reward': -1,
        'total_lose_reward': -1,
    },
    'baseline2': {
        'detect_reward': 0,
        'destroy_reward': 0,
        'die_reward': 0,
        'alive_reward': 0,
        'total_win_reward': 0,
        'win_reward': 0,
        'draw_reward': 0,
        'lose_reward': 0,
        'total_lose_reward': 0,
    }
}