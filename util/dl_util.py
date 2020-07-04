import torch
import torch.nn as nn

GAMMA = 0.999
EXPLORE = True

AE_MODEL_FOLDER = 'model/xuance/ae'
RL_MODEL_FOLDER = 'model/xuance/rl'

AE_HIDDEN_DIMS = [600, 600]
AE_HIDDEN_ACTIVATION = nn.Tanh
AE_Z_DIM = 300
AE_Z_ACTIVATION = nn.Tanh
AE_DECODERS_NUM = 10

COORDINATE_DIM = 2
INPUT_COURSE_DIM = 2

DECISION_INPUT_DIM = AE_Z_DIM + COORDINATE_DIM + INPUT_COURSE_DIM # 环境+坐标+角度
DECISION_HIDDEN_DIMS = [AE_Z_DIM//2, AE_Z_DIM//4, AE_Z_DIM//8]
DECISION_HIDDEN_ACTIVATION = nn.Tanh
DECISION_OUT_DIM = 1
DECISION_OUT_ACTIVATION = nn.Sigmoid

SHIFT_INPUT_DIM = AE_Z_DIM + COORDINATE_DIM + INPUT_COURSE_DIM # 环境+坐标+角度
SHIFT_HIDDEN_DIMS = DECISION_HIDDEN_DIMS
SHIFT_HIDDEN_ACTIVATION = nn.Tanh
SHIFT_OUT_DIM = 2
SHIFT_OUT_ACTIVATION = nn.Sigmoid

def cross_entropy(x, l):
    return - l * torch.log(x) - (1 - l) * torch.log(1 - x)

def mean_square(x, l):
    return (x - l) ** 2 / 2