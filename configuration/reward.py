from util.other_util import *

class GlobalVar:
    # reward
    
    # detector radar see a detector
    reward_radar_detector_detector = DETECT_SIGNAL_REWARD
    # detector radar see a fighter
    reward_radar_detector_fighter = DETECT_SIGNAL_REWARD
    # fighter radar see a detector
    reward_radar_fighter_detector = DETECT_SIGNAL_REWARD
    # fighter radar see a fighter
    reward_radar_fighter_fighter = DETECT_SIGNAL_REWARD

    # Missile hit a detector
    reward_strike_detector_success = DESTROY_SIGNAL_REWARD
    # Missile miss a detector
    reward_strike_detector_fail = 0
    # Missile hit a fighter
    reward_strike_fighter_success = DESTROY_SIGNAL_REWARD
    # Missile miss a fighter
    reward_strike_fighter_fail = 0

    # A detector been destroyed
    reward_detector_destroyed = DIE_SIGNAL_REWARD
    # A fighter been destroyed
    reward_fighter_destroyed = DIE_SIGNAL_REWARD

    # A valid attack action
    reward_strike_act_valid = 0
    # An invalid attack action
    reward_strike_act_invalid = 0

    # Keep alive in a step
    reward_keep_alive_step = 0

    # Round reward：totally win
    reward_totally_win = REWARD[CHOOSE_REWARD]['total_win_reward']
    # Round reward：totally lose
    reward_totally_lose = REWARD[CHOOSE_REWARD]['total_lose_reward']
    # Round reward：win
    reward_win = REWARD[CHOOSE_REWARD]['win_reward']
    # Round reward：lose
    reward_lose = REWARD[CHOOSE_REWARD]['lose_reward']
    # Round reward：draw
    reward_draw = REWARD[CHOOSE_REWARD]['draw_reward']


def get_reward_radar_detector_detector():
    return GlobalVar.reward_radar_detector_detector


def get_reward_radar_detector_fighter():
    return GlobalVar.reward_radar_detector_fighter


def get_reward_radar_fighter_detector():
    return GlobalVar.reward_radar_fighter_detector


def get_reward_radar_fighter_fighter():
    return GlobalVar.reward_radar_fighter_fighter


def get_reward_strike_detector_success():
    return GlobalVar.reward_strike_detector_success


def get_reward_strike_detector_fail():
    return GlobalVar.reward_strike_detector_fail


def get_reward_strike_fighter_success():
    return GlobalVar.reward_strike_fighter_success


def get_reward_strike_fighter_fail():
    return GlobalVar.reward_strike_fighter_fail


def get_reward_detector_destroyed():
    return GlobalVar.reward_detector_destroyed


def get_reward_fighter_destroyed():
    return GlobalVar.reward_fighter_destroyed


def get_reward_strike_act_valid():
    return GlobalVar.reward_strike_act_valid


def get_reward_strike_act_invalid():
    return GlobalVar.reward_strike_act_invalid


def get_reward_keep_alive_step():
    return GlobalVar.reward_keep_alive_step


def get_reward_win():
    return GlobalVar.reward_win


def get_reward_lose():
    return GlobalVar.reward_lose


def get_reward_totally_win():
    return GlobalVar.reward_totally_win


def get_reward_totally_lose():
    return GlobalVar.reward_totally_lose


def get_reward_draw():
    return GlobalVar.reward_draw
