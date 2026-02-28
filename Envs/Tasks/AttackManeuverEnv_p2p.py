'''出生点改在外面指定'''
import numpy as np
from random import random
import random
from gym import spaces
import copy
import jsbsim
import sys
import os
import importlib
import copy


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from Envs.battle6dof1v1_missile0919 import *


# 通过继承构建观测空间、奖励函数和终止条件

class AttackTrainEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.attack_key_order = [
            "target_information",  # 8
            "ego_main",  # 7
            "border",  # 2
        ]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.lock_time_count = 0

    # 进攻策略观测量
    def attack_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.attack_key_order}
        full_obs["ego_main"][6]=0
        
        # 将观测按顺序拉成一维数组
        # flat_obs = flatten_obs(full_obs, self.key_order)
        flat_obs = flatten_obs(full_obs, self.attack_key_order)
        return flat_obs, full_obs

    def attack_terminate_and_reward(self, side):  # 进攻策略训练与奖励
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]

        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV

        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV

        target_alt = enm.alt

        # if alpha < 10*pi/180:
        #     self.lock_time_count += dt_maneuver
        # if alpha > 30*pi/180:
        #     self.lock_time_count = 0

        # 结束判断
        if self.t > self.game_time_limit:
            terminate = True
            # self.lose = 1  # 还没进入范围判定为负

        if not self.min_alt <= alt <= self.max_alt:
            terminate = True
            self.lose = 1

        if dist < 5e3 and alpha < pi / 4  or  self.lock_time_count > 10:
            terminate = True
            self.win = 1

        # 角度奖励
        r_angle = 1 - alpha / (pi / 3)  # 超出雷达范围就惩罚狠一点
        sin_phi = state["ego_main"][4]
        cos_phi = state["ego_main"][5]
        phi = atan2(sin_phi, cos_phi)
        # 滚转角惩罚
        r_angle -= 0.1 * abs(phi / pi)
        # 负过载惩罚
        if ego.Ny<0:
            r_angle -= 0.1 * abs(ego.Ny) / 2
        # 侧滑角惩罚
        r_angle -= 0.05 * np.clip(abs(ego.beta_air*180/pi / 5), 0, 1)
        # 迎角惩罚
        r_angle -= 0.01 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))

        # 高度奖励
        pre_alt_opt = target_alt + np.clip((dist - 10e3) / (40e3 - 10e3) * 5e3, 0, 5e3)
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)

        r_alt = (alt <= alt_opt) * (alt - self.min_alt) / (alt_opt - self.min_alt) + \
                (alt > alt_opt) * (1 - (alt - alt_opt) / (self.max_alt - alt_opt))

        # 高度限制奖励/惩罚
        r_alt += (alt <= self.min_alt_safe + 1e3) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)


        # 速度奖励
        speed_opt = 2.5 * 340
        r_speed = 1 - abs(speed - speed_opt) / (2 * 340)

        # 距离奖励
        # r_dist = (dist <= 10e3) * (dist - 0) / (10e3 - 0) + \
        #          (dist > 10e3) * (1 - (dist - 10e3) / (50e3 - 10e3))
        L_ = enm.pos_ - ego.pos_
        delta_v_ = enm.vel_ - ego.vel_
        dist_dot = np.dot(delta_v_, L_) / dist
        self.last_dist_dot = dist_dot
        r_dist = -dist_dot/340  # 接近率越高奖励越高

        # # 边界距离奖励 ###
        obs = self.base_obs(side, reward_fn=1)
        d_hor = obs["border"][0]
        r_border = d_hor

        # 事件奖励
        reward_event = 0
        if self.lose:
            reward_event = -30
        if self.win:
            reward_event = 30

        # 0.2? 0.02?
        reward = np.sum(np.array([2, 1, 1, 1, 1, 0.2]) * \
                        np.array([r_angle, r_alt, r_speed, r_dist, reward_event, r_border]))

        if terminate:
            self.running = False

        reward_for_show = np.array([r_angle, r_alt, r_speed, r_dist, reward_event, r_border])

        return terminate, reward, reward_for_show
