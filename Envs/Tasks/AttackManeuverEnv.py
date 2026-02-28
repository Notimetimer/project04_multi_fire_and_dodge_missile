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
from math import pi, cos, sin # [新增] 引入数学库
import random # [新增] 引入随机库

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

    def reset(self, seed=None, options=None,\
                red_birth_state=None, blue_birth_state=None, \
                  red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo, seed=seed, options=options)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.lock_time_count = 0

    # 进攻策略观测量
    def attack_obs(self, side, pomdp=0):
        pre_full_obs = self.base_obs(side, pomdp)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.attack_key_order}
        full_obs["ego_main"][6]=0

        # 将观测按顺序拉成一维数组
        # flat_obs = flatten_obs(full_obs, self.key_order)
        flat_obs = flatten_obs(full_obs, self.attack_key_order)
        return flat_obs, full_obs

    def get_terminate_and_reward(self, side):  # 进攻策略训练与奖励
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


        # # 锁定时间达到就判为胜利，用于提高训练效率
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

        # 高度奖励
        pre_alt_opt = target_alt + np.clip((dist - 10e3) / (40e3 - 10e3) * 5e3, 0, 5e3)
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)

        r_alt = (alt <= alt_opt) * (alt - self.min_alt) / (alt_opt - self.min_alt) + \
                (alt > alt_opt) * (1 - (alt - alt_opt) / (self.max_alt - alt_opt))

        # 高度限制奖励/惩罚
        r_alt += (alt <= self.min_alt_safe + 1e3) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)


        # 速度奖励
        speed_opt = 1.5 * 340
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

    
    # # 并行化修改
    
    # def get_red_rule_action(self):
    #     """
    #     供并行 Worker 调用的辅助函数。
    #     利用环境内部状态生成红方规则动作。
    #     """
    #     # 1. 获取红方内部状态 (可以直接读属性，也可以用 get_state)
    #     # 为了兼容性，我们通过 obs 获取感知信息
    #     r_state = self.get_state('r')
        
    #     dist = r_state["target_information"][3]
    #     # 注意：decision_rule 需要的参数需与 base_obs 的定义对应
    #     # 这里做一个简化的参数提取
        
    #     r_action = self.decision_rule(
    #         ego_pos_=self.RUAV.pos_,
    #         ego_psi=self.RUAV.psi,
    #         enm_delta_psi=r_state["target_information"][1], # 相对方位角
    #         distance=dist,
    #         warning=r_state["warning"],
    #         threat_delta_psi=r_state["threat"][0], # 假设 threat[0] 是方位
    #         ally_missiles=self.Rmissiles,
    #         wander=1 # 启用规则漫游
    #     )
        
    #     # 2. 顺便处理红方发射逻辑
    #     # 因为这是在 Worker 内部，直接调用全局或类方法均可
    #     # 假设 launch_missile_if_possible 已经导入
    #     launch_missile_if_possible(self, 'r')
        
    #     return r_action
    
    # # [新增] 封装随机出生状态生成逻辑
    # def _get_random_birth_states(self):
    #     # 这里完全照搬你在串行代码中的逻辑
    #     red_R_ = random.uniform(30e3, 40e3) 
    #     red_beta = random.uniform(0, 2 * pi)
    #     red_psi = random.uniform(0, 2 * pi)
    #     red_height = random.uniform(3e3, 10e3)
    #     red_N = red_R_ * cos(red_beta)
    #     red_E = red_R_ * sin(red_beta)
    #     blue_height = random.uniform(3e3, 10e3)

    #     DEFAULT_RED_BIRTH_STATE = {
    #         'position': np.array([red_N, red_height, red_E]),
    #         'psi': red_psi
    #     }
    #     DEFAULT_BLUE_BIRTH_STATE = {
    #         'position': np.array([0.0, blue_height, 0.0]),
    #         'psi': pi
    #     }
    #     return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    # # [修改] Reset 逻辑：如果没有传入具体的 birth_state，则自动随机
    # def reset(self, seed=None, options=None, 
    #           red_birth_state=None, blue_birth_state=None, 
        #       red_init_ammo=6, blue_init_ammo=6):
        
        # # 1. 如果外部没有指定出生状态，则使用内部随机逻辑
        # if red_birth_state is None and blue_birth_state is None:
        #     red_birth_state, blue_birth_state = self._get_random_birth_states()

        # # 2. 调用父类 Battle 的 reset
        # super().reset(seed=seed, options=options, 
        #               red_birth_state=red_birth_state, blue_birth_state=blue_birth_state, 
        #               red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo)
        
        # # 3. 初始化额外变量
        # self.last_dist_dot = None
        # self.last_dhor = None
        # self.lock_time_count = 0
        
        # # 4. 获取观测并返回
        # flat_obs, _ = self.attack_obs('b')
        # return flat_obs.astype(np.float32), {}