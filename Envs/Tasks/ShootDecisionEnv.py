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

class ShootTrainEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.attack_key_order = [
            "target_observable", # 1
            "target_locked", # 1
            "missile_in_mid_term",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "weapon", # 1
        ]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.lock_time_count = 0

        self.target_hit = 0
        self.target_out = 0

    
    # 进攻策略观测量
    def attack_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.attack_key_order}
        
        '''
        todo:
        原始观测输入需要转为如下量，否则难以训练：
        1、目标进入角（水平）
        2、目标高度
        3、目标速度
        4、目标距离
        5、我机速度
        6、我机高度
        7、ATA_h
        8、我机速度倾角
        9、上一枚导弹射出计时
        10、目标存活与否
        10、剩余弹量
        '''

        # 将观测按顺序拉成一维数组
        # flat_obs = flatten_obs(full_obs, self.key_order)
        flat_obs = flatten_obs(full_obs, self.attack_key_order)
        return flat_obs, full_obs

    def attack_terminate_and_reward(self, side, u, missile_id=None):  # 进攻策略训练与奖励
        # 兼容：若 u 是一维数组或列表，则取第0个元素；若为标量则直接使用（不考虑更高维度）
        if isinstance(u, (list, tuple, np.ndarray)):
            arr = np.array(u)
            if arr.ndim == 1:
                ut = arr[0]
            elif arr.ndim == 0:
                ut = arr.item()
            else:
                ut = u
        else:
            ut = u
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        AA_hor = state["target_information"][6]
        launch_interval = state["weapon"]

        missile_time_since_shoot = state["weapon"]

        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            all_ally_missiles = self.Rmissiles
            alive_ally_missiles = self.alive_r_missiles

        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            all_ally_missiles = self.Bmissiles
            alive_ally_missiles = self.alive_b_missiles

        target_alt = enm.alt
        enm_state = self.get_state(enm.side)

        # 结束判断
        if self.t > self.game_time_limit:
            terminate = True
            self.lose = 1  # 还没进入范围判定为负

        # 被对面刀了直接判负
        if dist < 10e3 and enm_state["target_information"][4]<pi/6:
            terminate = True
            self.lose = 1

        # 命中判断
        if enm.dead: # or self.out_range(enm): # 驱赶也行， 新增
            terminate = True
            self.win = 1

        if enm.dead:
            self.target_hit = 1
        # if self.out_range(enm):
        #     self.target_out = 1
        #     terminate = True
        #     self.win = 1

        if self.out_range(ego) or ego.dead:
            terminate = True
            self.lose = 1

        if self.win and self.lose:
            self.draw = 1

        # reward_base = 100 / (self.game_time_limit/dt_maneuver)  # 防自杀奖励


        # # 发射惩罚，根据 missile_time_since_shoot
        reward_shoot = 0
        # if missile_id is not None:
        #     reward_shoot -= 10
        if ut == 1:
            reward_shoot -= 30
        if ut == 1:
            reward_shoot += np.clip((missile_time_since_shoot-30)/30, -1,1)  # 过30s发射就可以奖励了
            reward_shoot += 0.5 * abs(AA_hor)/pi-1  # 要把敌人骗进来杀
            reward_shoot -= np.clip(dist/40e3, 0, 1)

        if len(alive_ally_missiles)>0 and dist>40e3:
            reward_shoot = reward_shoot-30 if ut==1 else reward_shoot+10

        if terminate and ego.ammo == ego.init_ammo:
            reward_shoot -= 600 # 一发都不打必须重罚 100
        if terminate and ego.ammo < ego.init_ammo:
            reward_shoot += 20 # 至少打了一枚

        # 重复发射导弹时惩罚, 否则有奖励
        reward_SuoHa = 0
        if len(alive_ally_missiles)>1 and ut==1:
            reward_SuoHa -= 30
        if len(alive_ally_missiles)>1 and ut==0:
            reward_SuoHa += 30

        # miss 惩罚
        reward_miss = 0
        if enm.escape_once:
            reward_miss -= 60 # 10

        # 命中奖励
        end_reward = 0
        reward_event = 0
        if self.lose:
            reward_event = -300
            
        if self.win:
            reward_event = 300 + 200*(ego.init_ammo-ego.ammo)/ego.init_ammo  ## 赢了，导弹省得越多奖励越高 test 300
            end_reward = 300

        # 0.2? 0.02?
        reward = np.sum([
            # 1 * reward_base,
            # 1 * reward_AA,

            1 * reward_shoot,
            1 * reward_SuoHa,
            
            # 1 * reward_violate,
            1 * reward_miss,
            1 * reward_event,
        ])

        event_rewards = np.sum([
            1 * reward_miss,
            1 * reward_event,
        ])

        # guide_rewards = np.sum([
        #     1 * reward_shoot,
        #     1 * reward_SuoHa,
        # ])

        
        if terminate:
            self.running = False
    
        return terminate, reward, event_rewards


def shoot_action_shield(at, distance, alpha, AA_hor, launch_interval):
    at0 = at
    interval_refer = 5  # 8

    if distance > 40e3:
        interval_refer = 30

    # elif distance>40e3:
    #     interval_refer = 20
    # elif distance>20e3:
    #     interval_refer = 15
    # else:
    #     interval_refer = 8

    # todo 对方在我射界内向我发射了一枚导弹，我也应该向对方来一枚
    
    # # test
    # if abs(distance-55e3) < 1e3 and alt>8000:
    #     at = 1


    # if distance<30e3 and abs(AA_hor)>pi/2:
    #     at = 1

    # if distance>20e3:
    #     interval_refer = 15
    # elif distance>10e3:
    #     interval_refer = 10
    # elif distance < 10e3:
    #     interval_refer = 5
    
    if distance > 80e3 or alpha > 60*pi/180:
        at = 0
    # if distance < 10e3 and alpha < pi/12 and abs(AA_hor) > pi*3/4 and launch_interval>30:
    #     at = 1
    if launch_interval <= interval_refer:
        at = 0

    if abs(AA_hor) < pi*1/3 and distance>12e3: ## 禁止超视距完全尾追发射 新增
        at=0

    same = int(bool(at0) == bool(at))
    xor  = int(bool(at0) != bool(at))  

    # # debug
    # at = 1


    return at, xor
