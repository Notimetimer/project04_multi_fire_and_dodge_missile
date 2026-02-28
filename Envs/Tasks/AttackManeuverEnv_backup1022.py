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
    # 进攻策略观测量
    def attack_obs(self, side):
        full_obs = self.base_obs(side)
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])
        full_obs["threat"] = copy.deepcopy(self.obs_init["threat"])
        full_obs["border"] = copy.deepcopy(self.obs_init["border"])


        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs
    
    def attack_terminate_and_reward(self, side): # 进攻策略训练与奖励
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        target_alt = alt+state["target_information"][0]
        delta_psi = state["target_information"][1]
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]

        if side == 'r':
            uav = self.RUAV
            enm = self.BUAV
            current_action = self.r_actions
            
        if side == 'b':
            uav = self.BUAV
            enm = self.RUAV
            current_action = self.b_actions

        # 结束判断：超时/损毁
        if self.t > self.game_time_limit:
            terminate = True
        # if alpha > pi/2 and self.t > self.game_time_limit: # 超时了还没hot就结束
        #     terminate = True
        #     self.lose = 1
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.lose = 1

        if dist<5e3 and alpha< pi/12:
            terminate = True
            self.win = 1

        # 角度奖励
        r_angle = 1-alpha/(60*pi/180)  # 超出雷达范围就惩罚狠一点

        # 高度奖励
        pre_alt_opt = target_alt + np.clip((dist-10e3)/(40e3-10e3)*5e3, 0, 5e3)
        alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)

        r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
                    (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
                
        # 速度奖励
        speed_opt = 1.5*340
        r_speed = abs(speed-speed_opt)/(2*340)

        # 距离奖励
        r_dist = (dist<=10e3)*(dist-0)/(10e3-0)+\
                    (dist>10e3)*(1-(dist-10e3)/(50e3-10e3))

        # 平稳性惩罚，debug 有错误，一直是0
        # delta_acts_ = np.array(state["ego_control"][0:2+1]) # 历史动作 current_action-np.array(state["ego_control"][0:2+1])
        # delta_acts_norm_ = delta_acts_/2/pi # pqr -abs(uav.p)/(2*pi) 历史动作 delta_acts_ * np.array([1/5000, 1/pi, 1/340])
        # r_steady = - norm(delta_acts_norm_) 

        r_steady = 0

        # 事件奖励
        reward_event = 0
        if self.lose:
            reward_event = -1
        if self.win:
            reward_event = 1
        
        # 0.2? 0.02?
        reward = np.sum(np.array([2,1,1,1,5,0.02])*\
            np.array([r_angle, r_alt, r_speed, r_dist, reward_event, r_steady]))

        if terminate:
            self.running = False
        
        reward_for_show = np.array([r_angle, r_alt, r_speed, r_dist, reward_event, r_steady])

        return terminate, reward, reward_for_show