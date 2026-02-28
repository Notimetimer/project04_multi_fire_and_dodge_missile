'''
下一次更改需要加上导弹发射相关的奖励和惩罚
'''

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
from math import *


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from Envs.battle6dof1v1_missile0919 import *
from Envs.Tasks.AttackManeuverEnv import *
from Envs.Tasks.CrankManeuverEnv import *
from Envs.Tasks.EscapeManeuverEnv import *
from Algorithms.Rules import *


# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

action_options = {
                    0: "track",
                    1: "30track",
                    2: "60track",
                    3: "-30track",
                    4: "-60track",
                    5: "+-30crank",
                    6: "+-60crank",
                    7: "snake",
                    8: "splitS",
                    9: "39",
                    10: "slowTurn",
                    11: "fastTurn",
                    12: "-30turn",
                    13: "-60turn",
                }

class ChooseStrategyEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=0):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        self.last_obs = None
        self.pomdp = pomdp     

    def maneuver14(self, UAV, action):
        # 输入动作与动力运动学状态
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        # state = self.get_state(UAV.side)
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])

        move_action = np.zeros(3)

        # 水平跟踪
        if action == 0:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 0
            speed_cmd = 400

        # 30°爬升加速
        if action == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 2500
            speed_cmd = 400

        # 60°爬升加速
        if action == 2:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000
            speed_cmd = 400

        # -30°俯冲跟踪
        if action == 3:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # -60°俯冲跟踪
        if action == 4:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400

        # ±30°水平偏移
        if action == 5:
            delta_psi_cmd = delta_psi - np.sign(delta_psi)*pi/6
            delta_height_cmd = 0
            speed_cmd = 350

        # ±60°水平偏移
        if action == 6:
            delta_psi_cmd = delta_psi - np.sign(delta_psi) * 55 * pi/180
            delta_height_cmd = 0
            speed_cmd = 350

        # 水平蛇形机动
        if action == 7:
            if delta_psi > 50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            if delta_psi < -50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            elif UAV.phi>=0:
                delta_psi_cmd = sub_of_radian(delta_psi+pi/3, 0)
            else:
                delta_psi_cmd = sub_of_radian(delta_psi-pi/3, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 破s
        if action == 8:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi, pi)
            delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
            speed_cmd = 300

        # 水平三九线机动
        if action == 9:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = delta_psi_temp - np.sign(delta_psi)*pi/2
            delta_height_cmd = 0
            speed_cmd = 400

        # 水平慢置尾
        if action == 10:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = -np.sign(delta_psi)*np.clip((1-abs(delta_psi_temp)/pi) * 2, 0, 1) * 10*pi/180
            delta_height_cmd = 0
            speed_cmd = 600

        # 水平快置尾
        if action == 11:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
            speed_cmd = 400

        # 水平快置尾后-30°俯冲
        if action == 12:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # 水平快置尾后-60°俯冲
        if action == 13:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])
    

    def combat_terminate_and_reward(self, side, action_label):
        terminate = self.get_terminate()
        done = terminate

        # todo 奖励函数调用或是重写都要在这实现
        self.update_missile_state()
        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            alive_enm_missiles = self.alive_b_missiles
            alive_ally_missiles = self.alive_r_missiles
        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_ally_missiles = self.alive_b_missiles

        # 如果对手和对手的所有导弹都没了，且己方飞机还在，判定为胜
        if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
            self.win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方飞机还在，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            self.lose = 1
            done = 1
        # 如果友方和敌方打光导弹且都存活，或双方飞机都没了，判定为平
        elif ego.ammo == 0 and enm.ammo == 0 and not ego.dead and not enm.dead or \
                ego.dead and enm.dead:
            self.draw = 1
            done = 1
        else:
            done = 0

        if self.t > self.game_time_limit:
            done = 1
            # 如果超时，我方打光导弹，导弹全自爆，对手导弹还有剩，且存活，判定为负
            if ego.ammo + len(alive_ally_missiles) == 0 and \
                enm.ammo + len(alive_enm_missiles) > 0 and not enm.dead:
                self.lose = 1

            # 如果超时，对手打光导弹，导弹全自爆，我方导弹还有剩，且存活，判定为胜
            elif enm.ammo + len(alive_enm_missiles) == 0 and \
                ego.ammo + len(alive_ally_missiles) > 0 and not ego.dead:
                self.win = 1                

            # 如果超时，双方均未打光导弹/仍有导弹在空中飞，且双方均存活，判定为平
            else:
                self.draw = 1

        reward = 0
        if self.win:
            reward += 300
        if self.lose:
            reward -= 300
        if self.draw:
            reward -= 0

        uav_states = self.get_state(side)  ### test 部分观测的话用1
        enm_state = self.get_state(enm.side)  ### test 部分观测的话用1
        delta_theta = uav_states["target_information"][2]
        distance = uav_states["target_information"][3]
        d_hor, leftright = uav_states["border"]
        # state = self.get_state(UAV.side)
        speed = uav_states["ego_main"][0]
        alt = uav_states["ego_main"][1]
        cos_delta_psi = uav_states["target_information"][0]
        sin_delta_psi = uav_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = uav_states["target_information"][4]
        warning = uav_states["warning"]
        missile_in_mid_term = uav_states["missile_in_mid_term"]
        missile_time_since_shoot = uav_states["weapon"]
        AA_hor = uav_states["target_information"][-2]
        
        cos_delta_psi_threat = uav_states["threat"][0]
        sin_delta_psi_threat = uav_states["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)

        # 事件/密集奖励
        # 为导弹提供制导
        if missile_in_mid_term:
            reward += 2

        # 锁定目标
        if uav_states["target_locked"]:
            reward += 1.1

        # 被目标锁定
        if uav_states["locked_by_target"]:
            reward -= 1

        # 收到导弹警告
        if warning:
            reward -= 3

        # 导弹锁定目标
        if enm_state["warning"]:
            reward += 3

        # 逃脱导弹
        if ego.escape_once:
            reward += 10

        # 导弹被逃脱
        if enm.escape_once:
            reward -= 5

        event_reward = reward

        # # 密集奖励
        if warning and action_label != 1 and len(alive_ally_missiles)>0:
            reward -= 5

        # 没发射导弹时alpha越小越好
        if len(alive_ally_missiles) == 0 and not warning:
            reward += 0.5 - alpha / pi
        
        # 导弹处于中制导阶段时alpha在±60*pi/180之间为奖励高台， 其余随alpha线性减少
        if missile_in_mid_term:
            # 目标没有跑，不该重复攻击
            if action_label==0 and abs(AA_hor)>pi*2/3:
                reward -= 5

            # 高台奖励：alpha在[-60*pi/180, 60*pi/180]区间奖励高，其余线性递减
            if abs(alpha) < np.pi / 3:
                reward += 1
            else:
                reward += 1 - (alpha - pi / 3) / pi

        # 有warning时alpha越大越好
        if warning:
            reward += 8 * abs(delta_psi_threat) / pi # 这里的alpha错了，应该是和导弹的threa_delta_psi有关的
        

        # 角度奖励
        r_angle = 0
        # 侧滑角惩罚
        r_angle -= 0.05 * np.clip(abs(ego.beta_air*180/pi / 5), 0, 1)
        # 迎角惩罚
        r_angle -= 0.01 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))

        # 高度限制奖励/惩罚
        r_alt = (alt <= self.min_alt_safe + 1e3) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)

        reward = reward + r_angle + r_alt
        
        # deltapsi变化率奖励 todobedontinued

        # 态势优势度 tobecontinued


        return done, reward, event_reward

    def is_action_complete(self, side, action_label):
        # if side == 'r':
        #     ego = self.RUAV
        #     enm = self.BUAV
        # if side == 'b':
        #     ego = self.BUAV
        #     enm = self.RUAV

        state = self.get_state(side)
        cos_delta_psi = state["target_information"][0]
        sin_delta_psi = state["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_theta = state["target_information"][2]
        alpha = state["target_information"][4]
        RWR = state["warning"]
        cos_delta_psi_threat = state["threat"][0]
        sin_delta_psi_threat = state["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)

        action = action_options[action_label]

        action_done = False
        if action == 'attack':
            if alpha < pi/6:
                action_done = True
        
        if action == 'escape':
            if abs(delta_psi_threat) > pi*5/6 and RWR or\
                  abs(alpha) > pi*5/6 and not RWR:
                action_done = True
        
        if action == 'left':
            if abs(delta_psi*180/pi - 50) < 10:
                action_done = True
        
        if action == 'right':
            if abs(delta_psi*180/pi + 50) < 10:
                action_done = True
            
        return action_done
        
        
