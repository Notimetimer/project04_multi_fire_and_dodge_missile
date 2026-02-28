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

import os

# # 在导入 scipy / 相关 Fortran/MKL 库之前禁用 Fortran 运行时的控制台处理器
# os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
# # 可选：限制线程数，减少并发 shutdown 问题
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# from scipy.interpolate import LinearNDInterpolator


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from Envs.battle6dof1v1_missile0919 import *


# 通过继承构建观测空间、奖励函数和终止条件

class CrankTrainEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.crank_key_order = [
            "target_locked",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "threat",  # 4
            "border",  # 2
        ]
        # # 1. 原始散点数据
        # # ----------------------------
        # x = np.array([-60, 50, 50, 50, 60, -180, 180, 50, 50, -180, -180, 180, 180, -60,-60])
        # y = np.array([0, -30, 0, 30, 0, 0, 0, -90, 90, 90,-90, 90,-90, -90, 90])
        # z = np.array([-1, -1, 1, -1, -1, -5, -5, -5, -5, -5,-5,-5,-5, -5, -5])
        # self.L_interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan) ###

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.last_delta_psi = None

    def crank_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.crank_key_order}
        
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.crank_key_order)
        return flat_obs, full_obs

    def left_crank_terminate_and_reward(self, side):  # 进攻策略训练与奖励
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        cos_delta_psi = state["target_information"][0]
        sin_delta_psi = state["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        p = state["ego_control"][0]
        # alpha = abs(delta_psi) # 实际上是把alpha换掉

        if side == 'r':
            ego = self.RUAV
            ego_missile = self.Rmissiles[0] if self.Rmissiles else None
            enm = self.BUAV
        if side == 'b':
            ego = self.BUAV
            ego_missile = self.Bmissiles[0] if self.Bmissiles else None
            enm = self.RUAV
        target_alt = enm.alt

        # 超时结束
        if self.t > self.game_time_limit:
            terminate = True
        # 雷达丢失目标判为失败 ###
        if alpha > ego.max_radar_angle:
            terminate = True
            self.lose = 1
        # 高度出界失败 ###
        if not self.min_alt <= alt:  # <= self.max_alt:
            terminate = True
            self.lose = 1

        # # 水平出界失败
        # if self.out_range(ego):
        #     terminate = True
        #     self.lose = 1

        # # 导弹命中目标成功
        # if enm.dead:
        #     terminate = True
        #     self.win = 1

        # 如果取得近距杀条件，判定为成功
        Los_ = enm.pos_ - ego.pos_
        dist = norm(Los_)
        # 求解hot-cold关系
        cos_ATA_ego = np.dot(Los_, ego.point_) / (dist * norm(ego.point_))
        # 近距杀
        if cos_ATA_ego >= cos(pi / 3) and dist < 8e3:
            terminate = True
            self.win = 1

        # 左crank角度奖励
        x = delta_psi * 180 / pi
        y = delta_theta * 180 / pi
        # r_angle = self.L_interp(x,y) ###
        r_angle = 4 * (x <= 50) * (x - 50) / 50 + (x > 50) * ((50 - x) / 10 - 10) - 5 * abs(y) / 20 + 120 / 60 + 90 / 20

        # 角速度惩罚
        if self.last_delta_psi is None:
            delta_psi_dot = 0
        else:
            delta_psi_dot = (delta_psi - self.last_delta_psi) / self.dt_maneuver
        self.last_delta_psi = delta_psi
        if abs(x) > 53:
            r_angle -= 3 * np.sign(x) * delta_psi_dot * 180 / pi * self.dt_maneuver / 4

        
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
        # 滚转角速度惩罚
        r_angle -= 0.15 * abs(p)*180/pi / 20 # 20°每秒已经很快了

        # # 垂直角度惩罚
        # q_epsilon = atan2(Los_[1], sqrt(Los_[0]**2+Los_[2]**2))
        r_angle_v = 0  # -abs(ego.theta-q_epsilon)/pi*2
        ### 
        r_angle_v -= abs(np.clip(ego.vu / 100, -1, 1)) * 0.5
        # 上升下降率惩罚
        if Los_[1] < 0:  # 目标在下面，我应该下降
            r_angle_v -= np.clip(ego.vu / 100, -1, 1)
        if Los_[1] > 0:  # 目标在上面，我应该上升
            r_angle_v += np.clip(ego.vu / 100, -1, 1)

        # 高度奖励
        pre_alt_opt = target_alt - 2e3  # 比目标低1000m方便增加阻力
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)
        r_alt = (alt <= alt_opt) * (alt - self.min_alt) / (alt_opt - self.min_alt) + \
                (alt > alt_opt) * (1 - (alt - alt_opt) / (self.max_alt - alt_opt))

        r_alt += (alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                 (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)

        # 速度奖励
        speed_opt = 0.95 * 340
        r_speed = - abs(speed - speed_opt) / (2 * 340)

        # # 边界距离奖励 ###
        obs = self.base_obs(side, reward_fn=1)
        d_hor = obs["border"][0]

        r_border = d_hor

        # 事件奖励
        r_event = 0

        if self.lose:
            r_event -= 30  # 20 100 50 70
        if self.win:
            r_event += 30  # 20 100 50 70

        # if alpha > ego.max_radar_angle:
        #     r_event -= 3 # 超出雷达范围惩罚

        reward = np.sum([
            1 * r_angle,
            2 * r_angle_v,  # 3
            1 * r_alt,
            1 * r_speed,
            1 * r_event,
            0.5 * r_border,
        ])

        if terminate:
            self.running = False

        return terminate, reward, r_event

    def right_crank_terminate_and_reward(self, side):
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        cos_delta_psi = state["target_information"][0]
        sin_delta_psi = state["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        p = state["ego_control"][0]
        # alpha = abs(delta_psi) # 实际上是把alpha换掉

        if side == 'r':
            ego = self.RUAV
            ego_missile = self.Rmissiles[0] if self.Rmissiles else None
            enm = self.BUAV
        if side == 'b':
            ego = self.BUAV
            ego_missile = self.Bmissiles[0] if self.Bmissiles else None
            enm = self.RUAV
        target_alt = enm.alt

        # 超时结束
        if self.t > self.game_time_limit:
            terminate = True
        # 雷达丢失目标判为失败 ###
        if alpha > ego.max_radar_angle:
            terminate = True
            self.lose = 1
        # 高度出界失败 ###
        if not self.min_alt <= alt:  # <= self.max_alt:
            terminate = True
            self.lose = 1

        # 如果取得近距杀条件，判定为成功
        Los_ = enm.pos_ - ego.pos_
        dist = norm(Los_)
        # 求解hot-cold关系
        cos_ATA_ego = np.dot(Los_, ego.point_) / (dist * norm(ego.point_))
        # 近距杀
        if cos_ATA_ego >= cos(pi / 3) and dist < 8e3:
            terminate = True
            self.win = 1

        # 右crank角度奖励
        x = delta_psi * 180 / pi
        y = delta_theta * 180 / pi
        # r_angle = self.L_interp(x,y) ###
        r_angle = -1 * (x <= -50) * ((-50-x) / 10 + 10) - 4 * (x > -50) * ((x - -50) / 50) - 5 * abs(y) / 20 + 120 / 60 + 90 / 20

        # 角速度惩罚
        if self.last_delta_psi is None:
            delta_psi_dot = 0
        else:
            delta_psi_dot = (delta_psi - self.last_delta_psi) / self.dt_maneuver
        self.last_delta_psi = delta_psi
        if abs(x) > 53:
            r_angle -= 3 * np.sign(x) * delta_psi_dot * 180 / pi * self.dt_maneuver / 4

        
        sin_phi = state["ego_main"][4]
        cos_phi = state["ego_main"][5]
        phi = atan2(sin_phi, cos_phi)
        # 滚转角惩罚
        r_angle -= 0.2 * abs(phi / pi) # 禁止倒飞
        # 负过载惩罚
        if ego.Ny<0:
            r_angle -= 0.1 * abs(ego.Ny) / 2
        # 侧滑角惩罚
        r_angle -= 0.05 * np.clip(abs(ego.beta_air*180/pi / 5), 0, 1)
        # 迎角惩罚
        r_angle -= 0.01 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))
        # 滚转角速度惩罚
        r_angle -= 0.15 * abs(p)*180/pi / 20 # 20°每秒已经很快了

        # # 垂直角度惩罚
        # q_epsilon = atan2(Los_[1], sqrt(Los_[0]**2+Los_[2]**2))
        r_angle_v = 0  # -abs(ego.theta-q_epsilon)/pi*2
        ### 
        r_angle_v -= abs(np.clip(ego.vu / 100, -1, 1)) * 0.5
        # 上升下降率惩罚
        if Los_[1] < 0:  # 目标在下面，我应该下降
            r_angle_v -= np.clip(ego.vu / 100, -1, 1)
        if Los_[1] > 0:  # 目标在上面，我应该上升
            r_angle_v += np.clip(ego.vu / 100, -1, 1)

        # 高度奖励
        pre_alt_opt = target_alt - 2e3  # 比目标低1000m方便增加阻力
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)
        r_alt = (alt <= alt_opt) * (alt - self.min_alt) / (alt_opt - self.min_alt) + \
                (alt > alt_opt) * (1 - (alt - alt_opt) / (self.max_alt - alt_opt))

        r_alt += (alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                 (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)

        # 速度奖励
        speed_opt = 0.95 * 340
        r_speed = - abs(speed - speed_opt) / (2 * 340)

        # # 边界距离奖励 ###
        obs = self.base_obs(side, reward_fn=1)
        d_hor = obs["border"][0]
        # # # 水平边界奖励
        # self.dhor = d_hor
        # if self.last_dhor is None:
        #     d_hor_dot = 0
        # else:
        #     d_hor_dot = (self.dhor - self.last_dhor) / self.dt_maneuver
        # self.last_dhor = self.dhor
        # r_border = d_hor_dot / 340 * 10e3
        r_border = d_hor

        # 事件奖励
        r_event = 0
        # if ego_missile is not None:
        #     # A-pole奖励
        #     if ego_missile.A_pole_moment:
        #         r_event += dist / 30e3 * 20
        #         r_event += 2 * (self.max_alt-alt)/(self.max_alt-self.min_alt)
        #     # F-pole奖励
        #     if ego_missile.hit:
        #         r_event += dist / 30e3 * 40
        #         r_event += alt
        #         r_event += 2 * (self.max_alt-alt)/(self.max_alt-self.min_alt)
        if self.lose:
            r_event -= 30  # 20 100 50 70
        if self.win:
            r_event += 30  # 20 100 50 70

        # if alpha > ego.max_radar_angle:
        #     r_event -= 3 # 超出雷达范围惩罚

        reward = np.sum([
            1 * r_angle,
            2 * r_angle_v,  # 3
            1 * r_alt,
            1 * r_speed,
            1 * r_event,
            0.5 * r_border,
        ])

        if terminate:
            self.running = False

        return terminate, reward, r_event


