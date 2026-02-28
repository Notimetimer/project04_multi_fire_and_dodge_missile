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
from math import *


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from Envs.battle6dof1v1_missile0919 import *


# 通过继承构建观测空间、奖励函数和终止条件

class EscapeTrainEnv(Battle):
    # # 任务：敌机一开始就发射导弹，规避的时候我机需要做置尾下高机动尽力逃脱追击
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.escape_key_order = [
            "locked_by_target",  # 1
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "threat",  # 4
            "border",  # 2
        ]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.last_dist2missile_dot = None
        self.last_delta_psi = None
        self.last_threat_delta_psi = None
        self.last_missile_speed_avg = None

    def escape_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.escape_key_order}
        # # 先对dict的元素mask
        # # 只需要 target_information 和 ego_main
        # full_obs["target_alive"] = copy.deepcopy(self.obs_init["target_alive"])
        # full_obs["missile_in_mid_term"] = copy.deepcopy(self.obs_init["missile_in_mid_term"])
        # full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        # full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])

        # 逃逸过程中会出现部分观测的情况，已在base_obs中写下规则
        # 只有在warning为TRUE的时候才能够获取威胁信息，已在get_state中写下规则
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.escape_key_order)
        return flat_obs, full_obs

    def escape_terminate_and_reward(self, side):  # 逃逸策略训练与奖励
        # copy了进攻的，还没改
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        cos_theta = state["ego_main"][3]
        sin_theta = state["ego_main"][2]
        theta = atan2(sin_theta, cos_theta)
        sin_phi = state["ego_main"][4]
        p = state["ego_control"][0]
        q = state["ego_control"][1]
        r = state["ego_control"][2]
        theta_v = state["ego_control"][3]
        alpha_air = state["ego_control"][5]

        cos_delta_psi = state["target_information"][0]
        sin_delta_psi = state["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]

        cos_delta_psi_threat = state["threat"][0]
        sin_delta_psi_threat = state["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)
        delta_theta_threat = state["threat"][2]
        threat_dist = state["threat"][3]
        fake_current_heading_ = np.array([cos(theta), sin(theta), 0])
        fake_L_threat_ = np.array([cos(delta_theta_threat)*cos_delta_psi_threat,
                                   sin(delta_theta_threat),
                                   cos(delta_theta_threat)*sin_delta_psi_threat])
        alpha_threat = np.arccos(np.dot(fake_L_threat_, fake_current_heading_) / (1 + 1e-5))
        p = state["ego_control"][0]
        sin_phi = state["ego_main"][4]
        cos_phi = state["ego_main"][5]
        phi = atan2(sin_phi, cos_phi)
        # cos_threat_psi, sin_threat_psi, threat_delta_theta, threat_distance =\
        #     state["threat"]

        RWR = state["warning"]
        obs = self.base_obs(side, reward_fn=1)
        d_hor = obs["border"][0]

        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            alive_enm_missiles = self.alive_b_missiles
        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles

        '''
        逃逸机动训练
        目标机从不可逃逸区外~40km向本机发射一枚导弹并对本机做纯追踪，
        本机被导弹命中有惩罚，除此之外根据和导弹的ATA和提供密集奖励
        '''
        self.close_range_kill()  # 加入近距杀

        # 被命中判为失败
        if ego.got_hit:
            terminate = True
            self.lose = 1

        # 高度出界失败
        if not self.min_alt <= alt <= self.max_alt:
            terminate = True
            self.lose = 1

        # 飞出水平边界失败
        if self.out_range(ego):
            terminate = True
            self.lose = 1

        # 导弹规避成功
        if self.t > self.game_time_limit \
                and not ego.dead and enm.ammo == 0 and \
                len(alive_enm_missiles) == 0:
            self.win = 1
            terminate = True

        if self.last_delta_psi is None:
            delta_psi_dot = 0
        else:
            delta_psi_dot = sub_of_radian(delta_psi, self.last_delta_psi) / self.dt_maneuver
        self.last_delta_psi = delta_psi

        if self.last_threat_delta_psi is None:
            delta_psi_threat_dot = 0
        else:
            delta_psi_threat_dot = sub_of_radian(delta_psi_threat, self.last_threat_delta_psi) / self.dt_maneuver
        self.last_threat_delta_psi = delta_psi_threat

        # 密集奖励
        if RWR:  # 存在雷达告警时, 躲导弹

            ### todo 远离导弹就交给距离的二阶导奖励，角度奖励只用来惩罚低高度俯冲行为

            cos_threat_delta_psi = state["threat"][0]
            acos(cos_threat_delta_psi)/pi
            r_angle = acos(cos_threat_delta_psi)/pi  # alpha_threat / pi
            if alpha >= 160 * pi / 180:  # abs(delta_psi) >= 3/4 * pi:
                r_angle -= alpha_air * 180 / pi / 5  # 10 ###
                r_angle -= abs(p) / (2 * pi / 2) * 2
                r_angle -= abs(delta_psi_threat_dot) / (2 * pi / 2) * 2

            # # # 垂直角度奖励，导弹相对飞机俯仰角>-30°时越低越好，否则应该水平规避
            
            r_angle_v = 0
            r_angle_v = 1 - abs((theta_v + 15 * pi / 180) / pi * 2)  # 3
            
            r_angle_v += 1  ###
            if alt > self.min_alt_safe + 1e3 and theta_v >= 0:
                r_angle_v -= theta_v / pi * 2 * 3
            if alt < self.min_alt_safe:
                r_angle_v = 1 - np.sqrt(abs(theta_v / pi * 2))

            # 下高奖励
            # ego.theta_v 在高空至少应该俯冲，但是俯冲角度超过-30°就不再额外奖励

            # sin_theta = state["ego_main"][2]
            # if -threat_delta_theta>-pi/6:
            #     sin_theta_opt = -1*np.clip((alt-self.min_alt_safe)/5000, -0.99, 0.99)
            # else:
            #     sin_theta_opt = 0
            # r_angle_v = (sin_theta<=sin_theta_opt)*(sin_theta-(-1))/(sin_theta_opt-(-1))+\
            #             (sin_theta>sin_theta_opt)*(1-(sin_theta-sin_theta_opt)/(1-sin_theta_opt))

            # 速度奖励，奖励和导弹之间距离的二阶导
            threat_dist_dot_avg = 0  # 没有考虑到多枚导弹的情况，只能想到用平均了
            L_threat_avg_ = np.zeros(3)
            for missile in alive_enm_missiles:
                L_m_ego_ = ego.pos_ - missile.pos_
                L_threat_avg_ += L_m_ego_/len(alive_enm_missiles) # 平均逃跑方向
                v_rel_as_m_ = ego.vel_ - missile.vel_
                dist2missile_dot = np.dot(L_m_ego_, v_rel_as_m_) / norm(L_m_ego_)
                threat_dist_dot_avg += dist2missile_dot / len(alive_enm_missiles)

            if self.last_dist2missile_dot is None:
                dist2m_dt2 = 0
            else:
                # dist2m_dt2 = (threat_dist_dot_avg - self.last_dist2missile_dot) / self.dt_maneuver # 距离变化率
                dist2m_dt2 = np.dot(ego.vel_, L_threat_avg_)/norm(L_threat_avg_) # 飞机自身速度对距离变化率的贡献
            self.last_dist2missile_dot = threat_dist_dot_avg
            r_v = np.clip(dist2m_dt2 / (9.8), -2, 2)

            # 过载量加入速度奖励
            if alpha_threat <= 175*pi/180:
                n_h = ego.Ny*sin(ego.phi)
                n_v = ego.Ny*cos(ego.phi)
                r_v += abs(n_h/6) * 2 * min(abs(alpha_threat-175*pi/180)/(pi/4), 1)
                r_v -= max(n_v, -3)/6
            else:
                r_v -= abs(ego.Ny/6) * 0.2
                
            # # 过载量加入速度奖励
            # if alpha_threat <= 160*pi/180:
            #     n_h = ego.Ny*sin(ego.phi)
            #     n_v = ego.Ny*cos(ego.phi)
            #     r_v += abs(n_h/6) * 2
            #     r_v -= max(n_v, -3)/6
            # else:
            #     r_v -= 0 # abs(ego.Ny/6) * 1

            # 高度奖励
            r_alt = (alt <= self.min_alt_safe + 1e3) * np.clip(ego.vu / 100, -1, 1) + \
                    (alt >= self.max_alt) * np.clip(-ego.vu / 100, -1, 1)
            # 导弹减速越快，奖励越大
            missile_speed_avg = 0
            for missile in alive_enm_missiles:
                missile_speed_avg += norm(missile.vel_)
            missile_speed_avg /= len(alive_enm_missiles)

            missile_speed_avg_dot = 0 if self.last_missile_speed_avg is None else\
                (missile_speed_avg-self.last_missile_speed_avg)/dt_maneuver

            self.last_missile_speed_avg = missile_speed_avg

            r_alt -= missile_speed_avg_dot/g /2 # 根据导弹平均减速度给奖励


            # 距离奖励，和导弹之间的距离变化率
            # dist2m = state["threat"][3]
            # r_dist = -1 + np.clip(dist2m / 30e3, -1, 1)
            r_dist = -threat_dist_dot_avg / (340*2)

        else:  # 躲飞机

            # 水平角度奖励， 奖励和敌机在同一高度层的置尾机动(√)

            r_angle = alpha / pi

            r_angle += 2  ###
            if alpha >= 160 * pi / 180:  # abs(delta_psi) >= 3/4 * pi:
                r_angle -= alpha_air * 180 / pi / 5  # 10 ###
                r_angle -= abs(p) / (2 * pi / 2) * 2
                r_angle -= abs(delta_psi_dot) / (2 * pi / 2) * 2

            r_angle_v = 1 - abs((theta_v + 3 * pi / 180) / pi * 2)
            r_angle_v += 1  ###
            if alt > self.min_alt_safe + 1e3 and theta_v >= 0:
                r_angle_v -= theta_v / pi * 2 * 3
            if alt < self.min_alt_safe:
                r_angle_v = 1 - np.sqrt(abs(theta_v / pi * 2))

            L_ = enm.pos_ - ego.pos_
            delta_v_ = enm.vel_ - ego.vel_
            dist_dot = np.dot(delta_v_, L_) / dist
            self.dist_dot = dist_dot
            # 速度奖励
            if self.last_dist_dot is None:
                dist_dot2 = 0
            else:
                dist_dot2 = (self.dist_dot - self.last_dist_dot) / self.dt_maneuver
            self.last_dist_dot = self.dist_dot
            r_v = dist_dot2 / 9.8
            # 高度奖励
            r_alt = (alt <= self.min_alt_safe + 1e3) * np.clip(ego.vu / 100, -1, 1) + \
                    (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)
            
            # 躲飞机时不要有太大的高度变化
            r_alt -= np.clip(abs(ego.vu) / 100, -1, 1)*1.5


            # 距离奖励，和目标机之间的距离 变化率
            # r_dist = -1 + np.clip(dist / 30e3, -1, 1)
            r_dist = - dist_dot / 340

        # # 水平边界奖励
        self.dhor = d_hor
        if self.last_dhor is None:
            d_hor_dot = 0
        else:
            d_hor_dot = (self.dhor - self.last_dhor) / self.dt_maneuver
        self.last_dhor = self.dhor
        r_border = d_hor_dot / 340 * 10e3 if d_hor*10e3 < 15e3 else 0

        # 稀疏奖励
        # 失败惩罚
        if self.lose:
            r_event = -30
        # 取胜奖励
        elif self.win:
            r_event = 30
        else:
            r_event = 0

        reward = np.sum([
            1 * r_angle,
            1 * r_angle_v,
            1 * r_v,
            2 * r_alt,
            1 * r_event,
            1 * r_border,  # 10
            0.5 * r_dist,
        ])

        if terminate:
            self.running = False

        return terminate, reward, r_event
