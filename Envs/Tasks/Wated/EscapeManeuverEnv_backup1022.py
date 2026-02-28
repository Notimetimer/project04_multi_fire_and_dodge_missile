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
    
    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):       
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.last_dist2missile_dot = None


    def escape_obs(self, side):
        full_obs = self.base_obs(side)
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["target_alive"] = copy.deepcopy(self.obs_init["target_alive"])
        full_obs["missile_in_mid_term"] = copy.deepcopy(self.obs_init["missile_in_mid_term"])
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])
        
        # 逃逸过程中会出现部分观测的情况，已在base_obs中写下规则
        # 只有在warning为TRUE的时候才能够获取威胁信息，已在get_state中写下规则
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs
    
    def escape_terminate_and_reward(self, side):  # 逃逸策略训练与奖励
        # copy了进攻的，还没改
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        target_alt = alt+state["target_information"][0]
        delta_psi = state["target_information"][1]
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        # alpha = state["target_information"][4]
        alpha = abs(delta_psi) # 实际上是把alpha换掉
        # threat_delta_psi, threat_delta_theta, threat_distance =\
        #     state["threat"]

        RWR = state["warning"]
        obs = self.base_obs(side)
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
        self.close_range_kill() # 加入近距杀

        # 被命中判为失败
        if ego.got_hit:
            terminate = True
            self.lose = 1

        # 高度出界失败
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.lose = 1

        # 飞出水平边界失败
        if self.out_range(ego):
            terminate = True
            self.lose = 1
        
        # 导弹规避成功
        if self.t > self.game_time_limit \
            and not ego.dead and enm.ammo==0 and \
                len(alive_enm_missiles)==0:
            self.win = 1
            terminate = True
        
        # 密集奖励
        if RWR:  # 存在雷达告警时, 躲导弹

            ### todo 远离导弹就交给距离的二阶导奖励，角度奖励只用来惩罚低高度俯冲行为

            # # 角度奖励， 奖励置尾机动， 因为状态信息被投影到了另一边，现在变成追踪训练了
            # L_anti_missile_ = np.array([cos(threat_delta_theta)*cos(threat_delta_psi),
            #                            sin(threat_delta_theta),
            #                            cos(threat_delta_theta)*sin(threat_delta_psi)])
            # L_only_theta_ = np.array([cos(ego.theta), sin(ego.theta), 0])
            # # L_anti_missile_ 前正号表示导弹指向飞机，负号即飞机指向导弹
            # cos_alpha_anti_missile = np.dot(-L_anti_missile_, L_only_theta_)

            r_angle = 0  # cos_alpha_anti_missile

            # # # 垂直角度奖励，导弹相对飞机俯仰角>-30°时越低越好，否则应该水平规避
            # if ego.alt > (self.min_alt_save + self.max_alt_save)/2 and -threat_delta_theta>-pi/6:
            #     r_angle_v = -sin(ego.theta)
            # elif ego.alt > self.min_alt_save:
            #     theta_opt = (ego.alt-self.min_alt_save)/((self.max_alt_save-self.min_alt_save)/2)*-pi/2
            #     r_angle_v = cos(ego.theta-theta_opt)
            # else:
            #     r_angle_v = cos(ego.theta)
            r_angle_v = 0
            if ego.alt <= self.min_alt_save and ego.theta < 0:
                r_angle_v -= abs(ego.theta/pi*2)
            if ego.alt >= self.max_alt_save and ego.theta > 0:
                r_angle_v -= abs(ego.theta / pi * 2)

            # sin_theta = state["ego_main"][2]
            # if -threat_delta_theta>-pi/6:
            #     sin_theta_opt = -1*np.clip((alt-self.min_alt_save)/5000, -0.99, 0.99)
            # else:
            #     sin_theta_opt = 0
            # r_angle_v = (sin_theta<=sin_theta_opt)*(sin_theta-(-1))/(sin_theta_opt-(-1))+\
            #             (sin_theta>sin_theta_opt)*(1-(sin_theta-sin_theta_opt)/(1-sin_theta_opt))
            
            # 速度奖励，奖励和导弹之间距离的二阶导
            temp = 0 # 没有考虑到多枚导弹的情况，只能想到用平均了
            for missile in alive_enm_missiles:
                L_m_ego_ = ego.pos_-missile.pos_
                v_rel_as_m_ = ego.vel_-missile.vel_
                dist2missile_dot = np.dot(L_m_ego_, v_rel_as_m_)/norm(L_m_ego_)
                temp += dist2missile_dot/len(alive_enm_missiles)

            if self.last_dist2missile_dot is None:
                dist2m_dt2 = 0
            else:
                dist2m_dt2 = (temp - self.last_dist2missile_dot)/self.dt_maneuver
            self.last_dist2missile_dot = temp
            r_v = np.clip(dist2m_dt2/(9.8), -2, 2)
            
            # 高度奖励
            r_alt = (alt<=self.min_alt_save) * np.clip(ego.vu/100, -1, 1) + \
                (alt>=self.max_alt_save) * np.clip(-ego.vu/100, -1, 1)

            # 距离奖励，和导弹之间的距离
            dist2m = norm(state["threat"])
            r_dist = -1+np.clip(dist2m/30e3, -1, 1)

        else:  # 躲飞机
            ###
            r_angle = alpha / pi
            r_angle_v = 1-abs(ego.theta/pi*2) - abs(delta_theta/pi*2)
            L_ = enm.pos_-ego.pos_
            delta_v_ = enm.vel_-ego.vel_
            dist_dot = np.dot(delta_v_, L_)/dist
            self.dist_dot = dist_dot
            # 速度奖励
            if self.last_dist_dot is None:
                dist_dot2 = 0
            else:
                dist_dot2 = (self.dist_dot-self.last_dist_dot)/self.dt_maneuver
            self.last_dist_dot = self.dist_dot
            r_v = dist_dot2/9.8
            # 高度奖励
            r_alt = (alt<=self.min_alt_save) * np.clip(ego.vu/100, -1, 1) + \
                (alt>=self.max_alt_save) * np.clip(-ego.vu/100, -1, 1)
            # 距离奖励，和目标机之间的距离
            r_dist = -1+np.clip(dist/30e3, -1, 1)

            # 原有
            # # 水平角度奖励， 奖励和敌机在同一高度层的 三九机动(×) 置尾机动(√)
            # # r_angle_h = abs(abs(threat_delta_psi)-pi/2)/pi*2
            # r_angle_h = 1-abs(delta_psi)/pi
            # r_angle_v = abs(ego.theta + obs["target_information"][2])/pi
            # # r_angle_h = abs(threat_delta_psi)/pi
            # # r_angle_v = abs(ego.theta - obs["target_information"][2])/pi
            # r_v = 1 - np.abs(speed-0.95*340)/(2*340)
            # pre_alt_opt = 5e3*obs["target_information"][0] # 和目标相同高度
            # alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)
            # r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
            #             (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
            # # 距离奖励，和目标机之间的距离
            # r_dist = -1+np.clip(dist/30e3, -1, 1)

        # 原有
        # # 速度奖励
        # temp = abs(threat_delta_psi)/pi # 远离度,对头时候最好是0.8Ma，置尾的时候越快越好
        # v_opt = (0.8+(2-0.8)*temp)*340
        # r_v = 1 - np.abs(speed-v_opt)/(2*340)

        # # 高度奖励
        # pre_alt_opt = self.min_alt_save + 1e3 # 比最小安全高度高1000m
        # alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)
        # r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
        #             (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
        
        # # 距离奖励，和目标机之间的距离
        # r_dist = -1+np.clip(dist/30e3, -1, 1)

        # # 水平边界奖励
        self.dhor = d_hor
        if self.last_dhor is None:
            d_hor_dot = 0
        else:
            d_hor_dot = (self.dhor-self.last_dhor)/self.dt_maneuver
        self.last_dhor = self.dhor
        r_border = d_hor_dot /340*50e3

        # 稀疏奖励
        # 失败惩罚
        if self.lose:
            r_event = -20
        # 取胜奖励
        elif self.win:
            r_event = 20
        else:
            r_event = 0

        w_angle =  1 # d_hor**2
        w_border = 0.5 # 1-w_angle
        reward = np.sum([
            w_angle * r_angle ,
            1 * r_angle_v ,
            1 * r_v ,
            2 * r_alt ,
            1 * r_event ,
            w_border * r_border , # 10
            0.5 * r_dist,
            ])
        
        # reward = np.sum(\
        #     np.array([1, 0.5, 1, 1, 1, 2, 0.5])*\
        #     np.array([r_angle_h, r_angle_v, r_v, r_alt, r_event, r_border, r_dist]))

        if terminate:
            self.running = False
        
        return terminate, reward, r_event