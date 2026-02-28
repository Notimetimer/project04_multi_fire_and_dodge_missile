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
# 在导入 scipy / 相关 Fortran/MKL 库之前禁用 Fortran 运行时的控制台处理器
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
# 可选：限制线程数，减少并发 shutdown 问题
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from scipy.interpolate import LinearNDInterpolator

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

    def crank_obs(self, side):
        full_obs = self.base_obs(side)
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["target_locked"] = copy.deepcopy(self.obs_init["target_locked"])
        full_obs["missile_in_mid_term"] = copy.deepcopy(self.obs_init["missile_in_mid_term"])
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["threat"] = copy.deepcopy(self.obs_init["threat"])
        
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs

    def left_crank_terminate_and_reward(self, side): # 进攻策略训练与奖励
        # copy了进攻的，还没改
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        target_alt = alt+state["target_information"][0]
        delta_psi = state["target_information"][1]
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        # alpha = abs(delta_psi) # 实际上是把alpha换掉

        if side == 'r':
            ego = self.RUAV
            ego_missile = self.Rmissiles[0] if self.Rmissiles else None
            enm = self.BUAV
        if side == 'b':
            ego = self.BUAV
            ego_missile = self.Bmissiles[0] if self.Bmissiles else None
            enm = self.RUAV

        # 超时结束
        if self.t > self.game_time_limit:
            terminate = True
        # 雷达丢失目标判为失败 ###
        if alpha > ego.max_radar_angle:
        # if alpha > pi/2: ### 放宽要求
            terminate = True
            self.lose = 1
        # 高度出界失败
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.lose = 1
        # 水平出界失败
        if self.out_range(ego):
            terminate = True
            self.lose = 1

        # # 导弹命中目标成功
        # if enm.dead:
        #     terminate = True
        #     self.win = 1

        # 如果取得近距杀条件，判定为成功
        Los_ = enm.pos_ - ego.pos_
        dist = norm(Los_)
        # 求解hot-cold关系
        cos_ATA_ego = np.dot(Los_, ego.point_)/(dist*norm(ego.point_))
        # 近距杀
        if cos_ATA_ego>=cos(60*pi/180) and dist<8e3:
            terminate = True
            self.win = 1
        
        # 左crank角度奖励
        x = delta_psi*180/pi
        y = delta_theta*180/pi
        # r_angle = self.L_interp(x,y) ###
        r_angle = sub_of_degree(x, 50)/50 -y/20 + 120/60 + 90/20

        # x = np.sign(delta_psi)*alpha * 180/pi
        # alpha_max = ego.max_radar_angle*180/pi # 60
        # x_opt = 50
        # r_angle_delta_psi = np.clip(delta_psi/(60*pi/180), -1, 1)
        # if alpha < 50*pi/180:
        #     r_angle_alpha = -(alpha/(50*pi/180))**2
        # else:
        #     r_angle_alpha = -(6/5)*(alpha/(60*pi/180))**2

        # r_angle = r_angle_alpha + 0.8*r_angle_delta_psi

        # temp = (delta_psi<x_opt)*(delta_psi+alpha_max)/(x_opt+alpha_max)+(delta_psi>=x_opt)*(delta_psi-(alpha_max-5))/(x_opt-(alpha_max-5))
        # temp = temp*np.clip(1-abs(ego.theta)/pi*6, 0, 1)
        # theta_threshold = pi/12
        # if abs(ego.theta)<theta_threshold:
        #     r_angle = temp * (theta_threshold-abs(ego.theta))/theta_threshold
        # else:
        #     r_angle = -(abs(ego.theta)-theta_threshold)/theta_threshold
        
        # mid_switch = sigmoid(0.4 * (x + alpha_max)) * sigmoid(0.4 * (alpha_max - x))
        # r_angle = (x/alpha_max * mid_switch - (1 - mid_switch))
        # if alpha > ego.max_radar_angle:
        #     r_angle -= 20

        # # 垂直角度惩罚
        # q_epsilon = atan2(Los_[1], sqrt(Los_[0]**2+Los_[2]**2))
        r_angle_v = 0 # -abs(ego.theta-q_epsilon)/pi*2
        ### 
        r_angle_v -= abs(np.clip(ego.vu/100, -1, 1)) * 0.5
        # 上升下降率惩罚
        if Los_[1]<0: # 目标在下面，我应该下降
            r_angle_v -= np.clip(ego.vu/100, -1, 1)
        if Los_[1]>0: # 目标在上面，我应该上升
            r_angle_v += np.clip(ego.vu/100, -1, 1)

        # 高度奖励
        pre_alt_opt = target_alt - 2e3 # 比目标低1000m方便增加阻力
        alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)
        r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
                    (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
        # if not self.min_alt<=alt<=self.max_alt:
        #     r_alt -= 20
        ###
        r_alt += (alt<=self.min_alt_save) * np.clip(ego.vu/100, -1, 1) + \
                (alt>=self.max_alt_save) * np.clip(-ego.vu/100, -1, 1)
                
        # 速度奖励
        speed_opt = 0.95*340
        r_speed = abs(speed-speed_opt)/(2*340)

        # 边界距离奖励
        obs = self.base_obs(side, reward_fn=1)
        d_hor = obs["border"][0]
        # # 水平边界奖励
        self.dhor = d_hor
        if self.last_dhor is None:
            d_hor_dot = 0
        else:
            d_hor_dot = (self.dhor-self.last_dhor)/self.dt_maneuver
        self.last_dhor = self.dhor
        r_border = d_hor_dot /340*50e3
        # r_border = 0

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
            r_event -= 70 # 20 100 50
        if self.win:
            r_event += 70 # 20 100 50

        # if alpha > ego.max_radar_angle:
        #     r_event -= 3 # 超出雷达范围惩罚

        reward = np.sum([
            1 * r_angle,
            2 * r_angle_v, # 3
            1 * r_alt,
            1 * r_speed,
            1 * r_event,
            0.5 * r_border,
            ])

        if terminate:
            self.running = False
        
        return terminate, reward, r_event

    # def right_crank_terminate_and_reward(self, side): # 进攻策略训练与奖励
    