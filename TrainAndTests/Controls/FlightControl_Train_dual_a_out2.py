'''
增加过载量的可见性
# 训练不稳定 三问:
    # 1、也许是常数的惩罚项太大导致了不稳定？
    # 2、critic loss 要使用hubber loss代替mse？
    # 3、学习率需要动态衰减？
'''

use_tacview = 0

import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
from gym import spaces
import copy
import matplotlib.pyplot as plt
import json
import glob
import argparse
import time  # 确保引入 time 模块
from datetime import datetime

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from _context import *
from Envs.UAVmodel6d0309 import UAVModel  # UAVmodel6d
from Visualize.tacview_visualize2 import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOHybrid23_0_distil2_one_step_KL import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from TrainAndTests.Controls.UPolicyWrapper import *

class track_env():
    def __init__(self, dt_move=0.02, tacview_show=0, time_limit=3*60):
        super(track_env, self).__init__()
        self.realistic = 1
        self.RUAV_ids = None
        self.dt_report = None
        self.dt_move = dt_move
        self.t = None
        self.fail = None # 负
        self.crash = 0
        self.stall = 0
        self.break_up = 0
        
        self.action_space = [spaces.Box(low=-1, high=+1, shape=(4,), dtype=np.float32)]
        self.DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, 0.0]),
                               'psi': 0
                               }
        
        self.time_limit = time_limit
        self.min_alt = 0.1e3
        self.min_alt_safe = 1e3

        self.max_alt_safe = 13e3
        self.max_alt = 15e3

        self.flight_key_order = [
            "ego_main",  # 7
            "ego_control",  # 7
            "flight_cmd", # 4
        ]
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
        self.AO = 0
        self.v_error = 0
        self.psi_error = 0
        self.theta_error = 0
        self.uav_pos_ = np.zeros(3)
        self.target_pos_ = np.zeros(3)
        self.theta_req = 0
    
    def reset(self, o00=None, birth_state=None, height_req=8e3, psi_req=0, v_req=340, dt_report=0.2, t0=0):
        self.t = t0
        self.fail = 0
        self.crash = 0
        self.stall = 0
        self.break_up = 0
        
        if o00 == None:
            o00 = np.array([118, 30])  # 地理原点的经纬
            self.o00 = o00
        if birth_state == None:
            birth_state = self.DEFAULT_RED_BIRTH_STATE
        self.dt_report = dt_report
        UAV = UAVModel(dt=self.dt_move)
        UAV.ammo = 0
        UAV.id = 1
        UAV.red = True
        UAV.blue = False
        UAV.side = "r"
        UAV.dead = 0
        UAV.color = np.array([1, 0, 0])
        # 红方出生点
        UAV.pos_ = birth_state['position']
        UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
        speed = UAV.speed
        UAV.psi = birth_state['psi']
        UAV.last_psi_v = UAV.psi
        UAV.theta = 0 * pi / 180
        UAV.gamma = 0 * pi / 180
        UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                            sin(UAV.theta),
                                            cos(UAV.theta) * sin(UAV.psi)])
        lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
        UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                    theta0=UAV.theta, o00=o00)
        self.RUAV = UAV
        
        # △h动作输出有效性测试
        self.height_req = height_req
        self.psi_req = psi_req
        self.v_req = v_req

        # 高度超调量
        self.height_overshoot = 0
        self.init_height_error = self.RUAV.alt - self.height_req
        # 航向超调量
        self.heading_overshoot = 0
        self.init_heading_error = sub_of_radian(self.RUAV.psi, self.psi_req)


    def get_state(self, side='r'):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        ego = self.RUAV

        # 夹角
        v_ = ego.vel_
        vh_ = ego.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = ego.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        
        # 速度观测量
        v_own = v
        # 本机高度
        h_own = ego.alt
        # 本机俯仰角
        sin_theta = sin(ego.theta)
        cos_theta = cos(ego.theta)
        # 本机滚转角
        sin_phi = sin(ego.phi)
        cos_phi = cos(ego.phi)

        # 剩余导弹量
        ammo = 0

        p = ego.p
        q = ego.q
        r = ego.r

        theta_v = ego.theta_v
        psi_v = ego.psi_v
        delta_psi_v = sub_of_radian(self.psi_req, psi_v)  # 水平速度分量和目标航向之间的差角(弧度)

        alpha_air = ego.alpha_air
        beta_air = ego.beta_air

        delta_psi_control = sub_of_radian(self.psi_req, self.RUAV.psi)
        cos_delta_psi = cos(delta_psi_control)
        sin_delta_psi = sin(delta_psi_control)
        delta_height_control = np.clip((self.height_req-self.RUAV.alt), -5000, 5000)

        one_side_states = {
            "ego_main": np.array([
                float(v_own),  # 0本机速度 m/s
                float(h_own),  # 1本机高度 m
                float(cos_theta),  # 2
                float(sin_theta),  # 3
                float(cos_phi),  # 4
                float(sin_phi),  # 5
                int(ammo)  # 6剩余导弹数量
            ]),

            "ego_control": np.array([
                float(p),  # 0 p rad/s act1_last
                float(q),  # 1 q rad/s act2_last
                float(r),  # 2 r rad/s act3_last
                float(theta_v),  # 3
                float(delta_psi_v),  # 4
                float(alpha_air),  # 5 rad
                float(beta_air),  # 6 rad
                float(ego.Ny),  # 7 过载量
            ]),

            "flight_cmd":  np.array([
                cos_delta_psi,
                sin_delta_psi,
                delta_height_control,
                self.v_req - self.RUAV.speed,
            ])
        }
        return one_side_states
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340
        s["ego_control"][7] /= 5 # 取一个不上不下的过载量

        s["flight_cmd"][2] /= 5000
        s["flight_cmd"][3] /= 340
        return s

    def unscale_state(self, obs_input):
        # 使用 deepcopy 避免修改传入对象
        o = copy.deepcopy(obs_input)
        o["ego_main"][0] *= 340
        o["ego_main"][1] *= 5e3
        o["ego_control"][0] *= (2 * pi)  # (2 * pi) 5000
        o["ego_control"][1] *= (2 * pi)  # (2 * pi) pi
        o["ego_control"][2] *= (2 * pi)  # (2 * pi) 340
        o["ego_control"][7] *= 5 # 取一个不上不下的过载量

        o["flight_cmd"][2] *= 5000
        o["flight_cmd"][3] *= 340
        return o
    
    def obs2obs_check(self, obs):
        """
        将扁平化的 obs (numpy array) 还原为 check_obs (dict)。
        该 check_obs 处于 scale 后的状态 (即可以直接输入 unscale_state)。
        
        Args:
            obs: (Dim, ) 或 (Batch, Dim) 的 numpy 数组或 tensor
        
        Returns:
            check_obs: 字典形式的状态
        """
        # 1. 确保 obs 是 numpy 数组
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().detach().numpy()
            
        key_dims = {
            "ego_main": 7,        # bool -> 1
            "ego_control": 8,   # int -> 1
            "flight_cmd": 4,       # bool -> 1
        }
        
        is_batch = (obs.ndim > 1)
        
        obs_dict = {}
        ptr = 0
        
        # 4. 按顺序切分重构
        for key in key_dims:
            if key not in key_dims:
                # 理论上 key_order_1v1 里的 key 都应该有定义，防御性编程
                continue
                
            dim = key_dims[key]
            
            # 切片
            if is_batch:
                val = obs[:, ptr : ptr + dim]
            else:
                val = obs[ptr : ptr + dim]
            
            ptr += dim
            
            if dim == 1:
                if not is_batch:
                    val = val[0] # 还原为标量
                # Batch 模式下通常保留 (B, 1) 维度以便后续处理
            
            obs_dict[key] = val

        # 注意：ego_control 不在 key_order_1v1 中，因此无法恢复。
        # unscale_state 函数中有 `if "ego_control" in s` 的检查，所以这是安全的。
            
        return obs_dict

    def base_obs(self, side='r', pomdp=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        state = self.get_state(side)  # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0])
        self.state_init["flight_cmd"] = np.array([0, 0, 0, 0])

        observation = self.scale_state(state)
        self.obs_init = self.scale_state(self.state_init)
        return observation

    def get_obs(self, side='r'):
        pre_full_obs = self.base_obs()
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.flight_key_order}
        
        # 弹药量不能被看到
        full_obs["ego_main"][6] = 0
        # full_obs["ego_control"][4] = 0

        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.flight_key_order)
        return flat_obs, full_obs

    def step(self, action):
        self.action = action
        # action['cont'] 由 PID 输出，顺序为 [aileron, elevator, rudder, throttle]
        aileron, elevator, rudder, throttle = action['cont']
        self.t += self.dt_report
        self.t = round(self.t, 2) # 保留两位小数
        time_rate = int(round(self.dt_report/self.dt_move))
        self.RUAV.last_psi_v = self.RUAV.psi_v
        for _ in range(time_rate):
            # UAVModel.move(p2p=True) 期望第一个参数对应 elevator, 第二个参数对应 aileron, 
            # 第四个参数对应 throttle, rudder 参数单独传递
            self.RUAV.move(target_height=elevator, delta_heading=aileron, target_speed=throttle, \
                relevant_height=True, relevant_speed=False, with_theta_req=False, p2p=True, rudder=rudder)
            
            # 记录高度超调量: 误差方向与初始误差相反，且取绝对值最大的值
            height_error = self.RUAV.alt - self.height_req
            if height_error * self.init_height_error < 0:
                if abs(height_error) > abs(self.height_overshoot):
                    self.height_overshoot = height_error
            # 记录航向超调量：
            heading_error = sub_of_radian(self.RUAV.psi, self.psi_req)
            if heading_error * self.init_heading_error < 0:
                if abs(heading_error) > abs(self.heading_overshoot):
                    self.heading_overshoot = heading_error

            done = self.get_done()
            if done:
                break
        next_obs, _ = self.get_obs()
        reward = self.get_reward(action)
        
        return next_obs, reward, done

    def get_done(self,):
        done = 0
        # 超时结束
        if self.t > self.time_limit:
            done = 1
        ruav_state = self.get_state()
        alt = ruav_state["ego_main"][1]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        ny = self.RUAV.Ny
        # 失败条件：失速、高度过低
        self.fail = 0
        # 撞地
        if alt < self.min_alt:
            self.crash = 1
            self.fail = 1
        
        if self.realistic: # 考虑迎角过载限制
            min_alpha_air = -7  # -5
            max_alpha_air = 26  # 29
            max_beta_air = 15
            min_ny = -3
            max_ny = 9.5
        else: # 无迎角过载限制
            min_alpha_air = -91
            max_alpha_air = 91
            max_beta_air = 181
            min_ny = -10
            max_ny = 30

        # 失速
        if alpha_air < min_alpha_air or alpha_air > max_alpha_air or abs(beta_air) > max_beta_air: # 负过载机动的限制可能比我想的还要严格
            self.stall = 1
            self.fail = 1
        # 结构过载
        if ny > max_ny or ny < min_ny: # 负过载机动的限制可能比我想的还要严格
            self.break_up = 1
            self.fail = 1

        # 任何一种失败都结束回合
        if self.fail:
            done = 1
            self.RUAV.dead = 1

        return done


    def get_reward(self, action):
        aileron, elevator, rudder, throttle = action['cont']
        ruav_state = self.get_state()
        speed = ruav_state["ego_main"][0]
        alt = ruav_state["ego_main"][1]
        sin_theta = ruav_state["ego_main"][3]
        cos_theta = ruav_state["ego_main"][2]
        theta = np.arctan2(sin_theta, cos_theta)
        sin_phi = ruav_state["ego_main"][5]
        cos_phi = ruav_state["ego_main"][4]
        phi = atan2(sin_phi, cos_phi)
        p = ruav_state["ego_control"][0]
        q = ruav_state["ego_control"][1]
        r = ruav_state["ego_control"][2]
        theta_v = ruav_state["ego_control"][3]
        delta_psi_v = ruav_state["ego_control"][4]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        cos_delta_psi = ruav_state["flight_cmd"][0]
        sin_delta_psi = ruav_state["flight_cmd"][1]
        height2req = ruav_state["flight_cmd"][2]
        speed2req = ruav_state["flight_cmd"][3]
        climb_rate = self.RUAV.vu

        delta_psi = np.arctan2(sin_delta_psi, cos_delta_psi)

        self.get_done()

        # 存活奖励
        reward_alive = 0.01 + 0.5 # 10

        # 失败惩罚
        reward_end = 0
        if self.fail:
            steps_wasted = (self.time_limit-self.t)/self.dt_report
            reward_end -= 800 # 400 + steps_wasted * 1 

        # 误差计算
        psi2req = delta_psi_v

        # 高度误差惩罚，从用法上看不如用俯仰角约束的效果好
        r_alt = 0
        # r_alt += 0.16 * np.sign(height2req) * np.clip(self.RUAV.vu / 100, -1, 1)
        r_alt += -0.01 * abs(np.clip(self.RUAV.vu / 100, -1, 1)) * (1-abs(height2req)/5000)  # 距离越近，调节越需要轻微的调节
        # 0.15 可能有些大

        # 高度限制奖励/惩罚
        r_alt += 2*((alt <= self.min_alt_safe) * np.clip(self.RUAV.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-self.RUAV.vu / 100, -1, 1))

        # 航向误差惩罚
        r_angle = 0  # 1 # DEBUG
        psi_dot = sub_of_radian(self.RUAV.psi_v, self.RUAV.last_psi_v)/self.dt_report  # 使用航迹角而非航向角，减少噪声
        
        r_angle += 10 * np.sign(delta_psi_v) * psi_dot  # 转弯角速度的奖励
        r_angle += - 0.5 * abs(psi_dot) * (1-abs(delta_psi_v)/pi)  # 遏制超调
        r_angle += - 5 * abs(delta_psi)/pi  # 航向误差绝对值的惩罚还是要存在
        

        # 俯仰角惩罚
        desired_theta = height2req/5000*pi/2  # 去除 pi/3 约束
                        # (height2req>=0)*height2req/5000*pi/3 + \
                        # (height2req<0)*height2req/5000*pi/2
        r_angle += - 5 * abs(theta - desired_theta) # 应和desired_theta保持一致  3小了

        # 和奖励无关，方便画图
        self.theta_req = height2req/5000*pi/2
        L_ = np.array([cos(self.theta_req)*cos(self.psi_req), sin(self.theta_req), cos(self.theta_req)*sin(self.psi_req)])
        self.AO = np.arccos(np.clip(np.dot(L_, self.RUAV.point_) / (1*1), -1, 1)) * 180/pi
        
        # 滚转角惩罚
        if desired_theta >= 0:
            r_angle += 0.001 * cos(phi)
        else:
            r_angle  += 0


        # 0.01 有些强了

        if abs(theta)*180/pi <= 70:
            # 需要右拐的时候 左倾带来惩罚，需要左拐的时候右倾带来惩罚
            sin_phi = np.sin(phi)  # (abs(sub_of_radian(phi,-pi/2))-abs(sub_of_radian(phi, pi/2)))/2

            r_angle += 2 * ((delta_psi > 0) * min(sin_phi, 0)/pi +\
                        (delta_psi < 0) * -max(sin_phi, 0)/pi)


        # 滚转角速度惩罚
        if abs(psi2req) < 15 * pi/180 \
            or abs(theta)*180/pi > 75:  # 大俯仰机动应该降低滚转角速度
            r_angle += -0.2 * abs(p)/pi  # 0.4 偏强？ 0.1偏弱？
        else:
            r_angle += -0.05 * abs(p)/pi # 0.01 可能有些弱？
        
        # r_angle -= (0.05 * abs(q)/pi + 0.01 * abs(r)/pi) # 加q和r的惩罚

        # 速度奖励: 使用纵向加速度 Nx 作为引导因子，加速收敛
        # 当速度偏低(speed2req > 0)时，正的纵向过载 Nx 会产生正向奖励
        r_speed = self.RUAV.Nx * np.sign(speed2req) * 2  # 0.5 小了
        # r_speed += -0.002 * min(abs(speed2req), 30) # 速度误差绝对值惩罚还是别给了，免得淹没其他微小的奖励信号

        # 迎角过载惩罚(惩罚负迎角和过大的正迎角)
        reward_alpha = 0.0 # 0.5
        if alpha_air >= 15:
            reward_alpha -= (alpha_air-15) * 5/(26-15) # 10 可能有些大，没有把大迎角的全部优势拿出来
        if alpha_air < -2:
            reward_alpha -= ((-2) - alpha_air) * 15/((-2) - (-5))
        ny = self.RUAV.Ny
        if ny<=-1:
            reward_alpha -= 3 + ((-1)-ny)*6 *2
        if ny >= 7:
            reward_alpha -= 3 + (ny-7)*3 *2
        
        # 侧滑角惩罚（尽量少侧滑）
        reward_beta = - 1.2 * abs(beta_air/5) # 1可能有些小，2可能大了
        reward_beta -= abs(rudder) * 0.3  # 方向舵在稳定的的时候尽量少打 0.1 可能小了

        reward = np.sum([
            1 * reward_alive,
            1 * reward_end,
            1 * r_angle,
            1 * r_alt,
            1 * r_speed,
            1 * reward_alpha,
            1 * reward_beta,
        ])

        self.v_error = -(speed2req)
        self.psi_error = -psi2req*180/pi
        self.theta_error = (theta - desired_theta)*180/pi

        # 其他奖励待续
        return reward
        

    def render(self, t_bias=0, target_range=5e3):
        # 记录 NUE 坐标
        loc_LLH = self.RUAV.lon, self.RUAV.lat, self.RUAV.alt
        N, U, E = LLH2NUE(loc_LLH[0], loc_LLH[1], loc_LLH[2], lon_o=self.o00[0], lat_o=self.o00[1])
        self.uav_pos_ = np.array([N, U, E])
        
        delta_N = target_range * cos(self.theta_req) * cos(self.psi_req)
        delta_U = target_range * sin(self.theta_req)
        delta_E = target_range * cos(self.theta_req) * sin(self.psi_req)
        self.target_pos_ = np.array([N + delta_N, U + delta_U, E + delta_E])

        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            if not self.RUAV.dead:
                pilot = 'Donkey'
                color = 'Red'
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                            f"{self.RUAV.phi * 180 / pi:.6f}|{self.RUAV.theta * 180 / pi:.6f}|{self.RUAV.psi * 180 / pi:.6f},"
                            f"Name=F16,Pilot={pilot},Color={color}\n"
                        )
                # 绘制目标
                delta_H = self.height_req
                lon_T, lat_T, _ = NUE2LLH(N+delta_N, U+delta_U, E+delta_E, lon_o=self.o00[0], lat_o=self.o00[1])
                
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id+1},T={(lon_T):.6f}|{(lat_T):.6f}|{delta_H:.6f},"
                            f"Name=Carrot,Color=Blue\n"
                        )

            else:
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"

            # loc_r = [self.RUAV.lon, self.RUAV.lat, self.RUAV.alt]
            # if self.tacview_show:
            #     data_to_send = ''
            #     data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            #             float(self.t + t_bias), self.RUAV.id, loc_r[0], loc_r[1], loc_r[2], self.RUAV.phi * 180 / pi, self.RUAV.theta * 180 / pi,
            #             self.RUAV.psi * 180 / pi)

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"
            self.tacview.send_data_to_client(data_to_send)


# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
max_steps = 30 * 65e4
hidden_dim = [128, 128] # [128, 128]
gamma = 0.9
lmbda = 0.9
epochs = 5  # 10
eps = 0.2
dt_decide = 0.2 # 0.2
dt_move=0.02

pre_train_rate = 0 # 0.25 # 0.25

state_dim = 7+8+4  # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
action_dim = 4 # test
# action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
action_bound = np.array([[-1.1,1.1],[-1.1,1.1],[-1.1,1.1],[-0.2,1.2]])  # aileron, elevator, rudder, throttle
mission_name = 'FlightControl_测试'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
log_dir = os.path.join(project_root, "./logs/control", mission_name + "-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__=='__main__':
    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=3*60, help="maximum episode time length")
    args = parser.parse_args()
    
    env = track_env(dt_move=dt_move, tacview_show=use_tacview, time_limit=args.max_episode_len)

    # 创建一个 dummy env 获取维度
    dummy_env = track_env(dt_move=dt_move, time_limit=args.max_episode_len)

    teacher_agent = UnifiedPolicyWrapper(dummy_env, dt_decide=dt_decide)

    action_dims_dict = {'cont': action_dim, 'cat': [], 'bern': 0}
    policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
    from Algorithms.MLP_heads import ValueNet
    critic = ValueNet(state_dim, hidden_dim).to(device)
    
    agent = PPOHybrid(actor, critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        
    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    def save_meta_once(path, state_dict):
        if os.path.exists(path):
            return
        meta = {k: list(v.shape) for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(meta, f)

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Visualize.tensorboard_visualize import TensorBoardLogger

    out_range_count = 0
    return_list = []
    steps_count = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    try:
        t_bias = 0
        # 强化学习训练
        rl_steps = 0
        i_episode = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
        beta_ao = 0.01 ** (dt_decide / 10.0) # 超出最后10s以前的误差忽略不计
        while rl_steps < max_steps:
            i_episode += 1
            episode_return = 0
            
            init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

            birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)
                                }
            warm_up = np.clip(rl_steps/30e4, 0, 1) # 预热，0.3 的步数后开始动真格
            
            height_req = np.clip(init_height + \
                    warm_up * np.random.uniform(-1, 1)*5000, \
                        1000, 13000)

            psi_req = np.random.uniform(-pi, pi) * warm_up # 预热，0.3以后真正开始动真格
            v_req = np.random.uniform(0.8, 2.5)*340

            alpha_distill = 1.0 * (1- 0.9 * warm_up)
            distil_epochs = max(int(10 * (1 - 0.9 * warm_up)), 1)

            env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
            ao_ema_episode = 0.0
            v_error_ema_episode = 0.0
            obs, obs_check = env.get_obs()
            done = False
            steps_in_epi = 0

            while not done:  # 每个训练回合
                # 1.执行动作得到环境反馈
                obs, obs_check = env.get_obs()
                action, u, _, _ = agent.take_action(obs, explore=True)
                rl_steps += 1

                # if abs(env.t % 0.5) <= env.dt_move:
                    # print("----")
                    # print("delta_psi", np.arctan2(obs_check["flight_cmd"][1], obs_check["flight_cmd"][0]) * 180 / pi)
                    # temp_state = env.unscale_state(obs_check)
                    # print("delta_height", temp_state["flight_cmd"][2])
                    # print("delta_speed", temp_state["flight_cmd"][3])
                    # print("--")
                    # print("aileron", action['cont'][0])
                    # print("elevator", action['cont'][1])
                    # print("rudder", action['cont'][2])
                    # print("throttle", action['cont'][3])
                    # print('--')
                    # print("obs_check", obs_check)
                    # print("----")
                    # print(f"Episode {i_episode}, Step {rl_steps}, time: {env.t}")

                
                next_obs, reward, done = env.step(action)
                ao_ema_episode = beta_ao * ao_ema_episode + (1 - beta_ao) * (env.AO)
                v_error_ema_episode = beta_ao * v_error_ema_episode + (1 - beta_ao) * abs(env.v_error)

                # debug 用
                height_req_show = env.height_req/1000
                height_show = env.RUAV.alt/1000
                psi_req_show = env.psi_req*180/pi
                psi_show = env.RUAV.psi*180/pi
                v_req_show = env.v_req
                v_show = env.RUAV.speed

                transition_dict['states'].append(obs)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_obs)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['action_bounds'].append(action_bound)
                obs = next_obs
                episode_return += reward
                env.render(t_bias)
                steps_in_epi += 1

            env.clear_render(t_bias)
            t_bias += env.t

            if env.fail==1:
                out_range_count+=1
            return_list.append(episode_return)

            if i_episode % 10 == 0:
                agent.update(transition_dict, adv_normed=1, mini_batch_size=512)
                agent.distil(transition_dict, teacher_agent=teacher_agent, epochs=distil_epochs, alpha=1.0) # alpha=alpha_distill
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}

            # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
            if i_episode % 10 == 1:
                # critic overwrite
                critic_path = os.path.join(log_dir, "critic.pt")
                th.save(agent.critic.state_dict(), critic_path)
                # actor RL snapshot
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

            
            # tqdm 训练进度显示
            if (i_episode + 1) >= 10:
                print(f"episode {i_episode+1}, 进度: {rl_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")

            # tensorboard 训练进度显示
            logger.add("train/0 episode_return", episode_return, rl_steps)
            logger.add("train/0 survive", 1-env.fail, rl_steps)
            logger.add("train_plus/0 height_overshoot", abs(env.height_overshoot), rl_steps)
            logger.add("train_plus/0 heading_overshoot", abs(env.heading_overshoot)*180/pi, rl_steps)
            
            # 记录平均 AO (回合结束时的 EMA 值，消除初始偏差)
            logger.add("train_plus/0 avg AO", ao_ema_episode/(1 - beta_ao**max(1, steps_in_epi)), rl_steps)
            logger.add("train_plus/0 avg v_e", v_error_ema_episode/(1 - beta_ao**max(1, steps_in_epi)), rl_steps)

            actor_grad_norm = model_grad_norm(agent.actor)
            critic_grad_norm = model_grad_norm(agent.critic)
            # 梯度监控
            logger.add("train/1 actor_grad_norm", actor_grad_norm, rl_steps)
            logger.add("train/2 critic_grad_norm", critic_grad_norm, rl_steps)
            # 损失函数监控
            logger.add("train/3 actor_loss", agent.actor_loss, rl_steps)
            logger.add("train/4 critic_loss", agent.critic_loss, rl_steps)
            # 强化学习actor特殊项监控
            logger.add("train/5 entropy", agent.entropy_mean, rl_steps)

            if hasattr(agent.actor.net, 'log_std_cont'):
                current_std = torch.exp(agent.actor.net.log_std_cont).mean().item()
                logger.add("train/5 std", current_std, rl_steps)

            logger.add("train/6 ratio", agent.ratio_mean, rl_steps)
            logger.add("train/7 steps", i_episode + 1, rl_steps)
            if hasattr(agent, 'dis_actor_loss') and agent.dis_actor_loss != 0:
                logger.add("train/8 distil_loss", agent.dis_actor_loss, rl_steps)


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()


        print(f"日志已保存到：{logger.run_dir}")

        end_time = datetime.now()
        print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
        elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
        print(f"Simulation duration: {elapsed_hours:.4f} hours")

