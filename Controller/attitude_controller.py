from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from math import *
from typing import Literal
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import sys
import os
# import matplotlib
# matplotlib.use('Qt5Agg')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
import threading
import time
import jsbsim
import matplotlib.pyplot as plt
import math
import numpy as np
from simple_pid import PID
from numpy.linalg import norm

tacview_show=1
tacview_show = tacview_show


class DeltaPID(object):
    """增量式PID算法实现"""
    def __init__(self, p=0, i=0, d=0) -> None:
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数
        self._pre_error = 0  # t-1 时刻误差值
        self._pre_pre_error = 0  # t-2 时刻误差值
    def calculate(self, error, dt=0.02):
        p_change = self.k_p * (error - self._pre_error)
        i_change = self.k_i * error * dt
        d_change = self.k_d * (error - 2 * self._pre_error + self._pre_pre_error) / dt
        delta_output = p_change + i_change + d_change  # 本次增量
        self._pre_pre_error = self._pre_error
        self._pre_error = error
        return delta_output

def active_rotation(vector, heading, theta, gamma):
    # vector是行向量，根据psi，theta，gamma的顺序旋转坐标系，最后输出行向量
    # 注意：北天东坐标
    psi = - heading
    R1 = np.array([
        [cos(psi), 0, sin(psi)],
        [0, 1, 0],
        [-sin(psi), 0, cos(psi)]
    ])
    R2 = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cos(gamma), -sin(gamma)],
        [0, sin(gamma), cos(gamma)]
    ])
    return vector @ R1.T @ R2.T @ R3.T

def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff

def sub_of_degree(input1, input2):
    # 计算两个角度的差值，范围为[-180, 180]
    diff = input1 - input2
    diff = (diff + 180) % (360) - 180
    return diff

class F16PIDController:
    def __init__(self):
        # 建议dt=0.02
        self.a_last = 0
        self.e_last = 0
        self.r_last = 0
        self.t_last = 0.5
        self.last_outputs = [self.a_last, self.e_last, self.r_last, self.t_last]

        self.yaw_pid = DeltaPID(p=1/pi, i=0.1/pi, d=0/pi)
        self.e_pid = DeltaPID(p=2/pi, i=1/pi, d=0.1/pi)
        self.r_pid = DeltaPID(p=5/pi, i=1/pi, d=1/pi)
        self.t_pid = PID(1, 0.3, 0.2, setpoint=0)
        self.t_pid.output_limits = (-1, 1)
        self.pids = [self.yaw_pid, self.e_pid, self.r_pid, self.t_pid]

    def flight_output(self):
        pass

    def att_output(self, input, dt=0.02):
        norm_act = self.att_calculate(input, dt=dt)
        alpha = input[6]
        # # 迎角限制器
        if -8<alpha<13:
            k_alpha_air = 0.001
        else:
            k_alpha_air = 0.01
        norm_act[1] = (1-k_alpha_air)*norm_act[1]+k_alpha_air*(alpha/20)        
        norm_act[1] = np.clip(norm_act[1],-1,1)
        return norm_act

    def att_calculate(self, input, dt):
        yaw_pid, e_pid, r_pid, t_pid = self.pids
        a_last, e_last, r_last, t_last = self.last_outputs
        theta_req = input[0]
        delta_heading_req = input[1]
        v_req = input[2]
        theta = input[3]
        v = input[4]
        phi = input[5]
        alpha_air = input[6]
        beta_air = input[7]
        p = input[8]
        q = input[9]
        r = input[10]
        climb_rad = input[11]
        delta_course_rad = input[12]

        # 油门控制
        t_pid.setpoint = v_req
        throttle = 0.5 + 0.5 * t_pid(v, dt)

        # 方向舵控制
        rudder = -beta_air/(5*pi/180)

        # 升降舵控制
        L_ = 1 * np.array(
            [np.cos(theta_req) * np.cos(delta_heading_req), np.sin(theta_req),
             np.cos(theta_req) * np.sin(delta_heading_req)])
        v_ =  1 * np.array(
            [np.cos(climb_rad) * np.cos(delta_course_rad), np.sin(climb_rad),
             np.cos(climb_rad) * np.sin(delta_course_rad)])
        x_b_ = 1 * np.array(
            [np.cos(theta) * np.cos(0), np.sin(theta), np.cos(theta) * np.sin(0)])
        
        # 将期望航向投影到体轴xy平面上，后根据与体轴x夹角设定升降舵量的大小
        y_b_ = active_rotation(np.array([0, 1, 0]), 0, theta, phi)
        z_b_ = active_rotation(np.array([0, 0, 1]), 0, theta, phi)
        L_xy_b_ = L_ - np.dot(L_, z_b_) * z_b_ / norm(z_b_)
        x_b_2L_xy_b_ = np.cross(x_b_, L_xy_b_) / norm(L_xy_b_)
        x_b_2L_xy_b_sin = np.dot(x_b_2L_xy_b_, z_b_)
        x_b_2L_xy_b_cos = np.dot(x_b_, L_xy_b_) / norm(L_xy_b_)
        delta_z_angle = np.arctan2(x_b_2L_xy_b_sin, x_b_2L_xy_b_cos)

        elevetor = e_last + e_pid.calculate(-delta_z_angle, dt=dt)
        elevetor = np.clip(elevetor, -1, 1)

        # 副翼战术机动控制
        L_yz_b_ = L_ - np.dot(L_, x_b_) * x_b_ / norm(x_b_)
        y_b_2L_yz_b_ = np.cross(y_b_, L_yz_b_) / norm(L_yz_b_)
        y_b_2L_yz_b_sin = np.dot(y_b_2L_yz_b_, x_b_)
        y_b_2L_yz_b_cos = np.dot(y_b_, L_yz_b_) / norm(L_yz_b_)
        delta_x_angle = np.arctan2(y_b_2L_yz_b_sin, y_b_2L_yz_b_cos)

        # 特例：abs(delta_x_angle)>5/6*pi 且 delta_z_angle<pi/6时delta_x_angle需要+pi
        if abs(delta_x_angle) > 5/6*pi and delta_z_angle < pi/6:
            delta_x_angle = sub_of_radian(delta_x_angle + pi, 0)

        phi_error = delta_x_angle
        aileron = phi_error/pi*6 - p/pi * 4

        # 副翼平稳飞行控制：delta_z_angle**2+delta_x_angle**2足够小时副翼由phi比例控制
        if acos(np.dot(L_, v_)/norm(L_)/norm(v_))*180/pi < 20:
            k_steady_yaw = 3/20 # 10/20
            phi_req = np.clip(delta_heading_req*180/pi * k_steady_yaw,-1,1) *(pi/3)
            phi_error = phi_req-phi
            aileron = (phi_error/pi*6 -p/pi*4)/3

        aileron = np.clip(aileron, -1, 1)

        norm_act = np.array([aileron, elevetor, rudder, throttle])

        return norm_act


def heightcontrol(self):
    return 0

# if __name__ == '__main__':

# tacview_show = 0  # 是否显示Tacview

dt = 0.02  # 0.02

if tacview_show:
    print('please prepare tacview')
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Visualize.tacview_visualize import Tacview

    tacview = Tacview()

# 启动 JSBSim
sim = jsbsim.FGFDMExec(None, None)
sim.set_debug_level(0)
sim.set_dt(dt)  # 解算步长 dt 秒

# 设置模型路径（一般 JSBSim pip 包自动包含）
sim.load_model("f16")  # f15, p51d, ball 等模型可选

# 初始姿态与目标姿态
# 连续输出并tacview中可视化
start_time = time.time()
target_theta = 0  # 3000 * 3.2808  # 6000 m to ft
target_heading = 160  # 度 to rad
target_speed = 350  # m/s
t_last = 180.0

# 设置初始状态（单位：英尺、节、角度）
sim["ic/h-sl-ft"] = 2000 * 3.2808  # 高度：m -> ft
sim["ic/vt-kts"] = 300 * 1.9438  # 空速： m/s-> 节
sim["ic/psi-true-deg"] = 0  # 航向角: °
sim["ic/phi-deg"] = 0
sim["ic/theta-deg"] = 0
sim["ic/alpha-deg"] = 0
sim["ic/beta-deg"] = 0
sim["ic/long-gc-deg"] = -118  # 设置初始经度（单位：度）
sim["ic/lat-gc-deg"] = 34  # 设置初始纬度（单位：度）

# 初始化状态
sim.run_ic()
sim.set_property_value('propulsion/set-running', -1)

# 记录轨迹和状态数据
psis = []
thetas = []
phis = []
heights=[]
alphas=[]
betas=[]

thrust_data = []  # 添加推力数据记录
time_steps = []

# 记录控制量
aileron_cmd = []
elevator_cmd = []
rudder_cmd = []
throttle_cmd = []

# hist_act=np.array([0,0,0,1])
for step in range(int(t_last / dt)):
    sim.run()
    current_time = step * dt
    time_steps.append(current_time)

    # delta_height = (target_height - sim["position/h-sl-ft"]) / 3.2808
    delta_heading = sub_of_degree(target_heading, sim["attitude/psi-deg"])
    # # delta_heading = 19 * pi / 180 # test
    # delta_speed = (target_velocity - sim["velocities/vt-fps"] * 0.3048) / 1.9438
    
    # 取姿态角度
    phi = sim["attitude/phi-deg"]      # 滚转角 (roll)
    theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
    psi = sim["attitude/psi-deg"]      # 航向角 (yaw)
    alpha = sim["aero/alpha-deg"]      # 迎角
    beta = sim["aero/beta-deg"]        # 侧滑角
    # 过载量
    nz_g = sim["accelerations/Nz"]  # 垂直过载
    ny_g = sim["accelerations/Ny"]  # 侧向过载
    nx_g = sim["accelerations/Nx"]  # 纵向过载
    
    # 角速度
    p = sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
    q = sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
    r = sim["velocities/r-rad_sec"]  # 偏航角速度（弧度/秒）

    # 速度矢量关于地面的角度
    vn = sim["velocities/v-north-fps"]    # 向北分量
    ve = sim["velocities/v-east-fps"]     # 向东分量
    vu = -sim["velocities/v-down-fps"]     # 向下分量（正表示下降）

    gamma_angle = atan2(vu, sqrt(vn**2+ve**2))*180/pi      # 爬升角（度）
    course_angle = atan2(ve, vn)*180/pi    # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

    # 构建观测向量
    obs_jsbsim = np.zeros(13)
    obs_jsbsim[0] = target_theta * pi / 180  # 期望俯仰角
    obs_jsbsim[1] = delta_heading * pi / 180  # 期望相对航向角
    obs_jsbsim[2] = target_speed / 340  # 期望速度
    obs_jsbsim[3] = sim["attitude/theta-deg"] * pi / 180  # 当前俯仰角
    obs_jsbsim[4] = sim["velocities/vt-fps"] * 0.3048 / 340  # 当前速度
    obs_jsbsim[5] = sim["attitude/phi-deg"] * pi / 180  # 当前滚转角
    obs_jsbsim[6] = sim["aero/alpha-deg"] * pi / 180  # 当前迎角
    obs_jsbsim[7] = sim["aero/beta-deg"] * pi / 180  # 当前侧滑角
    obs_jsbsim[8] = p
    obs_jsbsim[9] = q
    obs_jsbsim[10] = r
    obs_jsbsim[11] = gamma_angle * pi / 180
    obs_jsbsim[12] = sub_of_degree(target_heading, course_angle) * pi / 180

    f16PIDController=F16PIDController()
    # 输出姿态控制指令
    norm_act = f16PIDController.att_output(obs_jsbsim, dt=dt)

    sim["fcs/aileron-cmd-norm"], \
        sim["fcs/elevator-cmd-norm"], \
        sim["fcs/rudder-cmd-norm"], \
        sim["fcs/throttle-cmd-norm"] = norm_act  # 设置控制量

    # # 记录控制量
    aileron_cmd.append(sim["fcs/aileron-cmd-norm"])
    elevator_cmd.append(sim["fcs/elevator-cmd-norm"])
    rudder_cmd.append(sim["fcs/rudder-cmd-norm"])
    throttle_cmd.append(sim["fcs/throttle-cmd-norm"])

    # 取当前位置
    lon = sim["position/long-gc-deg"]  # 经度
    lat = sim["position/lat-gc-deg"]  # 纬度
    alt = sim["position/h-sl-ft"] * 0.3048  # 高度（英尺转米）

    # 取速度分量
    u = sim["velocities/u-fps"] * 0.3048  # X轴速度 (fps转m/s)
    v = sim["velocities/v-fps"] * 0.3048  # Y轴速度 (fps转m/s)
    w = sim["velocities/w-fps"] * 0.3048  # Z轴速度 (fps转m/s)

    # 记录状态量
    phis.append(phi)
    psis.append(psi)
    thetas.append(theta)
    heights.append(alt)
    alphas.append(alpha)
    betas.append(beta)

    # 通过tacview可视化
    if tacview_show and step % np.round(1 / dt) == 0:
        send_t = f"{current_time:.2f}"
        name_R = '001'
        loc_r = [float(lon), float(lat), float(alt)]
        # data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
        data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
        float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], phi, theta, psi)
        tacview.send_data_to_client(data_to_send)
        # time.sleep(0.001)

    mach = sim["velocities/mach"]
    # # 可以记录或打印
    # print(f"Time: {current_time:.1f}s, Mach: {mach:.3f}")

    # time.sleep(0.01)

end_time = time.time()
print(f"程序运行时间: {end_time - start_time:.2f} 秒")


