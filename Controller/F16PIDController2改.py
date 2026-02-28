# from abc import ABC
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from math import *
# from typing import Literal
# import matplotlib.pyplot as plt
# from abc import ABC, abstractmethod
import sys
import os
# import matplotlib
# matplotlib.use('Qt5Agg')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import socket
# import threading
import time
import jsbsim
# import matplotlib.pyplot as plt
import numpy as np
# from simple_pid import PID
from numpy.linalg import norm


class PositionPID(object):
    """位置式PID算法实现"""

    # output_list = []
    def __init__(self, max, min, p, i, d) -> None:
        self._max = max  # 最大输出限制，规避过冲
        self._min = min  # 最小输出限制
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数
        self._pre_error = 0  # t-1 时刻误差值
        self._integral = 0  # 误差积分值

    def calculate(self, error, dt):
        """
        计算t时刻PID输出值cur_val
        """
        # 比例项
        p_out = self.k_p * error
        # 积分项
        self._integral += (error * dt)

        # 仿照simple_pid，将积分项整项预先限幅
        self._integral = np.clip(self._integral, self._min/self.k_i, self._max/self.k_i) \
            if self.k_i!=0 else self._integral
        
        i_out = self.k_i * self._integral
        # 微分项
        derivative = (error - self._pre_error) / dt
        d_out = self.k_d * derivative
        # t 时刻pid输出
        output = p_out + i_out + d_out
        # 限制输出值
        if output > self._max:
            output = self._max
        elif output < self._min:
            output = self._min
        self._pre_error = error
        return output

    def clear_integral(self):
        self._integral = 0


def active_rotation(vector,heading,theta,gamma):
    # 飞机的轴在外界看来是怎样的， 前上右在北天东看来朝向哪个方向
    # vector是行向量，旋转顺序仍然是 psi, theta, gamma，但是计算顺序相反
    # 主动旋转，-在1下
    # 注意：北天东坐标
    psi = - heading
    Rpsi=np.array([
        [cos(psi), 0, sin(psi)],
        [0, 1, 0],
        [-sin(psi), 0, cos(psi)]
        ])
    Rtheta=np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
        ])
    Rgamma=np.array([
        [1, 0, 0],
        [0, cos(gamma), -sin(gamma)],
        [0, sin(gamma), cos(gamma)]
        ])
    return vector@Rgamma.T@Rtheta.T@Rpsi.T


def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


def sub_of_degree(input1, input2):
    # 计算两个角度的差值，范围为[-180, 180]
    diff = input1 - input2
    diff = (diff + 180) % 360 - 180
    return diff


class F16PIDController:
    def __init__(self):
        # 建议dt=0.02
        self.check_switch2 = None
        self.check_switch1 = None
        self.a_last = 0
        self.e_last = 0
        self.r_last = 0
        self.t_last = 0.5
        self.last_outputs = [self.a_last, self.e_last, self.r_last, self.t_last]

        # 调参
        self.yaw_pid = None
        self.e_pid = PositionPID(max=1, min=-1, p=16 / pi, i=0 / pi, d=0 / pi)  # 16, 0.3, 8
        self.r_pid = None
        self.t_pid = PositionPID(max=1, min=-1, p=1, i=0.3, d=0.2)
        # self.t_pid = PID(1, 0.3, 0.2, setpoint=0)
        # self.t_pid.output_limits = (-1, 1)
        self.pids = [self.yaw_pid, self.e_pid, self.r_pid, self.t_pid]

        self.type = None
    def flight_output(self, state_input, dt=0.02):
        target_height_devided = state_input[0]
        current_height_devided = state_input[13]
        v = state_input[4]
        gamma_rad = state_input[11]       

        error_h = np.clip(target_height_devided - current_height_devided, -1, 1)

        if error_h >= 0:
            kh = pi / 2 # 3
        else:
            kh = pi / 2 # + pi/8

        target_theta = error_h * kh
        # print('target_theta',target_theta*180/pi)

        temp = state_input
        temp[0] = target_theta
        norm_act = self.att_output(temp, dt=dt)
        return norm_act

    def att_output(self, state_input, dt=0.02):
        norm_act = self.att_calculate(state_input, dt=dt)
        theta = state_input[3]
        alpha = state_input[6] * 180 / pi
        q = state_input[9]
        # # 迎角限制器
        if -8 < alpha < 13:
            k_alpha_air = 0.01
        else:
            k_alpha_air = 0.2

        if theta * 180 / pi < -70:
            k_alpha_air = 0

        norm_act[1] = (1 - k_alpha_air) * norm_act[1] + k_alpha_air * (alpha / 20)
        norm_act[1] = np.clip(norm_act[1], -1, 1)

        return norm_act

    def att_calculate(self, state_input, dt):
        yaw_pid, e_pid, r_pid, t_pid = self.pids
        a_last, e_last, r_last, t_last = self.last_outputs
        theta_req = state_input[0]
        delta_heading_req = state_input[1]
        v_req = state_input[2]
        theta = state_input[3]
        v = state_input[4]
        phi = state_input[5]
        alpha_air = state_input[6]
        beta_air = state_input[7]
        p = state_input[8]
        q = state_input[9]
        r = state_input[10]
        climb_rad = state_input[11]
        delta_course_rad = state_input[12]

        # abs_target_heading = state_input[14]
        # abs_psi = state_input[15]

        # 油门控制
        # t_pid.setpoint = v_req
        # throttle = 0.5 + 0.5 * t_pid(v, dt)

        v_error = v_req-v
        throttle = 0.5 + 0.5 * t_pid.calculate(v_error, dt)

        # # 方向舵控制
        # rudder=0 # abaaba
        rudder = -beta_air / (5 * pi / 180)

        # 升降舵控制
        L_ = 1 * np.array(
            [np.cos(theta_req) * np.cos(delta_heading_req), np.sin(theta_req),
             np.cos(theta_req) * np.sin(delta_heading_req)])
        v_ = 1 * np.array(
            [np.cos(climb_rad) * np.cos(delta_course_rad), np.sin(climb_rad),
             np.cos(climb_rad) * np.sin(delta_course_rad)])
        x_b_ = 1 * np.array(
            [np.cos(theta) * np.cos(0), np.sin(theta), np.cos(theta) * np.sin(0)])

        # 将期望航向投影到体轴xy平面上，后根据与体轴x夹角设定升降舵量的大小
        # 体轴系的两个基当做向量转到“转过一个航向角的惯性系”
        y_b_ = active_rotation(np.array([0, 1, 0]), 0, theta, phi)
        z_b_ = active_rotation(np.array([0, 0, 1]), 0, theta, phi)
        L_xy_b_ = L_ - np.dot(L_, z_b_) * z_b_ / norm(z_b_)
        x_b_2L_xy_b_ = np.cross(x_b_, L_xy_b_) / norm(L_xy_b_)
        x_b_2L_xy_b_sin = np.dot(x_b_2L_xy_b_, z_b_)
        x_b_2L_xy_b_cos = np.dot(x_b_, L_xy_b_) / norm(L_xy_b_)
        delta_z_angle = np.arctan2(x_b_2L_xy_b_sin, x_b_2L_xy_b_cos)

        # 重写的位置式pid
        elevetor = -e_pid.calculate(delta_z_angle, dt=dt)
        elevetor = np.clip(elevetor, -1, 1)

        # 特例：大坡度时不允许推杆
        if abs(phi) * 180 / pi > 50: # and v / 300 > 1:
            elevetor = np.clip(elevetor, -1, 0)

        # 副翼战术机动控制
        # combat flight
        L_yz_b_ = L_ - np.dot(L_, x_b_) * x_b_ / norm(x_b_)
        y_b_2L_yz_b_ = np.cross(y_b_, L_yz_b_) / norm(L_yz_b_)
        y_b_2L_yz_b_sin = np.dot(y_b_2L_yz_b_, x_b_)
        y_b_2L_yz_b_cos = np.dot(y_b_, L_yz_b_) / norm(L_yz_b_)
        delta_x_angle = np.arctan2(y_b_2L_yz_b_sin, y_b_2L_yz_b_cos)

        # 特例：压机头能够得着的，就不翻转机身
        if abs(delta_x_angle) > 5 / 6 * pi and -pi / 6 < delta_z_angle < 0 and abs(theta) < 80 * pi / 180:
            delta_x_angle = sub_of_radian(delta_x_angle + pi, 0)
            # print('push')
        # else:
        # print('pull')

        # # 特例：abs(delta_x_angle *180/pi)>45 时杆量减半
        # if abs(delta_x_angle *180/pi)>30:
        #     elevetor /= 2

        # 通用
        roll_error = delta_x_angle
        aileron = roll_error / pi * 3 - p / pi * 1 # 1
        # aileron = roll_error/pi*3 - p/pi * 2

        # # debug
        # if alpha_air*180/pi<-5:
        #     print('strange')
        #     print('delta_x_angle', delta_x_angle * 180 / pi)
        #     aileron = (roll_error / pi * 6 - p / pi * 8) / 4
        #     if 0.9 < abs(delta_x_angle / (pi / 2)) < 1.1:
        #         elevetor = 0

        # steady filght
        # 副翼平稳飞行控制：delta_z_angle**2+delta_x_angle**2足够小时副翼由phi比例控制
        steady_switch_angle = 15  # 20 15 30

        self.check_switch1 = acos(np.dot(L_, v_) / norm(L_) / norm(v_)) * 180 / pi
        self.check_switch2 = abs(theta_req)*180/pi

        if acos(np.dot(L_, v_) / norm(L_) / norm(v_)) * 180 / pi < steady_switch_angle and \
                abs(theta_req) < 60 * pi / 180:
            k_steady_yaw = 3 / steady_switch_angle
            phi_req = np.clip(delta_heading_req * 180 / pi * k_steady_yaw, -1, 1) * (pi / 3)
            roll_error = phi_req - phi
            # aileron = (roll_error / pi * 6 - p / pi * 3) / 2
            aileron = (roll_error / pi * 6 - p / pi * 3) / 2

            self.type = 0  # steady
        else:
            self.type = 1  # fast

        # print(self.check_switch1)
        # print(self.check_switch2)
        # input("Press any key to continue...")
        
        # if alpha_air*180/pi<-5:
        #     print('?')
        #     pass

        aileron = np.clip(aileron, -1, 1)
        norm_act = np.array([aileron, elevetor, rudder, throttle])
        return norm_act


if __name__ == '__main__':

    tacview_show = 1  # 是否显示Tacview

    # dt = 0.02  # 0.02
    dt_control = 0.02
    dt_move = 0.01
    control_move_rate = int(round(dt_control/dt_move))

    if tacview_show:
        print('please prepare tacview')
        # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Visualize.tacview_visualize import Tacview

        tacview = Tacview()

    # 记录轨迹和状态数据
    psis = []
    thetas = []
    phis = []
    heights = []
    alphas = []
    betas = []
    speeds = []
    overloads = []
    thrusts = []  # 添加推力数据记录
    fuel_flows = []
    fuels = []

    time_steps = []
    # 记录控制量
    aileron_cmd = []
    elevator_cmd = []
    rudder_cmd = []
    throttle_cmd = []

    # 启动 JSBSim
    sim = jsbsim.FGFDMExec(None, None)
    sim.set_debug_level(0)
    sim.set_dt(dt_move)  # 解算步长 dt 秒

    # 设置模型路径（一般 JSBSim pip 包自动包含）
    sim.load_model("f16")  # f15, p51d, ball 等模型可选

    # 初始姿态与目标姿态
    # 连续输出并tacview中可视化
    start_time = time.time()
    # target_theta = 1 # 测试姿态控制
    target_height = 4e3  # m # 测试飞行控制器 1e3
    target_heading = 50  # 度 to rad 270
    target_speed = 340 * 1.13  # m/s
    t_last = 60 * 5

    # 设置初始状态（单位：英尺、节、角度）
    sim["ic/h-sl-ft"] = 4e3 * 3.2808  # 高度：m -> ft
    sim["ic/vt-kts"] = 340 * 1.13 * 1.9438  # 空速： m/s-> 节
    sim["ic/psi-true-deg"] = 90  # 航向角: °
    sim["ic/phi-deg"] = 0
    sim["ic/theta-deg"] = 0
    sim["ic/alpha-deg"] = 0
    sim["ic/beta-deg"] = 0
    sim["ic/long-gc-deg"] = -118  # 设置初始经度（单位：度）
    sim["ic/lat-gc-deg"] = 34  # 设置初始纬度（单位：度）

    # 设置初始油量(实际燃油会按照F-16的最大容量提供，会小于这个设定值)
    sim["propulsion/tank[0]/contents-lbs"] = 5000.0  # 设置0号油箱油量
    sim["propulsion/tank[1]/contents-lbs"] = 5000.0  # 设置1号油箱油量（如果有）

    # 初始化状态
    sim.run_ic()
    sim.set_property_value('propulsion/set-running', -1)

    # 指数滑动平均控制量
    aileron_last = 0
    elevator_last = 0
    rudder_last = 0
    throttle_last = 0.5

    f16PIDController = F16PIDController()
    norm_act = np.array([0,0,0,0.5])

    # hist_act=np.array([0,0,0,1])
    for step in range(int(t_last / dt_move)):
        current_t = step * dt_move

        # 无限燃油
        sim["propulsion/tank[0]/contents-lbs"] = 5000.0  # 设置0号油箱油量
        sim["propulsion/tank[1]/contents-lbs"] = 5000.0  # 设置1号油箱油量（如果有）

        # 随机
        # if step % 10 == 0:
        #     target_heading = np.random.rand()*160 # test
        #     target_height = sim["position/h-sl-ft"] * 0.3048 # test

        # # ### 逗猫
        # if current_t < 15:
        #     target_height = 5000  # m
        #     target_heading = 90  # 度 to rad
        #     target_speed = 300
        # elif current_t < 1 * 60:
        #     target_height = 10000  # m
        #     target_heading = -120  # 度 to rad
        # elif current_t < 1 * 60 + 27:
        #     target_height = 7000  # m
        #     target_heading = 0  # 度 to rad
        # elif current_t < 2 * 60 + 10:
        #     target_height = 8000  # m
        #     target_heading = sub_of_degree(sim["attitude/psi-deg"], 60)  # 度 to rad
        # else:
        #     target_heading = sub_of_degree(sim["attitude/psi-deg"], -10)

        sim.run()
        current_time = step * dt_move
        time_steps.append(current_time)

        # delta_height = (target_height - sim["position/h-sl-ft"]) / 3.2808
        delta_heading = sub_of_degree(target_heading, sim["attitude/psi-deg"])
        # # delta_heading = 19 * pi / 180 # test
        # delta_speed = (target_velocity - sim["velocities/vt-fps"] * 0.3048) / 1.9438

        # 取姿态角度
        phi = sim["attitude/phi-deg"]  # 滚转角 (roll)
        theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
        psi = sim["attitude/psi-deg"]  # 航向角 (yaw)
        alpha = sim["aero/alpha-deg"]  # 迎角
        beta = sim["aero/beta-deg"]  # 侧滑角
        # 过载量
        nz_g = sim["accelerations/Nz"]  # 垂直过载
        ny_g = sim["accelerations/Ny"]  # 侧向过载
        nx_g = sim["accelerations/Nx"]  # 纵向过载

        # 角速度
        p = sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
        q = sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
        r = sim["velocities/r-rad_sec"]  # 偏航角速度（弧度/秒）

        # 速度矢量关于地面的角度
        vn = sim["velocities/v-north-fps"]  # 向北分量
        ve = sim["velocities/v-east-fps"]  # 向东分量
        vu = -sim["velocities/v-down-fps"]  # 向下分量（正表示下降）

        gamma_angle = atan2(vu, sqrt(vn ** 2 + ve ** 2)) * 180 / pi  # 爬升角（度）
        course_angle = atan2(ve, vn) * 180 / pi  # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

        # 构建观测向量
        obs_jsbsim = np.zeros(14)
        # obs_jsbsim[0] = target_theta * pi / 180  # 期望俯仰角 # 测试姿态控制器
        obs_jsbsim[0] = target_height / 5000  # 期望高度 # 测试飞行控制器
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
        obs_jsbsim[11] = gamma_angle * pi / 180  # 爬升角
        obs_jsbsim[12] = sub_of_degree(target_heading, course_angle) * pi / 180  # 相对航迹角
        obs_jsbsim[13] = sim["position/h-sl-ft"] * 0.3048 / 5000  # 高度/5000（英尺转米）
        # obs_jsbsim[14] = target_heading * pi / 180  # test
        # obs_jsbsim[15] = psi * pi / 180  # test

        # 输出姿态控制指令
        if step % control_move_rate==0:
            # norm_act = f16PIDController.att_output(obs_jsbsim, dt=dt) # 测试姿态控制器
            norm_act = f16PIDController.flight_output(obs_jsbsim, dt=dt_control)  # # 测试飞行控制器

        # # 指数滑动平均控制量
        # last_control = np.array([aileron_last, elevator_last, rudder_last, throttle_last])
        # norm_act=last_control*0.1+0.9*np.array(norm_act)

        sim["fcs/aileron-cmd-norm"], \
            sim["fcs/elevator-cmd-norm"], \
            sim["fcs/rudder-cmd-norm"], \
            sim["fcs/throttle-cmd-norm"] = norm_act  # .tolist()  # 设置控制量

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
        speeds.append(sim["velocities/vt-fps"] * 0.3048)
        overloads.append(sim["forces/load-factor"])
        thrust = sim.get_property_value('propulsion/engine/thrust-lbs')
        fuel_flow = sim["propulsion/engine/fuel-flow-rate-pps"]  # 燃油流量
        thrusts.append(thrust)
        fuel_flows.append(fuel_flow)

        fuels.append(sim["propulsion/total-fuel-lbs"])

        # 通过tacview可视化
        if tacview_show:
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
