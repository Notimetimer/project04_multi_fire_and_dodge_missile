from random import random
import random
from gym import spaces
import copy
import numpy as np
from math import *
import jsbsim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Controller.Controller_function import *


def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


def LLH2NUE(lon, lat, h, lon_o=118, lat_o=30, h_o=0):
    x = (lat - lat_o) * 111000  # 纬度差转米（近似）
    y = h - h_o
    z = (lon - lon_o) * (111413 * cos(lat_o * pi / 180) - 94 * cos(3 * lat_o * pi / 180))  # 经度差转米（近似）
    return x, y, z


def NUE2LLH(N, U, E, lon_o=118, lat_o=30, h_o=0):
    lon = lon_o + N / (111413 * cos(lat_o * pi / 180) - 94 * cos(3 * lat_o * pi / 180))
    lat = lat_o + E / 111000
    h = U + h_o
    return lon, lat, h


# 无人机模型
class UAVModel(object):
    def __init__(self, dt=0.02):
        super(UAVModel, self).__init__()
        # 无人机的标识
        self.sim = None
        self.id = None
        self.red = False
        self.blue = False
        self.side = None  # 阵营
        self.size = 5  # meter
        self.area = 27.87  # F-16机翼面积
        self.m = 12000  # F-16 正常起飞重量
        self.color = None
        # 无人机的状态
        self.pos_ = np.zeros(3)
        self.speed = None  # 速率
        self.vel_ = None  # 速度矢量
        self.mach = None
        self.psi = None  # 航向角
        self.theta = None  # 俯仰角
        self.gamma = None  # 滚转角
        self.nx = None
        self.ny = None
        self.nz = None
        # 无人机飞行约束
        self.speed_max = 2 * 340
        self.speed_min = 120
        # 无人机轨迹
        self.trajectory = np.empty((0, 3))  # 新增轨迹列表
        self.vellist = np.empty((0, 3))
        self.lon = None
        self.lat = None
        self.alt = None

        # 导弹相关属性
        self.ammo = 6  # 最大可携带导弹数量
        self.missile_count = 0  # 已发射导弹数量
        self.missile_launch_interval = 9.1 * 4  # 发射间隔
        self.last_launch_time = -10  # 上次发射时间
        self.missile_detect_range = 20e3  # 探测范围
        self.missile_launch_angle = 45  # 发射视角范围
        self.missile_min_range = 5e3  # 最小发射距离
        self.missile_optimal_range = 25e3  # 最佳发射距离
        self.missile_max_range = 40e3  # 最大发射距离 50
        self.missile_launch_speed_threshold = 100  # 300  # 最小发射速度要求

        # 过载量控制
        self.nx_limit = [-1, 1.5]
        self.ny_limit = [-1.5, 6]
        self.gamma_max = 170 * pi / 180

        # 无人机对抗相关
        self.got_hit = False
        self.crash = False
        self.attacking = False
        self.dead = False
        self.dt = dt

    def reset(self, lon0=118, lat0=30, h0=8000, v0=200, psi0=0, phi0=0, theta0=0):
        sim = jsbsim.FGFDMExec(None, None)
        sim.set_debug_level(0)
        sim.set_dt(self.dt)  # 解算步长 dt 秒
        # 设置模型路径（一般 JSBSim pip 包自动包含）
        sim.load_model("f16")  # f15, p51d, ball 等模型可选
        # 设置初始状态（单位：英尺、节、角度）
        sim["ic/h-sl-ft"] = h0 * 3.2808  # 高度：m -> ft
        sim["ic/vt-kts"] = v0 * 1.9438  # 空速： m/s-> 节
        sim["ic/psi-true-deg"] = psi0 * 180 / pi  # 航向角: °
        sim["ic/phi-deg"] = phi0 * 180 / pi
        sim["ic/theta-deg"] = theta0 * 180 / pi
        sim["ic/alpha-deg"] = 0
        sim["ic/beta-deg"] = 0
        sim["ic/long-gc-deg"] = lon0
        sim["ic/lat-gc-deg"] = lat0
        self.start_lon = lon0
        self.start_lat = lat0

        # 初始化状态
        sim.run_ic()
        sim.set_property_value('propulsion/set-running', -1)
        self.sim = sim

        self.psi = psi0
        self.phi = phi0
        self.theta = theta0
        self.rnn_states = np.zeros((1, 1, 128))
        self.hist_act = np.array([0, 0, 0, 1])

    # todo 阻力系数：应该是和马赫数和迎角有关的，但是先借用下导弹的阻力系数函数了
    def Cd(self, mach):
        if 0 < mach <= 0.9:
            cd = 0.16
        if 0.9 < mach <= 1.1:
            cd = 0.16 + 0.29 * (mach - 0.9) / 0.2
        if 1.1 < mach <= 3:
            cd = 0.45 - 0.25 * (mach - 1.1) / 1.9
        else:
            cd = 0.2
        return cd / 10

    # todo 补充无人机的运动方程和动作逻辑
    def move(self, target_height, delta_heading, target_speed, record=False):
        # 单位：m, rad, mm/s, \

        target_height_english = target_height * 3.2808
        # target_delta_heading = target_delta_heading

        delta_height = (target_height_english - self.sim["position/h-sl-ft"]) / 3.2808
        delta_heading = delta_heading  # sub_of_radian(target_delta_heading, sim["attitude/psi-deg"] * pi / 180)
        # delta_heading = 19 * pi / 180 # test
        delta_speed = (target_speed - self.sim["velocities/vt-fps"] * 0.3048) / 1.9438

        # 构建观测向量
        obs_jsbsim = [
            delta_height,  # ego delta altitude (unit: m)
            delta_heading,  # ego delta heading (unit rad)
            delta_speed,  # ego delta velocities_u (unit: m/s)
            self.sim["position/h-sl-ft"] / 3.2808,  # ego_altitude (unit: m)
            self.sim["attitude/phi-deg"] * pi / 180,  # ego_roll (unit: rad)
            self.sim["attitude/theta-deg"] * pi / 180,  # ego_pitch (unit: rad)
            self.sim["velocities/u-fps"] * 0.3048,  # ego_body_v_x (unit: f/s -> m/s)
            self.sim["velocities/v-fps"] * 0.3048,  # ego_body_v_y (unit: f/s -> m/s)
            self.sim["velocities/w-fps"] * 0.3048,  # ego_body_v_z (unit: f/s -> m/s)
            self.sim["velocities/vt-fps"] * 0.3048  # ego_vc (unit: m/s)
        ]

        # norm_act由F16control函数输出
        norm_act, self.rnn_states, self.hist_act = F16control(obs_jsbsim, self.rnn_states, self.hist_act)

        self.sim["fcs/aileron-cmd-norm"], \
            self.sim["fcs/elevator-cmd-norm"], \
            self.sim["fcs/rudder-cmd-norm"], \
            self.sim["fcs/throttle-cmd-norm"] = norm_act  # 设置控制量

        self.sim.run()  # 运行一步

        # 取当前位置
        lon = self.sim["position/long-gc-deg"]  # 经度
        lat = self.sim["position/lat-gc-deg"]  # 纬度
        alt = self.sim["position/h-sl-ft"] / 3.2808  # 高度（英尺）

        self.lon, self.lat, self.alt = lon, lat, alt

        # 简单的相对位置计算
        self.x, self.y, self.z = LLH2NUE(lon, lat, alt, lon_o=self.start_lon, lat_o=self.start_lat)
        # self.x = (lat - start_lat) * 110540 # 经度差转米（近似）
        # self.y = alt * 0.3048 # 高度转米
        # self.z = (lon - start_lon) * 111320 # 纬度差转米（近似）
        v = self.sim["velocities/vt-fps"] * 0.3048  # ego_vc (unit: m/s)

        # 取姿态角度
        self.phi = self.sim["attitude/phi-deg"] * pi / 180  # 滚转角 (roll)
        self.theta = self.sim["attitude/theta-deg"] * pi / 180  # 俯仰角 (pitch)
        self.psi = self.sim["attitude/psi-deg"] * pi / 180  # 航向角 (yaw)
        self.alpha_air = self.sim["aero/alpha-deg"]  # 迎角
        self.beta_air = self.sim["aero/beta-deg"]  # 侧滑角

        v_ = np.array([v * cos(self.theta) * cos(self.psi),
                       v * sin(self.theta),
                       v * cos(self.theta) * sin(self.psi)])

        self.vel_ = v_ * v / np.linalg.norm(v_)
        # 速度更新位置
        self.pos_ = np.array([self.x, self.y, self.z])

        if record:
            self.trajectory = np.vstack((self.trajectory, self.pos_))
            self.vellist = np.vstack((self.vellist, self.vel_))

    pass


if __name__ == '__main__':
    dt = 0.02
    UAV = UAVModel(dt=dt)
    UAV.ammo = 0
    UAV.id = 1
    UAV.red = True
    UAV.blue = False
    UAV.side = "red"
    UAV.color = np.array([1, 0, 0])

    o00 = np.array([118, 30])  # 地理原点的经纬

    # 红方出生点
    pos_ = np.array([-38841.96119795, 3000.02131746, -1686.95469864])
    speed = 300
    psi = 0 * pi / 180
    theta = 0 * pi / 180
    phi = 0 * pi / 180

    lon_uav, lat_uav, h_uav = NUE2LLH(pos_[0], pos_[1], pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
    print(h_uav)
    UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=speed, psi0=psi, phi0=phi, theta0=theta)

    # 红方期望的飞行状态
    expected_height = 5000
    expected_heading = 60 * pi / 180
    target_speed = 400

    t = 0
    tacview_show = True  # 是否显示Tacview

    if tacview_show:
        print('please prepare tacview')
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Visualize.tacview_visualize import Tacview

        tacview = Tacview()

    while t <= 2 * 60:
        t += dt
        delta_heading = sub_of_radian(expected_heading, UAV.psi)
        UAV.move(expected_height, delta_heading, target_speed, record=False)

        # 通过tacview可视化
        if tacview_show:
            send_t = f"{t:.2f}"
            name_R = '001'
            loc_r = [float(UAV.lon), float(UAV.lat), float(UAV.alt)]
            # print('φθψ',UAV.phi, UAV.theta, UAV.psi)
            data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % \
                           (float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], UAV.phi * 180 / pi,
                            UAV.theta * 180 / pi, UAV.psi * 180 / pi)
            tacview.send_data_to_client(data_to_send)
            # time.sleep(0.001)
