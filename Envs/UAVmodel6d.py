from random import random
import random
# [修改] 替换 gym 为 gymnasium
import gymnasium as gym
from gymnasium import spaces
import copy
import numpy as np
from math import *
import jsbsim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from controller.Controller_function import *
# from Envs.MissileModel1 import *  # test

from Controller.F16PIDController2 import *
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *

# 无人机模型
class UAVModel(object):
    def __init__(self, dt=0.02, dt_fire=0.2):
        super(UAVModel, self).__init__()
        # 无人机的标识
        self.last_calc_missile_time = -10
        self.shoot_prob = 0
        self.PIDController = None
        self.phi = None
        self.rnn_states = None
        self.hist_act = None
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
        self.set_height = None
        self.set_speed = None
        # 雷达性能约束
        self.max_radar_angle = 60*pi/180
        self.max_radar_range = 120e3

        # 导弹相关属性
        self.ammo = 6  # 最大可携带导弹数量
        self.missile_count = 0  # 已发射导弹数量
        self.missile_launch_interval = 10  # 发射间隔 36
        self.last_launch_time = -10  # 上次发射时间
        self.missile_detect_range = 20e3  # 探测范围
        self.missile_launch_angle = 15  # 发射视角范围 45
        self.missile_min_range = 0.8e3  # 最小发射距离
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
        self.dt_fire = dt_fire

        self.o00 = None
        self.start_lon = None
        self.start_lat = None

    def reset(self, lon0=118, lat0=30, h0=8000, v0=200, psi0=0, phi0=0, theta0=0, o00=np.array([118, 30])):
        sim = jsbsim.FGFDMExec(None, None)
        sim.set_debug_level(0)
        sim.set_dt(self.dt)  # 解算步长 dt 秒
        # 设置模型路径（一般 JSBSim pip 包自动包含）
        sim.load_model("f16")  # f15, p51d, ball 等模型可选
        # 设置初始状态（单位：英尺、节、角度）
        sim["ic/h-sl-ft"] = h0 * 3.2808  # 高度：m -> ft
        sim["ic/vt-kts"] = v0 * 1.9438  # 空速： m/s-> 节 vt 真空速 ic/vc 仪表空速
        sim["ic/psi-true-deg"] = psi0 * 180 / pi  # 航向角: °
        temp = phi0 * 180 / pi
        sim["ic/phi-deg"] = temp if abs(abs(temp)/90-1)> 0.1 else temp*0.9
        sim["ic/theta-deg"] = theta0 * 180 / pi
        sim["ic/alpha-deg"] = 0
        sim["ic/beta-deg"] = 0
        sim["ic/long-gc-deg"] = lon0
        sim["ic/lat-gc-deg"] = lat0

        # 设置初始油量(实际燃油会按照F-16的最大容量提供，会小于这个设定值)
        sim["propulsion/tank[0]/contents-lbs"] = 5000.0  # 设置0号油箱油量
        sim["propulsion/tank[1]/contents-lbs"] = 5000.0  # 设置1号油箱油量（如果有）

        self.start_lon = lon0
        self.start_lat = lat0
        self.o00 = o00
        self.set_height = h0
        self.set_speed = v0

        # 初始化状态
        sim.run_ic()
        sim.set_property_value('propulsion/set-running', -1)
        self.sim = sim

        self.psi = psi0
        self.phi = phi0
        self.theta = theta0
        self.p = self.sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
        self.q = self.sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
        self.r = self.sim["velocities/r-rad_sec"]        
        self.speed = self.sim["velocities/vt-fps"] * 0.3048  # ego_vc (unit: m/s)
        self.point_ = active_rotation(np.array([1,0,0]), self.psi, self.theta, self.phi)
        
        # 取当前位置
        lon = self.sim["position/long-gc-deg"]  # 经度
        lat = self.sim["position/lat-gc-deg"]  # 纬度
        alt = self.sim["position/h-sl-ft"] / 3.2808  # 高度（英尺）
        self.lon, self.lat, self.alt = lon, lat, alt
        # 简单的相对位置计算
        self.x, self.y, self.z = LLH2NUE(lon, lat, alt, lon_o=self.o00[0], lat_o=self.o00[1])

        vn = self.sim["velocities/v-north-fps"] * 0.3048  # 向北分量
        ve = self.sim["velocities/v-east-fps"] * 0.3048  # 向东分量
        vu = -self.sim["velocities/v-down-fps"] * 0.3048  # 向上分量（正表示上升）
        self.climb_rate = vu
        self.vn, self.ne, self.vu = vn, ve, vu
        
        # 过载量
        self.Ny = self.sim["accelerations/Nz"]  # 垂直过载
        self.Nz = self.sim["accelerations/Ny"]  # 侧向过载
        self.Nx = self.sim["accelerations/Nx"]  # 纵向过载

        gamma_angle = atan2(vu, sqrt(vn ** 2 + ve ** 2)) * 180 / pi  # 爬升角（度）
        course_angle = atan2(ve, vn) * 180 / pi  # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

        self.theta_v = gamma_angle * pi/180
        self.psi_v = course_angle * pi/180

        self.alpha_air = self.sim["aero/alpha-deg"] * pi / 180  # 当前迎角
        self.beta_air = self.sim["aero/beta-deg"] * pi / 180  # 当前侧滑角

        # self.rnn_states = np.zeros((1, 1, 128))
        # self.hist_act = np.array([0, 0, 0, 1])
        self.PIDController = F16PIDController()

        self.obs_memory = None
        self.act_memory = np.array([0,0,340]) # 动作记忆
        
        self.missile_launch_time = []

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

    def move(self, target_height, delta_heading, target_speed, relevant_height=True, relevant_speed=False, with_theta_req=False, p2p=False, rudder=None):
        # 单位：m, rad, mm/s, metric公制单位，imperial英制单位
        if relevant_height==False: # 使用绝对高度指令
            self.set_height = target_height
        else: # 使用相对高度指令
            delta_target_height_english = target_height * 3.2808
            self.set_height = (delta_target_height_english + self.sim["position/h-sl-ft"]) / 3.2808
        delta_heading = delta_heading  # sub_of_radian(target_delta_heading, sim["attitude/psi-deg"] * pi / 180)
        
        if relevant_speed==False: # 使用绝对速度指令
            self.set_speed = target_speed
        else: # 使用相对速度指令
            delta_speed_english = target_speed * 3.2808
            self.set_speed = (delta_speed_english + self.sim["velocities/vt-fps"]) / 3.2808

        # 角速度
        p = self.sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
        q = self.sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
        r = self.sim["velocities/r-rad_sec"]  # 偏航角速度（弧度/秒）
        self.p, self.q, self.r = p,q,r

        # 速度矢量关于地面的角度
        vn = self.sim["velocities/v-north-fps"] * 0.3048  # 向北分量
        ve = self.sim["velocities/v-east-fps"] * 0.3048  # 向东分量
        vu = -self.sim["velocities/v-down-fps"] * 0.3048  # 向上分量（正表示上升）
        self.climb_rate = vu
        self.vn, self.ne, self.vu = vn, ve, vu

        # 过载量
        self.Ny = self.sim["accelerations/Nz"]  # 垂直过载
        self.Nz = self.sim["accelerations/Ny"]  # 侧向过载
        self.Nx = self.sim["accelerations/Nx"]  # 纵向过载

        gamma_angle = atan2(vu, sqrt(vn ** 2 + ve ** 2)) * 180 / pi  # 爬升角（度）
        course_angle = atan2(ve, vn) * 180 / pi  # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

        self.theta_v = gamma_angle * pi/180
        self.psi_v = course_angle * pi/180

        current_heading = self.sim["attitude/psi-deg"] * pi / 180
        target_heading = sub_of_radian(current_heading + delta_heading, 0) * 180/pi

        obs_jsbsim = np.zeros(14)
        # obs_jsbsim[0] = target_theta * pi / 180  # 期望俯仰角 # 测试姿态控制器
        obs_jsbsim[0] = self.set_height / 5000  # 期望高度 # 测试飞行控制器
        obs_jsbsim[1] = delta_heading  # 期望相对航向角
        obs_jsbsim[2] = self.set_speed / 340  # 期望速度
        obs_jsbsim[3] = self.sim["attitude/theta-deg"] * pi / 180  # 当前俯仰角
        obs_jsbsim[4] = self.sim["velocities/vt-fps"] * 0.3048 / 340  # 当前速度
        obs_jsbsim[5] = self.sim["attitude/phi-deg"] * pi / 180  # 当前滚转角
        obs_jsbsim[6] = self.sim["aero/alpha-deg"] * pi / 180  # 当前迎角
        obs_jsbsim[7] = self.sim["aero/beta-deg"] * pi / 180  # 当前侧滑角
        obs_jsbsim[8] = p
        obs_jsbsim[9] = q
        obs_jsbsim[10] = r
        obs_jsbsim[11] = gamma_angle * pi / 180  # 爬升角
        obs_jsbsim[12] = sub_of_degree(target_heading, course_angle) * pi / 180  # 相对航迹角
        obs_jsbsim[13] = self.sim["position/h-sl-ft"] * 0.3048 / 5000  # 高度/5000（英尺转米）

        # norm_act由F16control函数输出
        # norm_act, self.rnn_states, self.hist_act = F16control(obs_jsbsim, self.rnn_states, self.hist_act)
        # print(self.side, norm_act)

        if p2p==False: # 通过控制器间接控制
            if with_theta_req == False:
                norm_act = self.PIDController.flight_output(obs_jsbsim, dt=self.dt)  # # 测试飞行控制器
            else:
                obs_jsbsim[0] = np.clip(target_height, -pi/2, pi/2)  # 高度接口当俯仰角接口用, 输入介于[-1,1]之间
                norm_act = self.PIDController.att_output(obs_jsbsim, dt=self.dt)
        else: # 直接输出舵偏角
            if rudder is None:
                rudder = 0.0
            norm_act = np.array([delta_heading, target_height, rudder, target_speed]) # 端到端控制，不使用方向舵,拿前三个当舵偏角来输入
        # print(obs_jsbsim)

        self.alpha_air = self.sim["aero/alpha-deg"] * pi / 180  # 当前迎角
        self.beta_air = self.sim["aero/beta-deg"] * pi / 180  # 当前侧滑角

        # norm_act=np.array([0.05, -1, 0.1, 1]) # test

        self.sim["fcs/aileron-cmd-norm"], \
            self.sim["fcs/elevator-cmd-norm"], \
            self.sim["fcs/rudder-cmd-norm"], \
            self.sim["fcs/throttle-cmd-norm"] = norm_act  # 设置控制量

        self.sim.run()  # 运行一步

        # 取当前位置
        lon = self.sim["position/long-gc-deg"]  # 经度
        lat = self.sim["position/lat-gc-deg"]  # 纬度
        alt = self.sim["position/h-sl-ft"] / 3.2808  # 高度（英尺转米）

        self.lon, self.lat, self.alt = lon, lat, alt

        # 简单的相对位置计算
        self.x, self.y, self.z = LLH2NUE(lon, lat, alt, lon_o=self.o00[0], lat_o=self.o00[1])
        # self.x = (lat - start_lat) * 110540 # 经度差转米（近似）
        # self.y = alt * 0.3048 # 高度转米
        # self.z = (lon - start_lon) * 111320 # 纬度差转米（近似）
        v = self.sim["velocities/vt-fps"] * 0.3048  # ego_vc (unit: m/s)
        self.speed = v

        # 取姿态角度
        self.phi = self.sim["attitude/phi-deg"] * pi / 180  # 滚转角 (roll)
        self.theta = self.sim["attitude/theta-deg"] * pi / 180  # 俯仰角 (pitch)
        self.psi = self.sim["attitude/psi-deg"] * pi / 180  # 航向角 (yaw)

        self.vel_ = np.array([vn, vu, ve]) # ft.s转m/s
        self.point_ = active_rotation(np.array([1,0,0]), self.psi, self.theta, self.phi)

        # 速度更新位置
        self.pos_ = np.array([self.x, self.y, self.z])
    
    def short_range_kill(self, target):
        # 近距杀，不需要导弹的模型
        pt_ = target.pos_
        L_ = pt_ - self.pos_
        distance = np.linalg.norm(L_)
        speed = np.linalg.norm(self.vel_)
        attack_angle = acos(np.dot(self.vel_, L_) / (distance * speed))
        if distance < 8e3 and attack_angle * 180 / pi < 30:
            return True  # "杀伤锥"

    def can_launch_missile(self, target, current_time):
        can_shoot = True
        """判断是否可以发射导弹"""
        if self.dead or target.dead:  # 如果自己或者目标死了，不能发射导弹
            return False

        # 检查导弹数量限制 fixme
        if self.ammo < 1:
            # print('导弹射完')
            can_shoot = False
            return False

        # 检查目标是否在探测范围内
        distance = np.linalg.norm(target.pos_ - self.pos_)
        if distance > self.missile_max_range or distance < self.missile_min_range:
            can_shoot = False

        # 检查目标是否在视角范围内
        target_vector = target.pos_ - self.pos_
        velocity_vector = self.vel_
        dot_product = np.dot(target_vector, velocity_vector)
        norms = np.linalg.norm(target_vector) * np.linalg.norm(velocity_vector)
        if norms == 0:
            can_shoot = False
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / pi
        if angle > self.missile_launch_angle:
            can_shoot = False

        # 检查发射间隔
        missile_launch_interval = self.missile_launch_interval * distance/10e3
        
        if current_time - self.last_launch_time < missile_launch_interval:
            can_shoot = False

        # 可发射区解算
        if can_shoot:
            # # test
            # if distance<40e3:
            #     can_shoot = True

            from LaunchZone.calc_hit_points_maneuver_from_RWR import hit_prob
            # 每次计算出的概率保持1s
            if current_time-self.last_calc_missile_time >= 1: # 0.5
                shoot_prob = hit_prob(self.pos_, self.vel_,
                                      target.pos_, target.vel_)
                self.last_calc_missile_time = current_time
                self.shoot_prob = shoot_prob
            else:
                shoot_prob = self.shoot_prob

            # 随机采样
            shoot_prob_in_1s = 1-(1-shoot_prob)**(self.dt_fire)
            if np.random.uniform(0, 1) <= shoot_prob_in_1s:
                can_shoot = True
            else:
                can_shoot = False

            # # 只在不可逃逸区发射
            # if shoot_prob>=1:
            #     can_shoot = True
            # else:
            #     can_shoot = False

        # 导弹发射概率输出
        return can_shoot

    def launch_missile(self, target, current_time, missile_class):
        """发射导弹"""
        # 创建新导弹
        # new_missile = missile_class(self.pos_, self.vel_, current_time)
        new_missile = missile_class(self.pos_, self.vel_, target.pos_, target.vel_, current_time)  # test
        if self.blue:  # target.red:
            new_missile.side = 'b'
            new_missile.id = 301 + self.missile_count
        if self.red:  # target.blue:
            new_missile.side = 'r'
            new_missile.id = 101 + self.missile_count
        # 根据距离调整导弹参数
        distance = np.linalg.norm(target.pos_ - self.pos_)
        new_missile.max_g = 40  # 最大过载
        new_missile.sight_angle_max = pi / 2  # 导引头视角
        new_missile.launcher_id = self.id  # 发射机id
        new_missile.target_id = target.id  # 目标机id
        # print('导弹已发射')
        # print('发射时间', current_time)
        self.missile_count += 1
        self.missile_launch_time.append(current_time)
        # print(self.side, '已发射', self.missile_count)
        self.last_launch_time = current_time
        return new_missile

    # 是否可探测到目标
    def can_detect_target(self, target):
        can = False
        if self.dead:  # 本机已死，拒绝探测目标
            return False
        L_ego_enm_ = target.pos_ - self.pos_
        dist = norm(L_ego_enm_)
        if dist <= 160e3:
            angle = np.arccos(np.dot(L_ego_enm_, self.vel_) / (dist * self.speed))
            if angle * 180 / pi <= 60:
                can = True
        return can

    # 是否可跟踪目标
    def can_track_target(self, target):
        can = False
        if self.dead:  # 本机已死，拒绝追踪目标
            return False
        L_ego_enm_ = target.pos_ - self.pos_
        dist = norm(L_ego_enm_)
        if dist <= 80e3:
            angle = np.arccos(np.dot(L_ego_enm_, self.vel_) / (dist * norm(self.vel_)))
            if angle * 180 / pi <= 55:
                can = True
        return can

    # 是否可为导弹提供数据链中制导
    def can_offer_guidance(self, missile, UAVs):
        can = False
        if self.dead:  # 本机已死，拒绝提供中制导
            return False
        L_ego_m_ = missile.pos_ - self.pos_
        dist = norm(L_ego_m_)

        # 刚发射导弹的时候距离为0，直接判定为可以中制导（因为初制导的逻辑没做）
        if dist <1e-3 or self.speed <1e-3:
            return True

        # 如果分母有效，继续计算
        angle = np.arccos(np.dot(L_ego_m_, self.vel_) / (dist * norm(self.vel_)))

        if angle * 180 / pi < 60 and dist <= 50e3:  # 假设飞机雷达和导弹的通信距离在50km
            target_uav = None
            # 根据导弹目标id查找目标
            for uav in UAVs:
                if uav.id == missile.target_id:
                    target_uav = uav
                    break
            # 检查该目标是否可被本机追踪
            if target_uav and self.can_track_target(target_uav):
                can = True
        return can
