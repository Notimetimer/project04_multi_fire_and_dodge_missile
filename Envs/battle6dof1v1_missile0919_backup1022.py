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
from numpy.linalg import norm


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
    try:
        shell = get_ipython().__class__.__name__  # ← 误报，不用管
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
            # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
            return os.getcwd()
        else:  # 其他 shell
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Envs.MissileModel1112 import *  # test
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Math_calculates.Calc_dist2border import calc_intern_dist2cylinder
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs

g = 9.81
dt_maneuver = 0.2  # 0.02 0.8 0.2
dt_move = 0.02
report_move_time_rate = int(round(dt_maneuver / dt_move))

o00 = np.array([118, 30])  # 地理原点的经纬
# t = 0
g_ = np.array([0, -g, 0])
# theta_limit = 85 * pi / 180

R_cage = 100e3

min_height = 0
max_height = 15e3

R_birth = 40e3

horizontal_center = np.array([0, 0])


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Battle(object):
    def __init__(self, args, tacview_show=0):
        # super(Battle, self).__init__() 
        self.alive_b_missiles = None
        self.alive_r_missiles = None
        self.BUAV = None
        self.RUAV = None
        self.dt_maneuver = dt_maneuver
        self.dt_move = dt_move
        self.UAV_ids = None
        self.UAV_hit = None
        self.Bmissiles = None
        self.Rmissiles = None
        self.missiles = None
        self.args = args
        self.RUAVs = None
        self.BUAVs = None
        self.UAVs = None
        self.RUAVsTable = None
        self.BUAVsTable = None
        self.UAVsTable = None
        self.RmissilesTable = None
        self.BmissilesTable = None
        self.missilesTable = None
        self.t = None
        self.game_time_limit = self.args.max_episode_len  # None
        self.running = None
        self.action_space = []
        # self.reset()  # 重置位置和状态
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.win = None
        self.lose = None
        self.draw = None
        self.max_alt = 15e3
        self.max_alt_danger = 14e3
        self.max_alt_save = 13e3
        self.min_alt_save = 3e3
        self.min_alt_danger = 2e3
        self.min_alt = 0.5e3  # 1e3
        self.R_cage = getattr(self.args, 'R_cage', R_cage) if hasattr(self.args, 'R_cage') else R_cage

        # # 智能体的观察空间
        # self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      r_obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]

        self.DEFAULT_RED_BIRTH_STATE = {'position': np.array([-R_birth * cos(0), 8000.0, -R_birth * sin(0)]),
                                        'psi': 0
                                        }
        self.DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([-R_birth * cos(pi), 8000.0, -R_birth * sin(pi)]),
                                         'psi': pi
                                         }
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
            self.visualize_cage()

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, ):  # 重置位置和状态

        if red_birth_state is None:
            red_birth_state = self.DEFAULT_RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.DEFAULT_BLUE_BIRTH_STATE

        self.Rmissiles = []
        self.Bmissiles = []
        self.missiles = []
        self.alive_r_missiles = []
        self.alive_b_missiles = []
        self.dt_maneuver = dt_maneuver  # simulation interval，1 second
        self.t = 0
        # self.game_time_limit = self.args.max_episode_len
        # 初始化无人机
        self.RUAVs = []
        self.BUAVs = []
        self.Rnum = 1
        self.Bnum = 1
        self.RUAVsTable = {}
        self.BUAVsTable = {}
        self.UAVsTable = {}
        self.RmissilesTable = {}
        self.BmissilesTable = {}
        self.missilesTable = {}
        self.win = 0
        self.lose = 0
        self.draw = 0
        # 红方初始化
        for i in range(self.Rnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = red_init_ammo
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.side = 'r'
            UAV.color = np.array([1, 0, 0])
            # 红方出生点
            UAV.pos_ = red_birth_state['position']  # np.array([-38841.96119795, 9290.02131746, -1686.95469864])
            UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
            speed = UAV.speed
            mach, _ = calc_mach(speed, UAV.pos_[1])
            UAV.mach = mach
            UAV.psi = red_birth_state[
                'psi']  # (0 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            self.RUAVs.append(UAV)
            self.RUAVsTable[UAV.id] = (UAV, UAV.side, UAV.dead)
        # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = blue_init_ammo
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.side = 'b'
            UAV.color = np.array([0, 0, 1])
            # 蓝方出生点
            UAV.pos_ = blue_birth_state['position']  # np.array([38005.14540582, 6373.80721704, -1734.42509136])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
            UAV.psi = blue_birth_state['psi']
            # (180 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.psi = sub_of_radian(UAV.psi, 0)
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            self.BUAVs.append(UAV)
            self.BUAVsTable[UAV.id] = (UAV, UAV.side, UAV.dead)
        self.running = True
        self.UAVs = self.RUAVs + self.BUAVs
        self.UAVsTable = {**self.RUAVsTable, **self.BUAVsTable}
        self.UAV_ids = [UAV.id for UAV in self.UAVs]
        self.UAV_hit = [False for _ in range(len(self.UAVs))]

        # todo 1v1的残留
        self.RUAV = self.RUAVs[0]
        self.BUAV = self.BUAVs[0]

        # print(red_birth_state)
        # print(blue_birth_state)

    def step(self, r_actions, b_actions):
        report_move_time_rate = int(round(self.dt_maneuver / dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]
        self.r_actions = r_actions.copy()
        self.b_actions = b_actions.copy()

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                # print(action)
                target_height = action[0]  # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                delta_heading = action[1]  # 相对方位(弧度)
                target_speed = action[2]  # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
                # print('target_height',target_height)
                # for i in range(int(self.dt_maneuver // dt_move)):
                UAV.move(target_height, delta_heading, target_speed, relevant_height=True)
                # 上一步动作
                # UAV.act_memory = np.array([action[0],action[1],action[2]])

            # 导弹移动
            self.missiles = self.Rmissiles + self.Bmissiles
            for missile in self.missiles[:]:  # 使用切片创建副本以允许删除
                target = self.get_target_by_id(missile.target_id)
                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹, 在飞机1V1的时候可以节省一点计算量，不用费事处理多目标的问题
                    missile.dead = True
                    continue
                else:
                    missile.target = target
                # if not missile.dead:
                # print('目标位置', target.pos_)
                # 计算前导弹和目标位速
                last_pmt_ = missile.pos_
                last_vmt_ = missile.vel_
                last_ptt_ = target.pos_
                last_vtt_ = target.vel_
                if not missile.dead:
                    # 获取目标信息
                    target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, last_ptt_)
                    # 更新导弹制导阶段
                    has_datalink = False
                    for uav in self.UAVs:
                        # 找到载机，判断载机能否为导弹提供中制导
                        if uav.id == missile.launcher_id:
                            if uav.can_offer_guidance(missile, self.UAVs):
                                has_datalink = True
                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt=self.dt_move, datalink=has_datalink)

                vmt1 = norm(last_vmt_)
                if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                    missile.dead = True
                if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                    missile.dead = True
                if missile.t > missile.t_max:  # 超时自爆
                    missile.dead = True
                if missile.t >= 0 + self.dt_move and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=self.dt_move)
                    if hit:
                        print(target.side, 'is hit')
                        missile.dead = True
                        missile.hit = True
                        missile.pos_ = point_m
                        missile.vel_ = last_vmt_
                        target.pos_ = point_t
                        target.vel_ = last_vtt_
                        target.dead = True
                        target.got_hit = True
                        self.UAV_hit[self.UAV_ids.index(target.id)] = True

            # 毁伤判断
            for i, UAV in enumerate(self.UAVs):
                # 飞机被导弹命中判断
                if UAV.red:
                    adv = self.BUAV
                if UAV.blue:
                    adv = self.RUAV
                if self.UAV_hit[i]:
                    UAV.dead = True
                    UAV.got_hit = True
                # 其他毁伤判断
                adv = self.UAVs[1 - i]
                pt_ = adv.pos_
                L_ = pt_ - UAV.pos_
                distance = np.linalg.norm(L_)
                # 近距杀
                #     short_range_killed = UAV.short_range_kill(adv)
                #     if short_range_killed:
                #         # self.running = False
                #         adv.got_hit = True
                # 出界判别
                if self.out_range(UAV):
                    UAV.dead = True
                    # self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
                UAV.dead = True
                # self.running = False

        r_dones = False
        b_dones = False
        if self.RUAV.dead:
            r_dones = True
        if self.BUAV.dead:
            b_dones = True

        self.RUAV = self.UAVs[0]
        self.BUAV = self.UAVs[1]

        if terminate:
            self.running = False

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def update_missile_state(self):
        alive_red_missiles = self.Rmissiles.copy()
        alive_blue_missiles = self.Bmissiles.copy()
        for missile in alive_red_missiles[:]:  # 遍历
            if missile.dead:
                alive_red_missiles.remove(missile)
        for missile in alive_blue_missiles[:]:  # 遍历
            if missile.dead:
                alive_blue_missiles.remove(missile)
        self.alive_r_missiles = alive_red_missiles
        self.alive_b_missiles = alive_blue_missiles
        return alive_red_missiles, alive_blue_missiles

    def get_state(self, side):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
            # own_missiles = self.Rmissiles
            # enm_missiles = self.Bmissiles
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
            # own_missiles = self.Bmissiles
            # enm_missiles = self.Rmissiles

        # alive_own_missiles = own_missiles.copy()
        # for missile in alive_own_missiles[:]:  # 遍历
        #     if missile.dead:
        #         alive_own_missiles.remove(missile)
        # alive_enm_missiles = enm_missiles.copy()
        # for missile in alive_enm_missiles[:]:  # 遍历
        #     if missile.dead:
        #         alive_enm_missiles.remove(missile)
        # if side == 'r':
        #     self.alive_r_missiles = alive_own_missiles
        #     self.alive_b_missiles = alive_enm_missiles
        # else:
        #     self.alive_r_missiles = alive_enm_missiles
        #     self.alive_b_missiles = alive_own_missiles

        alive_red_missiles, alive_blue_missiles = self.update_missile_state()
        if side == 'r':
            alive_own_missiles = alive_red_missiles
            alive_enm_missiles = alive_blue_missiles
        if side == 'b':
            alive_own_missiles = alive_blue_missiles
            alive_enm_missiles = alive_red_missiles

        # 目标存活标志
        target_alive = not adv.dead
        # 目标可见性标志 0 完全不可见 1 可获取角度信息 2 可获取全部信息
        target_observable = 2  # 难保不搞成one-hot的形式
        # 目标相对高度
        delta_alt_obs = adv.alt - own.alt
        # 目标相对方位角
        L_ = adv.pos_ - own.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        delta_psi = sub_of_radian(q_beta, own.psi)
        # 目标相对俯仰角
        delta_theta = sub_of_radian(q_epsilon, own.theta)
        # 目标相对距离
        dist = norm(L_)
        dist_obs = dist
        # 夹角
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        ATA = np.arccos(np.dot(L_, own.point_) / (dist * norm(own.point_) + 0.001))  # 防止计算误差导致分子>分母
        # 速度观测量
        v_own = v
        # 本机高度
        h_own = own.alt
        # 本机俯仰角
        sin_theta = sin(own.theta)
        cos_theta = cos(own.theta)
        # 本机滚转角
        sin_phi = sin(own.phi)
        cos_phi = cos(own.phi)

        # 剩余导弹量
        ammo = own.ammo

        # 雷达可跟踪标志
        if ATA < own.max_radar_angle and dist < own.max_radar_range:
            target_locked = 1
        else:
            target_locked = 0

        # 导弹中制导状态 bool 与 预计碰撞时间
        missile_in_mid_term = 0
        missile_time_to_hit_obs = 120
        if not alive_own_missiles:  # len(alive_own_missiles) == 0
            pass
        else:
            time2hits = np.ones(len(alive_own_missiles)) * 120
            for i, missile in enumerate(alive_own_missiles):
                time2hits[i] = missile.time2hit
                if missile.guidance_stage < 3:
                    missile_in_mid_term = 1
                    break
            missile_time_to_hit_obs = min(time2hits)

        # 首先找到所有存活的友方导弹是否由本机发射
        # 然后判断该导弹的 .guidance_stage是否<3

        # 目标雷达跟踪标志 bool
        alpha_enm = np.arccos(np.dot(-L_, adv.vel_) / (norm(adv.vel_) * dist + 0.01))  # 防止计算误差导致分子>分母
        if alpha_enm < own.max_radar_angle and dist < own.max_radar_range:
            locked_by_target = 1
        else:
            locked_by_target = 0

        # 告警信息 pi和-pi是突变点，置尾机动的时候不易训练，暂时不想用sin和cos，试试改为追一个东西
        if not alive_enm_missiles:
            warning = 0
            threat_delta_psi = pi  # pi 0
            threat_delta_theta = 0
            threat_distance = 30e3
        else:
            warnings = np.zeros(len(alive_enm_missiles))
            distances = np.ones(len(alive_enm_missiles)) * 30e3
            threat_delta_psis = np.zeros(len(alive_enm_missiles))
            threat_delta_thetas = np.zeros(len(alive_enm_missiles))
            for i, missile in enumerate(alive_enm_missiles):
                if missile.distance < missile.detect_range and missile.in_angle:
                    warnings[i] = 1
                    distances[i] = missile.distance
                    # # 作为追踪去训练，视图避开(pi -pi)突变点问题
                    # threat_delta_psis[i] = sub_of_radian(missile.q_beta, own.psi)
                    # threat_delta_thetas[i] = sub_of_radian(missile.q_epsilon, own.theta) # missile.q_epsilon
                    # 作为远离去训练，效果不是很好
                    threat_delta_psis[i] = sub_of_radian(pi + missile.q_beta, own.psi)
                    threat_delta_thetas[i] = -missile.q_epsilon

            # 告警标志 bool
            warning = bool(max(warnings))
            min_idx = int(np.argmin(distances))
            threat_delta_psi = threat_delta_psis[min_idx]
            threat_delta_theta = threat_delta_thetas[min_idx]
            threat_distance = distances[min_idx]

        p = own.p
        q = own.q
        r = own.r

        # 上一步动作
        act1_last, act2_last, act3_last = own.act_memory

        theta_v = own.theta_v
        psi_v = own.psi_v

        alpha_air = own.alpha_air
        beta_air = own.beta_air

        speed_T = adv.speed

        # 目标相对方位角速度 (rad/s) / 0.35 与 目标相对俯仰角速度 (rad/s) / 0.35
        vT_ = adv.vel_
        # vr_ = vT_ - v_
        # vr_radial = np.dot(vr_, L_) / dist  # 径向速度
        # vr_tangent_ = np.cross(L_, vr_) / dist # 目标周向速度矢量
        # omega_ = vr_tangent_/dist # 目标相对角速度矢量
        # down_ = np.array([0,-1,0])
        # L_left_ = np.cross(L_, down_)/dist
        # L_left_ = L_left_/norm(L_left_)
        # omega_vert_ = np.dot(omega_, down_)*1
        # omega_hor_ = omega_-omega_vert_
        # delta_psi_dot = np.dot(omega_vert_, down_) # 目标相对方位角速度
        # delta_theta_dot = -np.dot(omega_hor_, L_left_) # 目标相对俯仰角速度

        psi_vT = atan2(vT_[2], vT_[0])
        theta_vT = atan2(vT_[1], sqrt(vT_[0] ** 2 + vT_[2] ** 2))

        # 目标水平/垂直进入角
        AA_hor = sub_of_radian(psi_vT, q_beta)  # 向右飞为正
        AA_vert = sub_of_radian(theta_vT, q_epsilon)  # 向上飞为正

        d, d_hor, left_or_right = calc_intern_dist2cylinder(self.R_cage, own.pos_, own.psi_v, own.theta_v)

        # 原先将所有量打包成一个 numpy array，这里改为 dict 结构
        self.key_order = [
            "target_alive", "target_observable", "target_locked",
            "missile_in_mid_term", "locked_by_target", "warning",
            "target_information",  # 8
            "ego_main",  # 7
            "ego_control",  # 7
            "weapon",  # 1
            "threat",  # 3
            "border",  # 2
        ]

        one_side_states = {
            # 单独键（标量或布尔）
            "target_alive": bool(target_alive),
            "target_observable": int(target_observable),  # 0 完全不可见 1 角度信息可见 2 完全可见
            "target_locked": bool(target_locked),  # 已锁定敌机
            "missile_in_mid_term": bool(missile_in_mid_term),
            "locked_by_target": bool(locked_by_target),  # 敌锁定
            "warning": bool(warning),

            # 打包的向量 / 子组
            "target_information": np.array([
                float(delta_alt_obs),  # 0相对高度 m
                float(delta_psi),  # 1相对方位 rad
                float(delta_theta),  # 2相对俯仰角 rad
                float(dist_obs),  # 3距离 m
                float(ATA),  # 4夹角 rad
                float(speed_T),  # 5目标速度 m/s
                float(AA_hor),  # 6水平进入角 rad
                float(AA_vert)  # 7垂直进入角 rad
            ]),

            "ego_main": np.array([
                float(v_own),  # 0本机速度 m/s
                float(h_own),  # 1本机高度 m
                float(sin_theta),  # 2
                float(cos_theta),  # 3
                float(sin_phi),  # 4
                float(cos_phi),  # 5
                int(ammo)  # 6剩余导弹数量
            ]),

            "ego_control": np.array([
                float(p),  # 0 p rad/s act1_last
                float(q),  # 1 q rad/s act2_last
                float(r),  # 2 r rad/s act3_last
                float(theta_v),  # 3
                float(psi_v),  # 4
                float(alpha_air),  # 5 rad
                float(beta_air)  # 6 rad
            ]),

            "weapon": float(missile_time_to_hit_obs),

            "threat": np.array([
                float(threat_delta_psi),  # 0
                float(threat_delta_theta),  # 1
                float(threat_distance)  # 2
            ]),

            "border": np.array([
                float(d_hor),  # 0
                int(left_or_right),  # 1
            ])

        }

        return one_side_states

    def base_obs(self, side, pomdp=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        state = self.get_state(side)  # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["target_alive"] = 1  # 默认目标存活
        self.state_init["target_observable"] = 2  # 默认完全可见
        self.state_init["target_locked"] = 0
        self.state_init["missile_in_mid_term"] = 0
        self.state_init["locked_by_target"] = 0
        self.state_init["warning"] = 0
        self.state_init["target_information"] = np.array([0, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
        self.state_init["weapon"] = 120
        self.state_init["threat"] = np.array([pi, 0, 30e3])  # [pi,0,30e3]  [0,0,30e3]
        self.state_init["border"] = np.array([50e3, 0])

        # todo 重构pomdp的代码实现，尤其是state[x]的部分，state已经被改成字典了
        if pomdp:  # 只有在部分观测情况下需要添加屏蔽
            if uav.obs_memory is None:  # 假如不存在记忆，给默认值
                memory = copy.deepcopy(self.state_init)
            else:
                memory = uav.obs_memory

            ATA = state["target_information"][4]
            dist = state["target_information"][3]

            # 超出探测距离
            if dist > 160e3:  # 啥也看不到
                state["target_observable"] = 0
                state["target_information"] = memory["target_information"].copy()
            # 探测距离到近距
            elif dist > 10e3:
                if ATA > pi / 3 and state["locked_by_target"] == 0:  # 夹角>3/pi时观测不到目标
                    state["target_observable"] = 0
                    state["target_information"] = memory["target_information"].copy()
                elif ATA > pi / 3 and state["locked_by_target"] == 1:  # 被目标探测后有对目标的角度信息
                    state["target_observable"] = 1
                    for idx in (0, 3, 5, 6, 7):
                        state["target_information"][idx] = memory["target_information"][idx]
                else:
                    state["target_observable"] = 2  # 否则除了不能发射导弹，都是可见的
            else:
                state["target_observable"] = 2  # 10km类信息完全可见

        uav.obs_memory = copy.deepcopy(state)  # 更新残留值

        # 尺度缩放
        def scale_state(state_input):
            # 使用 deepcopy 避免修改传入对象
            s = copy.deepcopy(state_input)
            s["target_information"][0] /= 5e3
            s["target_information"][3] /= 10e3
            s["target_information"][5] /= 340
            s["ego_main"][0] /= 340
            s["ego_main"][1] /= 5e3
            s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
            s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
            s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340
            s["weapon"] /= 120
            s["threat"][2] /= 10e3
            s["border"][0] = min(1, s["border"][0] / 50e3)
            s["border"][1] = 0 if s["border"][0] == 1 else s["border"][1]
            return s

        observation = scale_state(state)
        self.obs_init = scale_state(self.state_init)
        return observation

    def get_reward(self, missiled_combat='Flase'):  # 策略选择器奖励
        if missiled_combat == True:
            # 添加导弹命中相关的奖励和惩罚
            pass
        '结果奖励部分'
        RUAV = self.RUAV
        BUAV = self.BUAV
        UAVs = [RUAV, BUAV]
        A = [0, 0]  # R, B
        rewards = [0, 0]  # R, B
        for i, UAV in enumerate(UAVs):  # UAVs[0]为红方，UAVs[1]为蓝方
            # adv = UAVs[1 - i]
            # # # 出界剩余时间惩罚：
            # t_last_max = 60
            # t_last = np.ones(6)
            # # 出界惩罚
            # if out_range(UAV):
            #     rewards[i] -= 100
            # # 超出限高惩罚
            # if UAV.pos_[1] >= max_height:
            #     rewards[i] -= 80
            # # 对手出界
            # if out_range(adv):
            #     rewards[i] += 10
            # # 被命中(不管是导弹还是"扫描枪")
            # if UAV.got_hit:
            #     rewards[i] -= 100
            # # 命中对手
            # if adv.got_hit:
            #     rewards[i] += 200  # 增加命中奖励
            # # 撞机
            # if UAV.crash:
            #     rewards[i] -= 70
            # # 平局
            # if self.t >= self.game_time_limit and not any([UAV.dead, adv.dead]):
            #     rewards[i] -= 10
            # test 目标追赶

            r_obs_n = self.base_obs('r')
            b_obs_n = self.base_obs('b')

            rewards[0] = 0
            rewards[1] = 0

            # rewards[0] = (1 - np.linalg.norm((r_obs_n[7] - RUAV.theta) / pi * 2 * 2)) * 100  # test
            # rewards[1] = (1 - np.linalg.norm((b_obs_n[7] - BUAV.theta) / pi * 2 * 2)) * 100  # test

            # rewards[0] += (1 - np.linalg.norm(r_obs_n[8] / pi)) * 100
            # rewards[1] += (1 - np.linalg.norm(b_obs_n[8] / pi)) * 100
            # 稀疏奖励
        # return rewards[0], rewards[1]
        # todo 奖励改成元组形式，第一项喂给经验池，第二项用作episode_return
        return (rewards[0], rewards[0]), (rewards[1], rewards[1])

    def get_target_by_id(self, target_id):
        for uav in self.UAVs:
            if uav.id == target_id:
                return uav
        return None

    def get_terminate(self):
        # 超时强制结束回合
        if self.t > self.game_time_limit:
            return True

        if all(self.UAV_hit):
            return True
        missile_dead_list = []
        uav_dead_list = []
        # battle和uav各自所属的missile没有同步，判断起来不方便，现改为所有发射的导弹都挂了,且无人机有一方坠落，
        # 则仿真就结束
        for missile in self.missiles:
            missile_dead_list.append(missile.dead)
        for uav in self.UAVs:
            uav_dead_list.append(uav.dead)

        if all(missile_dead_list) and any(uav_dead_list):
            return True

        # r_dead = [self.RUAV.got_hit]
        # b_dead = [self.BUAV.got_hit]
        # if self.running == False:
        #     return True
        # if all(r_dead) or all(b_dead):
        #     return True
        return False

    def out_range(self, UAV):
        horizontal_center = np.array([0, 0])
        position = UAV.pos_
        pos_h = np.array([position[0], position[2]])
        R_uav = norm(pos_h - horizontal_center)
        out = True
        if self.min_alt <= position[1] <= self.max_alt:
            if R_uav <= self.R_cage:
                out = False
        return out

    # 近距处理
    def close_range_kill(self):
        # todo 使用这个最好在奖励函数里面加上距离奖励
        for ruav in self.RUAVs:
            if ruav.dead:
                continue
            for buav in self.BUAVs:
                if buav.dead:
                    continue
                elif norm(ruav.pos_ - buav.pos_) >= 8e3:
                    continue
                else:
                    Lbr_ = ruav.pos_ - buav.pos_
                    Lrb_ = buav.pos_ - ruav.pos_
                    dist = norm(Lbr_)
                    # 求解hot-cold关系
                    cos_ATA_r = np.dot(Lrb_, ruav.vel_) / (dist * ruav.speed)
                    cos_ATA_b = np.dot(Lbr_, buav.vel_) / (dist * buav.speed)
                    # 双杀
                    if cos_ATA_r >= cos(pi / 3) and cos_ATA_b >= cos(pi / 3):
                        ruav.dead = True
                        buav.dead = True
                        ruav.got_hit = True
                        buav.got_hit = True
                        # todo win-lose
                    # 单杀
                    if cos_ATA_r >= cos(pi / 3) and cos_ATA_b < cos(pi / 3):
                        buav.dead = True
                        buav.got_hit = True
                    if cos_ATA_r < cos(pi / 3) and cos_ATA_b >= cos(pi / 3):
                        ruav.dead = True
                        ruav.got_hit = True

    def render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            # 传输飞机信息
            for UAV in self.UAVs:
                loc_LLH = UAV.lon, UAV.lat, UAV.alt
                if not UAV.dead:
                    if UAV.side == 'r':
                        color = 'Red'
                    elif UAV.side == 'b':
                        color = 'Blue'
                    else:
                        color = 'Black'

                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{UAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{UAV.phi * 180 / pi:.6f}|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Color={color}\n"
                    )
                    # data_to_send+=(
                    #     f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                    #     f"0|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                    #     f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange=100000, RadarHorizontalBeamwidth=120, RadarVerticalBeamwidth=90\n"
                    # )
                else:
                    data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                    # data_to_send += f"#-{UAV.id+1000}\n"

            # 传输导弹信息
            for missile in self.missiles:
                if hasattr(missile, 'dead') and missile.dead:
                    data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
                else:
                    # 记录导弹的位置
                    loc_m = NUE2LLH(missile.pos_[0], missile.pos_[1], missile.pos_[2], lon_o=o00[0], lat_o=o00[1],
                                    h_o=0)
                    if missile.side == 'r':
                        color = 'Orange'
                    elif missile.side == 'b':
                        color = 'Green'
                    else:
                        color = 'White'
                    data_to_send += f"#{send_t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                    f"Name=AIM-120C,Color={color}\n"

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            for UAV in self.UAVs:
                data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                # data_to_send += f"#{send_t:.2f}\n-{UAV.id+1000}\n"
            for missile in self.missiles:
                data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
            self.tacview.send_data_to_client(data_to_send)

    def visualize_cage(self, ):
        # 航路点画法
        # temp = np.zeros((19,3))
        # cage = np.zeros((19,3))
        # cage_dot_id = 10000
        # data_to_send=''
        # for i in range(18):
        #     temp[i] = np.array([self.R_cage*cos(i/18*2*pi), 5000, self.R_cage*sin(i/18*2*pi)])
        #     cage[i][:]=NUE2LLH(temp[i][0], temp[i][1], temp[i][2], lon_o=o00[0], lat_o=o00[1], h_o=0)
        #     cage_dot_id += 1
        #     data_to_send += (
        #                 f"{cage_dot_id},Type=Navaid+Static+Waypoint,"
        #                 f"T={cage[i][0]:.6f}|{cage[i][1]:.6f}|{cage[i][2]:.6f},Name=RedWP{i+1},Color=Red,"
        #                 f"Next={cage_dot_id+1}\n"
        #                 )
        # data_to_send += (
        #                 f"{cage_dot_id+1},Type=Navaid+Static+Waypoint,"
        #                 f"T={cage[0][0]:.6f}|{cage[0][1]:.6f}|{cage[0][2]:.6f},Name=RedWP19,Color=Red\n"
        #                 )
        # 雷达画法
        data_to_send = (
            f"10000,T={o00[0]}|{o00[1]}|{1000}"
            f",Type=Beam,ShortName=Cage,Color=White,Visible=1,Radius=0.0,RadarMode=1"
            f",RadarRange={self.R_cage},RadarHorizontalBeamwidth=360,RadarVerticalBeamwidth=0\n"
        )

        self.tacview.send_data_to_client(data_to_send)
        print('cage set')

    # 规则机动模型
    def track_behavior(self, ego_height, delta_psi):
        """
        追踪行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = delta_psi
        speed_cmd = 1.5 * 340
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def escape_behavior(self, ego_height, enm_delta_psi, warning, threat_delta_psi):
        """
        逃逸行为：返回 (heading_cmd, speed_cmd)
        没有导弹威胁的时候躲飞机，有导弹威胁的时候躲导弹
        """
        height_cmd = 7e3 - ego_height
        if warning:
            heading_cmd = np.clip(sub_of_radian(threat_delta_psi, pi), -pi / 2, pi / 2)
        else:
            heading_cmd = np.clip(sub_of_radian(enm_delta_psi, pi), -pi / 2, pi / 2)
        speed_cmd = 1.5 * 340
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def left_crank_behavior(self, ego_height, delta_psi):
        """
        crank 行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = np.clip(delta_psi - pi / 4, -pi / 2, pi / 2)
        # temp = 0.4 * (delta_psi - pi / 4) / (pi / 4) * 2
        # heading_cmd = np.clip(temp, -0.4, 0.4)
        speed_cmd = 1.1 * 340
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def right_crank_behavior(self, ego_height, delta_psi):
        """
        crank 行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = np.clip(delta_psi + pi / 4, -pi / 2, pi / 2)
        # temp = 0.4 * (delta_psi + pi / 4) / (pi / 4) * 2
        # heading_cmd = np.clip(temp, -0.4, 0.4)
        speed_cmd = 1.1 * 340
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def wander_behavior(self):
        """
        wander 随机漫步行为：返回 (alt_cmd, heading_cmd, speed_cmd)
        """
        alt_cmd = 3000 * np.random.uniform(-1, 1)
        heading_cmd = np.random.normal(0, 25 * pi / 180)
        speed_cmd = 300
        return np.array([alt_cmd, heading_cmd, speed_cmd])

    def back_in_cage(self, cmd, ego_pos_, ego_psi):
        height_cmd, heading_cmd, speed_cmd = cmd
        ego_height = ego_pos_[1]
        R_to_o00 = sqrt(ego_pos_[0] ** 2 + ego_pos_[2] ** 2)
        if ego_height > 13e3:
            height_cmd = -5000
        elif ego_height < 3e3:
            height_cmd = 5000
        if self.R_cage - R_to_o00 < 8e3:
            beta_of_o00 = atan2(-ego_pos_[2], -ego_pos_[0])
            heading_cmd = sub_of_radian(beta_of_o00, ego_psi)
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def decision_rule(self, ego_pos_, ego_psi, enm_delta_psi, distance, warning, threat_delta_psi, ally_missiles,
                      wander=0):
        ego_height = ego_pos_[1]
        # 输出为所需的绝对高度、相对方位和绝对速度
        # 是否有导弹可用
        has_missile_in_the_air = 0
        for missile in ally_missiles:
            if not missile.dead:
                has_missile_in_the_air = 1
                break
        # 是否被敌机导弹锁定
        if warning:
            should_escape = 1
        else:
            should_escape = 0
        action_n = np.array([0.0, 0.0, 400])

        # 行为决策：按原逻辑分支调用独立函数
        if distance > 40e3:
            cmd = self.track_behavior(ego_height, enm_delta_psi)
        elif not should_escape and has_missile_in_the_air:
            if enm_delta_psi >= 0:
                cmd = self.left_crank_behavior(ego_height, enm_delta_psi)
            else:
                cmd = self.right_crank_behavior(ego_height, enm_delta_psi)
        elif should_escape:
            # rel_psi_m 在 should_escape 时应已被设置
            cmd = self.escape_behavior(ego_height, enm_delta_psi, warning, threat_delta_psi)
        else:
            cmd = self.track_behavior(ego_height, enm_delta_psi)

        # 追踪任务的目标在散步
        if wander:
            cmd = self.wander_behavior()

        # 最高优先级：不许出圈
        cmd = self.back_in_cage(cmd, ego_pos_, ego_psi)

        return cmd


def launch_missile_if_possible(env, side='r'):
    """
    根据条件判断是否发射导弹
    """
    if side == 'r':
        uav = env.RUAV
        ally_missiles = env.Rmissiles
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        ally_missiles = env.Bmissiles
        target = env.RUAV

    waite = False
    for missile in ally_missiles:
        if not missile.dead:
            waite = True
            break

    if not waite:
        # 判断是否可以发射导弹
        if uav.can_launch_missile(target, env.t):
            # 发射导弹
            new_missile = uav.launch_missile(target, env.t, missile_class)
            uav.ammo -= 1
            new_missile.side = 'r' if side == 'r' else 'b'
            if side == 'r':
                env.Rmissiles.append(new_missile)
            else:
                env.Bmissiles.append(new_missile)
            env.missiles = env.Rmissiles + env.Bmissiles
            print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")
