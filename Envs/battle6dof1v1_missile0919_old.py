'''出生点改在外面指定'''

from random import random
import random
from gym import spaces
import copy
import jsbsim
import sys
import os
import importlib

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
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *

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

class Battle(object):
    def __init__(self, args, tacview_show=0):
        super(Battle, self).__init__()
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
        self.train_side_win = None
        self.train_side_lose = None
        self.train_side_draw = None
        self.max_alt = 15e3
        self.max_alt_safe = 13e3
        self.min_alt_safe= 3e3
        self.min_alt = 1e3
        self.R_cage = getattr(self.args, 'R_cage', R_cage) if hasattr(self.args, 'R_cage') else R_cage

        # # 智能体的观察空间
        # r_obs_n, b_obs_n = self.get_obs()
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
        
    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6,):  # 重置位置和状态

        if red_birth_state is None:
            red_birth_state = self.DEFAULT_RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.DEFAULT_BLUE_BIRTH_STATE

        self.Rmissiles = []
        self.Bmissiles = []
        self.missiles = []
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
        self.train_side_win = 0
        self.train_side_lose = 0
        self.train_side_draw = 0
        # 红方初始化
        for i in range(self.Rnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = red_init_ammo
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.side = "Red"
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
            self.RUAVsTable[UAV.id]=(UAV, UAV.side, UAV.dead)
        # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = blue_init_ammo
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.side = "Blue"
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
            self.BUAVsTable[UAV.id]=(UAV, UAV.side, UAV.dead)
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

    def get_obs_spaces(self, side):
        self.reset()
        
        # 智能体的观察空间
        # r_obs_n, b_obs_n = self.get_obs()
        obs_n = self.get_obs(side)
        # b_obs_n = self.get_obs('b')
        self.obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]
        return self.obs_spaces

    def step(self, r_actions, b_actions):
        # 输入动作（范围为[-1,1]
        self.t += dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                # print(action)
                target_height = action[0] # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                delta_heading = action[1] # 相对方位(弧度)
                target_speed = action[2] # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
                # print('target_height',target_height)
                # for i in range(int(self.dt_maneuver // dt_move)):
                UAV.move(target_height, delta_heading, target_speed, relevant_height=True)

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

    def get_state(self, side):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        更正，UAVmodel具有一部分判断目标可见性的功能，需要管理

        输入给1v1智能体的状态空间包含以下结构：
        0  目标可见性标志 bool
        1  目标相对高度 m
        2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        4  目标相对距离 m
        5  夹角 (rad)
        6  本机速度 m/s
        7  本机高度 m
        8  sinθ
        9  cosθ
        10 sinφ
        11 cosφ
        12 目标可跟踪标志 bool
        13 导弹中制导状态 bool
        14 导弹预计碰撞时间 s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        15 目标雷达跟踪标志 bool
        16 告警标志 bool
        17 最近来袭导弹相对方位角(rad), 如果没有导弹，在get_obs置为pi
        18 最近来袭导弹相对俯仰角(rad), 如果没有导弹，在get_obs置为0
        19 最近来袭导弹距离模糊值(0:<8km, 1: <20km, 2:>20km), 如果没有导弹，在get_obs置为最大
        20 目标雷达可探测标志 bool
        
        （废弃不用）    目标相对方位角速度 (rad/s) / 0.35
        （废弃不用）    目标相对俯仰角速度 (rad/s) / 0.35
        '''
        side = self.UAV_ids
        
        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
            own_missiles = self.Rmissiles
            enm_missiles = self.Bmissiles
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
            own_missiles = self.Bmissiles
            enm_missiles = self.Rmissiles
        # 目标可见性标志 bool
        target_observable = 1
        # 目标相对高度
        delta_alt_obs = adv.alt-own.alt
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
        alpha = np.arccos(np.dot(L_,v_)/(v*dist))
        # 速度观测量
        v_obs = v
        # 本机高度
        h_abs_obs = own.alt
        # 本机俯仰角
        sin_theta = sin(own.theta)
        cos_theta = cos(own.theta)
        # 本机滚转角
        sin_phi = sin(own.phi)
        cos_phi = cos(own.phi)
        # 雷达可跟踪标志
        target_tracked = 1

        # 目标相对方位角速度 (rad/s) / 0.35 与 目标相对俯仰角速度 (rad/s) / 0.35
        # vT_ = adv.vel_
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

        # 导弹中制导状态 bool
        missile_in_mid_term = 0
        pass # todo 需要由导弹的类实例报告

        # 导弹预计碰撞时间 / 30s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        missile_time_since_shoot = 4
        pass # todo 需要由导弹的类实例报告

        # 目标雷达跟踪标志 bool
        locked_by_target = 0
        pass # 和雷达跟踪目标的逻辑一起做

        # 告警标志 bool
        warning = 0
        pass # 取自目标发射导弹的雷达状态

        # 最近来袭导弹判断
        for missile in enm_missiles:
            pass

        # 最近来袭导弹相对方位角(rad), 如果没有导弹，在get_obs置为pi
        threat_delta_psi = pi
        pass

        # 最近来袭导弹相对俯仰角(rad), 如果没有导弹，在get_obs置为0
        threat_delta_theta = 0
        pass

        # 最近来袭导弹距离模糊值(0:<8km, 1: <20km, 2:>20km), 如果没有导弹，在get_obs置为最大
        threat_distance = 2
        pass

        # 目标雷达可探测标志
        target_radar_detects_me = 0
        
        one_side_obs = np.array([
            target_observable, 
            delta_alt_obs, 
            delta_psi, 
            delta_theta,
            dist_obs,
            alpha,
            v_obs,
            h_abs_obs,
            sin_theta,
            cos_theta,
            sin_phi,
            cos_phi,
            target_tracked,
            missile_in_mid_term,
            missile_time_since_shoot,
            locked_by_target,
            warning,
            threat_delta_psi,
            threat_delta_theta,
            threat_distance,
            target_radar_detects_me
        ])

        return one_side_obs

    def get_obs(self, side, pomdp=0):
        # 0  目标可见性标志 bool
        # 1  目标相对高度/5e3， 如果看不到，在get_obs置为0
        # 2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        # 3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        # 4  目标相对距离 / 10e3, 如果看不到，在get_obs置为最大
        # 5  夹角 (rad)
        # 6  本机速度 /340
        # 7  本机高度 /5e3
        # 8  sinθ
        # 9  cosθ
        # 10 sinφ
        # 11 cosφ
        # 12 目标可跟踪标志 bool
        # 13 导弹中制导状态 bool
        # 14 导弹预计碰撞时间 / 30s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        # 15 目标雷达跟踪标志 bool
        # 16 告警标志 bool
        # 17 最近来袭导弹相对方位角(rad), 如果没有导弹，在get_obs置为pi
        # 18 最近来袭导弹相对俯仰角(rad), 如果没有导弹，在get_obs置为0
        # 19 最近来袭导弹距离模糊值(0:<8km, 1: <20km, 2:>20km), 如果没有导弹，在get_obs置为最大
        # （废弃不用）    目标相对方位角速度 (rad/s) / 0.35
        # （废弃不用）    目标相对俯仰角速度 (rad/s) / 0.35
        # 20 目标雷达可探测标志 bool

        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV
        
        state = self.get_state(side) # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组
        
        if pomdp: # 只有在部分观测情况下需要添加屏蔽

            if not uav.obs_memory: # 假如不存在记忆，给默认值
                memory = np.array([0,0,0,0,100e3, 0, 1, 2, 0, 1, 0, 1, 0, 0, 4, 0, 0, 0, 0, 2, 1])
            else:
                memory = uav.obs_memory
            
            alpha = state[5]
            dist = state[4]

            # 超出探测距离
            if dist > 160e3: # 啥也看不到
                state[0, 12, 15] = 0
                state[1:5+1] = memory[1:5+1]
            # 跟踪距离到探测距离
            elif dist > 120e3: # 不能跟踪目标，
                state[12] = 0
                if alpha > 60*pi/180 and state[20]==0:  # 夹角>3/pi时观测不到目标
                    state[0] = 0
                    state[1:5+1] = memory[1:5+1]
                elif state[20] == 0: # 被目标探测后有对目标的角度信息
                    state[1, 4] = memory[1, 4]
                else:
                    pass # 否则除了不能发射导弹，都是可见的
            # 跟踪距离内超出角度
            elif dist > 10e3:
                if alpha > 60*pi/180 and state[20]==0: # 看不到对面
                    state[0] = 0
                    state[1:5+1] = memory[1:5+1]
                elif state[20] == 0: # 被目标探测后有对目标的角度信息
                    state[1, 4] = memory[1, 4]
                else:
                    pass # 否则所有信息可见
            else:
                pass # 10km类信息完全可见

        uav.obs_memory = state

        # 尺度缩放
        state_observed = state
        state_observed[1]/=5e3
        state_observed[4]/=10e3
        state_observed[6]/=340
        state_observed[7]/=5e3
        state_observed[14]/=30
        return state_observed

    # 进攻策略观测量
    def attack_obs(self, side):
        # 0  目标可见性标志 bool
        # 1  目标相对高度/5e3， 如果看不到，在get_obs置为0
        # 2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        # 3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        # 4  目标相对距离 / 10e3, 如果看不到，在get_obs置为最大
        # 5  夹角 (rad)
        # 6  本机速度 /340
        # 7  本机高度 /5e3
        # 8  sinθ
        # 9  cosθ
        # 10 sinφ
        # 11 cosφ
        # 12 目标可跟踪标志 bool
        mask = np.ones(20)
        mask[13:] = 0
        
        return self.get_obs(side)*mask
        
    # crank策略观测量
    def crank_obs(self, side):
        # 0  目标可见性标志 bool
        # 1  目标相对高度/5e3， 如果看不到，在get_obs置为0
        # 2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        # 3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        # 4  目标相对距离 / 10e3, 如果看不到，在get_obs置为最大
        # 5  夹角 (rad)
        # 6  本机速度 /340
        # 7  本机高度 /5e3
        # 8  sinθ
        # 9  cosθ
        # 10 sinφ
        # 11 cosφ
        # 12 目标可跟踪标志 bool
        # 13 导弹中制导状态 bool
        # 14 导弹预计碰撞时间 / 30s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        mask = np.ones(20)
        mask[15:] = 0
        return self.get_obs(side)*mask # todo

    # 规避策略观测量
    def escape_obs(self, side):
        pass # todo
        
    def attack_terminate_and_reward(self, side): # 进攻策略训练与奖励
        terminate = False
        state = self.get_state(side)
        alt = state[7]
        target_alt = state[1]+state[7]
        delta_psi = state[2]
        delta_theta = state[3]
        dist = state[4]
        alpha = state[5]
        speed = state[6]

        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        # 结束判断：超时/损毁
        if self.t > self.game_time_limit:
            terminate = True
        # if alpha > pi/2 and self.t > self.game_time_limit: # 超时了还没hot就结束
        #     terminate = True
        #     self.train_side_lose = 1
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.train_side_lose = 1

        if dist<5e3 and alpha< pi/12:
            terminate = True
            self.train_side_win = 1

        # 角度奖励
        r_angle = 1-alpha/(60*pi/180)  # 超出雷达范围就惩罚狠一点

        # 高度奖励
        pre_alt_opt = target_alt + np.clip((dist-10e3)/(40e3-10e3)*5e3, 0, 5e3)
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)

        r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
                    (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
                
        # 速度奖励
        speed_opt = 1.5*340
        r_speed = abs(speed-speed_opt)/(2*340)

        # 距离奖励
        r_dist = (dist<=10e3)*(dist-0)/(10e3-0)+\
                    (dist>10e3)*(1-(dist-10e3)/(50e3-10e3))

        # 平稳性惩罚
        r_roll = -abs(uav.p)/(2*pi) # 假设最大角速度是1s转一圈

        # 事件奖励
        reward_event = 0
        if self.train_side_lose:
            reward_event = -1
        if self.train_side_win:
            reward_event = 1

        reward = np.sum(np.array([2,1,1,1,5,0.5])*\
            np.array([r_angle, r_alt, r_speed, r_dist, reward_event, r_roll]))

        if terminate:
            self.running = False
        
        return terminate, reward, reward_event


    def get_reward(self, missiled_combat='Flase'): # 策略选择器奖励
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
            r_obs_n = self.get_obs('r')
            b_obs_n = self.get_obs('b')
            # r_obs_n, b_obs_n = self.get_obs()
            # rewards[0] = (1 - np.linalg.norm(r_obs_n[7] / pi * 2)) * 100
            rewards[0] = (1 - np.linalg.norm((r_obs_n[7] - RUAV.theta) / pi * 2 * 2)) * 100  # test
            rewards[1] = (1 - np.linalg.norm((b_obs_n[7] - BUAV.theta) / pi * 2 * 2)) * 100  # test
            # rewards[1] = (1-np.linalg.norm(BUAV.theta/pi*2)) * 100
            rewards[0] += (1 - np.linalg.norm(r_obs_n[8] / pi)) * 100
            rewards[1] += (1 - np.linalg.norm(b_obs_n[8] / pi)) * 100

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
    
    def render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t+t_bias
            data_to_send = ''
            # 传输飞机信息
            for UAV in self.UAVs:
                loc_LLH = UAV.lon, UAV.lat, UAV.alt
                if not UAV.dead:
                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{UAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{UAV.phi * 180 / pi:.6f}|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Color={UAV.side}\n"
                    )
                    data_to_send+=(
                        f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"0|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Type=Beam, Color={UAV.side},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange=100000, RadarHorizontalBeamwidth=120, RadarVerticalBeamwidth=20\n"
                    )
                else:
                    data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                    data_to_send += f"#-{UAV.id+1000}\n"

            # 传输导弹信息
            for missile in self.missiles:
                if hasattr(missile, 'dead') and missile.dead:
                    data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
                else:
                    # 记录导弹的位置
                    loc_m = NUE2LLH(missile.pos_[0], missile.pos_[1], missile.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
                    if missile.side == 'red':
                        color = 'Orange'
                    else:
                        color = 'Green'
                    data_to_send += f"#{send_t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                    f"Name=AIM-120C,Color={color}\n"

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t+t_bias
            data_to_send = ''
            for UAV in self.UAVs:
                data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{UAV.id+1000}\n"
            for missile in self.missiles:
                data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
            self.tacview.send_data_to_client(data_to_send)
        

    def visualize_cage(self,):
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