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
    return os.path.dirname(os.path.abspath(__file__))
    # # 判断是否在 Jupyter Notebook 环境
    # try:
    #     shell = get_ipython().__class__.__name__  # ← 误报，不用管
    #     if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
    #         # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
    #         return os.getcwd()
    #     else:  # 其他 shell
    #         return os.path.dirname(os.path.abspath(__file__))
    # except NameError:
    #     # 普通 Python 脚本
    #     return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Envs.MissileModel1112 import *  # 传统三自由度动力学导弹模型
# from Envs.MissileModel260123 import * # 可垂直飞行的导弹模型
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
# report_move_time_rate = int(round(dt_maneuver / dt_move))

o00 = np.array([144.7, 13.4])  # 地理原点的经纬
# t = 0
g_ = np.array([0, -g, 0])
# theta_limit = 85 * pi / 180

R_cage = 100e3

# min_height = 0
# max_height = 15e3

R_birth = 40e3

horizontal_center = np.array([0, 0])


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Battle(object):
    def __init__(self, args, tacview_show=0):
        # super(Battle, self).__init__() 
        # self.p2p_control = False
        self.ego_side = None  # 我到底在哪一边
        self.shielded = False
        self.no_out = False
        self.alive_b_missiles = None
        self.alive_r_missiles = None
        self.alive_missiles = None
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
        # self.RmissilesTable = None
        # self.BmissilesTable = None
        # self.missilesTable = None
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
        self.max_alt_safe = 13e3
        self.min_alt_safe = 3e3
        self.min_alt_danger = 2e3
        self.min_alt = 0.5e3  # 1e3
        self.R_cage = getattr(self.args, 'R_cage', R_cage) if hasattr(self.args, 'R_cage') else R_cage

        # # 智能体的观察空间
        # self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      r_obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]

        self.RED_BIRTH_STATE = {'position': np.array([-R_birth * cos(0), 8000.0, -R_birth * sin(0)]),
                                        'psi': 0,
                                        'p2p': False
                                        }
        self.BLUE_BIRTH_STATE = {'position': np.array([-R_birth * cos(pi), 8000.0, -R_birth * sin(pi)]),
                                         'psi': pi,
                                         'p2p': False
                                         }
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
            self.visualize_cage()
            
    def set_ego_side(self, side='b'):
        self.ego_side = side
        
    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, seed=None, options=None, ego_side='b'):  # 重置位置和状态
        self.set_ego_side(ego_side)
        
        # [新增] 如果需要支持随机种子控制，可以在这里设置
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        if red_birth_state is None:
            red_birth_state = self.RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.BLUE_BIRTH_STATE

        self.BLUE_BIRTH_STATE = blue_birth_state
        self.RED_BIRTH_STATE = red_birth_state

        self.Rmissiles = []
        self.Bmissiles = []
        self.missiles = []
        self.alive_r_missiles = []
        self.alive_b_missiles = []
        self.alive_missiles = []
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
        # self.RmissilesTable = {}
        # self.BmissilesTable = {}
        # self.missilesTable = {}
        self.win = 0
        self.lose = 0
        self.draw = 0
        # 红方初始化
        for i in range(self.Rnum):
            UAV = UAVModel(dt=self.dt_move)
            UAV.state_memory = None
            UAV.last_state = None
            UAV.current_state = None
            UAV.launch_states = []
            UAV.launch_states_order = None
            UAV.init_ammo = red_init_ammo
            UAV.ammo = red_init_ammo
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.side = 'r'
            UAV.color = np.array([1, 0, 0])
            # 红方出生点
            UAV.pos_ = red_birth_state['position']  # np.array([-38841.96119795, 9290.02131746, -1686.95469864])
            # 不能出生在外面
            init_R = norm([UAV.pos_[0], UAV.pos_[2]])
            safe_R_cage = self.R_cage-5e3
            if init_R > self.R_cage-5e3:
                UAV.pos_[0] *= safe_R_cage/init_R
                UAV.pos_[2] *= safe_R_cage/init_R
            # 判断是否有自定义初始速度、theta、phi
            UAV.speed = red_birth_state.get('speed', 300)  # (UAV.speed_max - UAV.speed_min) / 2
            # speed = UAV.speed
            # mach, _ = calc_mach(speed, UAV.pos_[1])
            # UAV.mach = mach
            UAV.psi = red_birth_state['psi']
            UAV.theta = red_birth_state.get('theta', 0 * pi / 180)
            UAV.gamma = red_birth_state.get('phi', 0 * pi / 180)
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            UAV.got_hit = False
            UAV.escape_once = 0
            self.RUAVs.append(UAV)
            self.RUAVsTable[UAV.id] = {'entity': UAV, 'side': UAV.side, 'dead': UAV.dead}
        # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel(dt=self.dt_move)
            UAV.state_memory = None
            UAV.last_state = None
            UAV.current_state = None
            UAV.launch_states = []
            UAV.launch_states_order = None
            UAV.init_ammo = blue_init_ammo
            UAV.ammo = blue_init_ammo
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.side = 'b'
            UAV.color = np.array([0, 0, 1])
            # 蓝方出生点
            UAV.pos_ = blue_birth_state['position']  # np.array([38005.14540582, 6373.80721704, -1734.42509136])
            # 不能出生在外面
            init_R = norm([UAV.pos_[0], UAV.pos_[2]])
            safe_R_cage = self.R_cage-5e3
            if init_R > self.R_cage-5e3:
                UAV.pos_[0] *= safe_R_cage/init_R
                UAV.pos_[2] *= safe_R_cage/init_R
            UAV.speed = blue_birth_state.get('speed', (UAV.speed_max - UAV.speed_min) / 2)
            UAV.psi = blue_birth_state['psi']
            UAV.theta = blue_birth_state.get('theta', 0 * pi / 180)
            UAV.gamma = blue_birth_state.get('phi', 0 * pi / 180)
            UAV.psi = sub_of_radian(UAV.psi, 0)
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            UAV.got_hit = False
            UAV.escape_once = 0
            self.BUAVs.append(UAV)
            self.BUAVsTable[UAV.id] = {'entity': UAV, 'side': UAV.side, 'dead': UAV.dead}
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

    def launch_missile(self, side='r'):
        """
        立即发射导弹
        """
        if side == 'r':
            uav = self.RUAV
            ally_missiles = self.Rmissiles
            target = self.BUAV
        if side == 'b':
            uav = self.BUAV
            ally_missiles = self.Bmissiles
            target = self.RUAV
        else:
            print("请检查阵营")
            raise ValueError

        # 发射导弹
        if uav.ammo>0 and not uav.dead:
            new_missile = uav.launch_missile(target, self.t, missile_class)
            uav.ammo -= 1
            new_missile.side = 'r' if side == 'r' else 'b'
            if side == 'r':
                self.Rmissiles.append(new_missile)
            else:
                self.Bmissiles.append(new_missile)
            self.missiles = self.Rmissiles + self.Bmissiles
            # print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")


    def step(self, r_actions, b_actions):
        report_move_time_rate = int(round(self.dt_maneuver / self.dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]
        self.r_actions = r_actions.copy()
        self.b_actions = b_actions.copy()

        # # 记录 step 开始时的“已存在”导弹 id（用于判断导弹是否为在本 step 开始时就已存在）
        # initial_alive_ids = {m.id for m in (self.Rmissiles + self.Bmissiles) if not m.dead}

        # 在整个 maneuver step 开始时只重置一次 escape_once（不要在内层子步或导弹循环中再次重置）
        for UAV in self.UAVs:
            UAV.escape_once = 0

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
                if len(action)==4:
                    rudder = action[3]
                else:
                    rudder = None
                # print('target_height',target_height)

                if UAV.blue:
                    # 如果 BLUE_BIRTH_STATE 包含 p2p 则使用其值，否则为 False
                    p2p = self.BLUE_BIRTH_STATE.get('p2p', False)
                if UAV.red:
                    # 对红方同样兼容 RED_BIRTH_STATE 中可能存在的 p2p 字段
                    p2p = self.RED_BIRTH_STATE.get('p2p', False)

                # 防撞地系统
                if self.shielded:
                    # 临近撞地强制拉起
                    if UAV.alt < self.min_alt_safe + 1e3:
                        target_height = max(self.min_alt_safe + 1e3 - UAV.alt, target_height)
                        p2p = False
                        delta_heading = np.clip(delta_heading, -pi/3, pi/3)

                    # 不许超过限高
                    if UAV.alt > self.max_alt_safe:
                        target_height = min(self.max_alt_safe - UAV.alt, target_height)
                        p2p = False

                    # 速度过低强制加油门
                    if UAV.speed/340 < 0.5:
                        if p2p:
                            UAV.target_speed = 1
                        else:
                            UAV.target_speed = max(340, target_speed)
                    
                d, d_hor, left_or_right = calc_intern_dist2cylinder(self.R_cage, UAV.pos_, UAV.psi_v, UAV.theta_v)
                # 不准出界
                if self.no_out:
                    if d_hor < 8e3:
                        if left_or_right == 1:
                            delta_heading = min(-pi/2, delta_heading)
                        if left_or_right == -1:
                            delta_heading = max(pi/2, delta_heading)

                # 出界就炸
                if self.out_range(UAV):
                    UAV.dead = 1
                    # target_direction_ = horizontal_center - np.array(UAV.pos_[0], UAV.pos_[2])
                    # delta_heading = sub_of_radian(atan2(target_direction_[1], target_direction_[0]), UAV.psi)
                    # target_speed = 340
                    # # target_height = 0
                    # p2p = False # 只能用PID来按回

                UAV.move(target_height, delta_heading, target_speed, relevant_height=True, p2p=p2p, rudder=rudder)
                # 上一步动作
                # UAV.act_memory = np.array([action[0],action[1],action[2]])

            # 导弹移动
            self.update_missile_state() # 先把存活的导弹找出来
            # self.missiles = self.Rmissiles + self.Bmissiles
            for missile in self.alive_missiles[:]:  # 使用切片创建副本以允许删除
                target = self.get_target_by_id(missile.target_id)
                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹
                    missile.dead = True # todo 改成missile.target = None, 并在missile类里改成丢失目标飞直线，并且无法触发hit
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
                # 获取目标信息
                target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, last_ptt_)
                # 更新导弹制导阶段
                has_datalink = False
                for uav in self.UAVs:
                    # 找到载机，判断载机能否为导弹提供中制导
                    if uav.id == missile.launcher_id:
                        if uav.can_offer_guidance(missile, self.UAVs):
                            has_datalink = True
                last_vmt_, last_pmt_, _, _, _, _, _, _, _, _ = \
                    missile.step(target_info, dt=self.dt_move, datalink=has_datalink)
                # 毁伤判别
                vmt1 = norm(last_vmt_)
                # 第一个漂亮结果之前的导弹慢速自爆逻辑
                # if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                #     missile.dead = True
                # 新导弹慢速自爆逻辑
                if vmt1 < missile.speed_min \
                    and missile.t > 0.5 + missile.stage1_time + missile.stage2_time \
                        and last_pmt_[1] < 3000:
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

                if missile.dead == True and not hit:
                    target.escape_once = 1
                    # 目标逃脱
                # else:
                #     target.escape_once = 0

            # 飞机接收毁伤判别信息
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

                # 出界判别
                if self.crash(UAV):
                    UAV.dead = True
                # self.running = False

        # r_reward_n, b_reward_n = self.get_reward()
        # terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or self.crash(UAV):  # or self.out_range(UAV): ###
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

        # if terminate:
        #     self.running = False

        return 0, 0, 0, 0, 0  # 废弃不再使用了

    def update_missile_state(self):
        alive_r_missiles = [m for m in self.Rmissiles if not m.dead]
        alive_b_missiles = [m for m in self.Bmissiles if not m.dead]

        self.alive_r_missiles = alive_r_missiles
        self.alive_b_missiles = alive_b_missiles
        self.alive_missiles = alive_r_missiles + alive_b_missiles

    def get_state(self, side):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        if side == 'r':
            own = self.RUAV
            adv = self.BUAV

        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV


        alive_r_missiles, alive_b_missiles = self.alive_r_missiles, self.alive_b_missiles
        if side == 'r':
            alive_own_missiles = alive_r_missiles
            alive_enm_missiles = alive_b_missiles
        if side == 'b':
            alive_own_missiles = alive_b_missiles
            alive_enm_missiles = alive_r_missiles

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
        if ATA <= own.max_radar_angle and dist <= own.max_radar_range and target_alive:
            target_locked = 1
        else:
            target_locked = 0

        # 导弹中制导状态 bool 与 导弹发射间隔时间
        missile_in_mid_term = 0
        # 废弃，剩余命中时间
        # missile_time_since_shoot = 120
        # if not alive_own_missiles:  # len(alive_own_missiles) == 0
        #     pass
        # else:
        #     time2hits = np.ones(len(alive_own_missiles)) * 120
        #     for i, missile in enumerate(alive_own_missiles):
        #         time2hits[i] = missile.time2hit
        #         if missile.guidance_stage < 3:
        #             missile_in_mid_term = 1
        #             break
        #     missile_time_since_shoot = min(time2hits)
        missile_time_since_shoot = 120
        if not alive_own_missiles:  # len(alive_own_missiles) == 0
            pass
        else:
            time_since_shoots = np.ones(len(alive_own_missiles)) * 120
            for i, missile in enumerate(alive_own_missiles):
                time_since_shoots[i] = missile.t
                if missile.guidance_stage < 3:
                    missile_in_mid_term = 1
            missile_time_since_shoot = min(time_since_shoots)

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
        else:  # 敌导弹一发射就告警
            warnings = np.zeros(len(alive_enm_missiles))
            distances = np.ones(len(alive_enm_missiles)) * 30e3
            threat_delta_psis = np.zeros(len(alive_enm_missiles))
            threat_delta_thetas = np.zeros(len(alive_enm_missiles))
            for i, missile in enumerate(alive_enm_missiles):
                distances[i] = missile.distance
                if missile.distance < missile.detect_range and missile.in_angle:
                    warnings[i] = 1
                    threat_delta_psis[i] = sub_of_radian(pi + missile.q_beta, own.psi)
                    threat_delta_thetas[i] = -missile.q_epsilon
                elif locked_by_target:  # 导弹未进入告警距离但我机仍被敌机锁定
                    # 进入告警距离前用敌机方位作为导弹告警方位
                    threat_delta_psis[i] = delta_psi
                    threat_delta_thetas[i] = delta_theta + own.theta

            # 告警标志 bool
            warning = bool(max(warnings))
            min_idx = int(np.argmin(distances))
            threat_delta_psi = threat_delta_psis[min_idx]
            threat_delta_theta = threat_delta_thetas[min_idx]
            threat_distance = distances[min_idx]

        p = own.p
        q = own.q
        r = own.r


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
        # omega_vert_ = np.dot(omega_, down_) * 1
        # omega_hor_ = omega_ - omega_vert_
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
            "target_alive",  # 1 暂未使用
            "target_observable",  # 1 仅用于动作切换
            "target_locked",  # 1
            "missile_in_mid_term",  # 1 仅用于动作切换
            "locked_by_target",  # 1 仅用于动作切换
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "ego_control",  # 7
            "weapon",  # 1 仅用于动作切换
            "threat",  # 4
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
                # float(delta_alt_obs),  # 0相对高度 m
                # float(delta_psi),  # 1相对方位 rad
                float(cos(delta_psi)),  # 0相对方位 cos
                float(sin(delta_psi)),  # 1相对方位 sin
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

            "weapon": float(missile_time_since_shoot),

            "threat": np.array([
                float(cos(threat_delta_psi)),  # 0
                float(sin(threat_delta_psi)),  # 1
                float(threat_delta_theta),  # 2
                float(threat_distance),  # 3
            ]),

            # "threat": np.array([
            #     float(cos(threat_delta_psi) * cos(threat_delta_theta)),  # 0
            #     float(sin(threat_delta_theta)),  # 1
            #     float(sin(threat_delta_psi) * cos(threat_delta_theta))  # 2
            #     float(threat_distance),
            # }),

            "border": np.array([
                float(d_hor),  # 0
                int(left_or_right),  # 1
            ])

        }

        return one_side_states
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["target_information"][3] /= 10e3
        s["target_information"][5] /= 340
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340
        s["weapon"] /= 120
        s["threat"][3] /= 10e3
        s["border"][0] = min(1, s["border"][0] / 50e3)
        s["border"][1] = 0 if s["border"][0] == 1 else s["border"][1]
        return s
    
    def unscale_state(self, obs_input):
        """把 scale_state 的缩放还原。仅判断 key 是否存在（不再检查长度）。"""
        s = copy.deepcopy(obs_input)

        if "target_information" in s and s["target_information"] is not None:
            s["target_information"][3] = s["target_information"][3] * 10e3
            s["target_information"][5] = s["target_information"][5] * 340

        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"][0] = s["ego_main"][0] * 340
            s["ego_main"][1] = s["ego_main"][1] * 5e3

        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"][0] = s["ego_control"][0] * (2 * pi)
            s["ego_control"][1] = s["ego_control"][1] * (2 * pi)
            s["ego_control"][2] = s["ego_control"][2] * (2 * pi)

        if "weapon" in s and s["weapon"] is not None:
            s["weapon"] = s["weapon"] * 120

        if "threat" in s and s["threat"] is not None:
            s["threat"][3] = s["threat"][3] * 10e3

        if "border" in s and s["border"] is not None:
            s["border"][0] = s["border"][0] * 50e3

        return s

    def scale_state2(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["target_information"][3] /= 10e3
        s["target_information"][5] /= 340
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        
        # ammo scaling: 0->0, 1->1, 6->1.5
        ammo = s["ego_main"][6]
        if ammo <= 0:
            s["ego_main"][6] = 0
        else:
            s["ego_main"][6] = 0.9 + 0.1 * ammo

        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340
        s["weapon"] /= 120
        s["threat"][3] /= 10e3
        s["border"][0] = min(1, s["border"][0] / 50e3)
        s["border"][1] = 0 if s["border"][0] == 1 else s["border"][1]
        return s

    def unscale_state2(self, obs_input):
        """把 scale_state2 的缩放还原。仅判断 key 是否存在（不再检查长度）。"""
        s = copy.deepcopy(obs_input)

        if "target_information" in s and s["target_information"] is not None:
            s["target_information"][3] = s["target_information"][3] * 10e3
            s["target_information"][5] = s["target_information"][5] * 340

        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"][0] = s["ego_main"][0] * 340
            s["ego_main"][1] = s["ego_main"][1] * 5e3
            
            # ammo unscaling
            val = s["ego_main"][6]
            if val <= 0.01:
                s["ego_main"][6] = 0
            else:
                s["ego_main"][6] = (val - 0.9) * 10

        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"][0] = s["ego_control"][0] * (2 * pi)
            s["ego_control"][1] = s["ego_control"][1] * (2 * pi)
            s["ego_control"][2] = s["ego_control"][2] * (2 * pi)

        if "weapon" in s and s["weapon"] is not None:
            s["weapon"] = s["weapon"] * 120

        if "threat" in s and s["threat"] is not None:
            s["threat"][3] = s["threat"][3] * 10e3

        if "border" in s and s["border"] is not None:
            s["border"][0] = s["border"][0] * 50e3

        return s
        
    def base_obs(self, side, pomdp=0, reward_fn=0):  # 默认为完全可观测，设置pomdp后为部分可观测
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        # 如果是用来计算奖励的或是被critic用的，强制全局信息
        if reward_fn == 1:
            pomdp = 0
        
        # [修改] 获取当前真实状态
        state = self.get_state(side) # np.stack(self.get_state(side)) stack用于多架无人机

        # [新增] 增加时间戳，用于状态管理
        state['t'] = self.t

        # [新增] 智能的状态快照更新逻辑：防止同一时间步多次调用导致 last_state 被覆盖
        if uav.current_state is None:
            # 初始化
            uav.last_state = copy.deepcopy(state)
        else:
            # 通过比对时间戳判断是否为新的仿真步
            # 只有当时间推进了，才把旧的 current_state 归档为 last_state
            if abs(self.t - uav.current_state['t']) > 0.1:
                uav.last_state = copy.deepcopy(uav.current_state)

        uav.current_state = copy.deepcopy(state)

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["target_alive"] = 1  # 默认目标存活
        self.state_init["target_observable"] = 2  # 默认完全可见
        self.state_init["target_locked"] = 0
        self.state_init["missile_in_mid_term"] = 0
        self.state_init["locked_by_target"] = 0
        self.state_init["warning"] = 0
        # self.state_init["target_information"] = np.array([0, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["target_information"] = np.array([1, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
        self.state_init["weapon"] = 120
        # self.state_init["threat"] = np.array([pi, 0, 30e3])  # [pi,0,30e3]  [0,0,30e3]
        self.state_init["threat"] = np.array([1, 0, 0, 30e3])
        self.state_init["border"] = np.array([50e3, 0])

        if pomdp:  # 只有在部分观测情况下需要添加屏蔽
            # 1. 获取记忆 (Rolling Memory)
            if uav.state_memory is None:
                memory = copy.deepcopy(self.state_init)
            else:
                memory = uav.state_memory

            ATA = state["target_information"][4]
            dist = state["target_information"][3]

            # 2. 根据条件决定是 "全覆盖" 还是 "部分覆盖"
            
            # 情况A: 超出探测距离 -> 完全不可见
            if dist > 160e3:
                state["target_observable"] = 0
                # 整体覆盖：所有信息都用旧的
                state["target_information"] = memory["target_information"].copy()
            
            # 情况B: 距离较近
            elif dist > 10e3:
                # B1: 角度大 且 未被锁定 -> 完全不可见
                if ATA > pi / 3 and state["locked_by_target"] == 0:
                    state["target_observable"] = 0
                    # 整体覆盖
                    state["target_information"] = memory["target_information"].copy()
                
                # B2: 角度大 但 被锁定 (RWR告警) -> 部分可见
                elif ATA > pi / 3 and state["locked_by_target"] == 1:
                    state["target_observable"] = 1
                    # 【核心逻辑】只覆盖运动学信息 (dist, speed, AA)，保留当前真实的 RWR 信息 (角度, ATA)
                    # 因为 memory['dist'] 已经是上一步复制下来的旧值，所以这里再次复制依然是旧值
                    for idx in (3, 5, 6, 7):
                        state["target_information"][idx] = memory["target_information"][idx]
                
                # B3: 角度合适 -> 完全可见
                else:
                    state["target_observable"] = 2
                    # 不做任何覆盖，state 保持 get_state() 获取的最新真实值
            
            # 情况C: 极近距离 -> 完全可见
            else:
                state["target_observable"] = 2

        # 3. 更新记忆 (Rolling Update)
        # 无论刚才发生了什么，把处理后的 state 存入 memory
        # 如果刚才发生了部分覆盖，这里存入的就是 "旧dist + 新ATA" 的混合体
        # 下一步循环时，读取这个混合体，dist 依然是旧的
        # [修改] 仅在 pomdp 开启时更新记忆，防止奖励函数调用(pomdp=0)时泄露真实状态
        
        if reward_fn == 0: # 防止在奖励函数里面调用的时候泄露信息
            uav.state_memory = copy.deepcopy(state)

        # 在把 state 传入 scale_state 之前移除时间戳 't'
        if 't' in state:
            del state['t']

        observation = self.scale_state(state)
        self.obs_init = self.scale_state(self.state_init)
        return observation

    # def get_reward(self, missiled_combat='Flase'):  # 策略选择器奖励
    #     if missiled_combat == True:
    #         # 添加导弹命中相关的奖励和惩罚
    #         pass
    #     '结果奖励部分'
    #     RUAV = self.RUAV
    #     BUAV = self.BUAV
    #     UAVs = [RUAV, BUAV]
    #     A = [0, 0]  # R, B
    #     rewards = [0, 0]  # R, B
    #     for i, UAV in enumerate(UAVs):  # UAVs[0]为红方，UAVs[1]为蓝方

    #         r_obs_n = self.base_obs('r')
    #         b_obs_n = self.base_obs('b')

    #         rewards[0] = 0
    #         rewards[1] = 0

    #     # todo 奖励改成元组形式，第一项喂给经验池，第二项用作episode_return
    #     return (rewards[0], rewards[0]), (rewards[1], rewards[1])

    def get_target_by_id(self, target_id):
        for uav in self.UAVs:
            if uav.id == target_id:
                return uav
        return None

    # def get_terminate(self):
    #     # # 超时强制结束回合
    #     # if self.t > self.game_time_limit:
    #     #     return True

    #     if all(self.UAV_hit):
    #         return True
    #     missile_dead_list = []
    #     uav_dead_list = []
    #     # battle和uav各自所属的missile没有同步，判断起来不方便，现改为所有发射的导弹都挂了,且无人机有一方坠落，
    #     # 则仿真就结束
    #     for missile in self.missiles:
    #         missile_dead_list.append(missile.dead)
    #     for uav in self.UAVs:
    #         uav_dead_list.append(uav.dead)

    #     if all(missile_dead_list) and any(uav_dead_list):
    #         return True

    #     # r_dead = [self.RUAV.got_hit]
    #     # b_dead = [self.BUAV.got_hit]
    #     # if self.running == False:
    #     #     return True
    #     # if all(r_dead) or all(b_dead):
    #     #     return True
    #     return False

    def crash(self, UAV):
        if UAV.alt < self.min_alt:
            return True
        else:
            return False

    def too_high(self, UAV):
        if UAV.alt > self.max_alt:
            return True
        else:
            return False

    def out_range(self, UAV):
        horizontal_center = np.array([0, 0])
        position = UAV.pos_
        pos_h = np.array([position[0], position[2]])
        R_uav = norm(pos_h - horizontal_center)
        out = True
        if R_uav <= self.R_cage:
            out = False
        return out

    # # 近距处理
    # def close_range_kill(self,):
    #     for ruav in self.RUAVs:
    #         if ruav.dead:
    #             continue
    #         for buav in self.BUAVs:
    #             if buav.dead:
    #                 continue
    #             elif norm(ruav.pos_ - buav.pos_) >= 8e3:
    #                 continue
    #             else:
    #                 Lbr_ = ruav.pos_ - buav.pos_
    #                 Lrb_ = buav.pos_ - ruav.pos_
    #                 dist = norm(Lbr_)
    #                 # 求解hot-cold关系
    #                 cos_ATA_r = np.dot(Lrb_, ruav.vel_) / (dist * ruav.speed)
    #                 cos_ATA_b = np.dot(Lbr_, buav.vel_) / (dist * buav.speed)
    #                 # 双杀
    #                 if cos_ATA_r >= cos(pi / 3) and cos_ATA_b >= cos(pi / 3):
    #                     ruav.dead = True
    #                     buav.dead = True
    #                     ruav.got_hit = True
    #                     buav.got_hit = True
    #                 # 单杀
    #                 if cos_ATA_r >= cos(pi / 3) and cos_ATA_b < cos(pi / 3):
    #                     buav.dead = True
    #                     buav.got_hit = True
    #                 if cos_ATA_r < cos(pi / 3) and cos_ATA_b >= cos(pi / 3):
    #                     ruav.dead = True
    #                     ruav.got_hit = True


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
                        pilot = 'Maverick'
                    elif UAV.side == 'b':
                        color = 'Blue'
                        pilot = 'Ice'
                    else:
                        color = 'Black'
                        pilot = 'invader'

                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{UAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{UAV.phi * 180 / pi:.6f}|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Pilot={pilot},Color={color}\n"
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
                    data_to_send += (
                                    f"#{send_t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                                    f"{0.0:.6f}|{missile.theta * 180 / pi:.6f}|{missile.psi * 180 / pi:.6f},"
                                    f"Name=AIM-120C,Color={color}\n"
                                    )

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
        else:
            pass

    def end_render(self,):
        if self.tacview_show:
            self.tacview.end_render()
        else:
            pass

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
    def track_behavior(self, ego_height, delta_psi, speed_cmd=1.5*340):
        """
        追踪行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = delta_psi
        return np.array([height_cmd, heading_cmd, speed_cmd])

    def escape_behavior(self, ego_height, enm_delta_psi, warning, threat_delta_psi, speed_cmd=1.5 * 340):
        """
        逃逸行为：返回 (heading_cmd, speed_cmd)
        没有导弹威胁的时候躲飞机，有导弹威胁的时候躲导弹
        """
        height_cmd = 7e3 - ego_height
        if warning:
            heading_cmd = np.clip(sub_of_radian(threat_delta_psi, pi), -pi / 2, pi / 2)
        else:
            heading_cmd = np.clip(sub_of_radian(enm_delta_psi, pi), -pi / 2, pi / 2)

        return np.array([height_cmd, heading_cmd, speed_cmd])

    def left_crank_behavior(self, ego_height, delta_psi, speed_cmd = 1.1 * 340):
        """
        crank 行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = np.clip(delta_psi - pi / 4, -pi / 2, pi / 2)
        # temp = 0.4 * (delta_psi - pi / 4) / (pi / 4) * 2
        # heading_cmd = np.clip(temp, -0.4, 0.4)

        return np.array([height_cmd, heading_cmd, speed_cmd])

    def right_crank_behavior(self, ego_height, delta_psi, speed_cmd = 1.1 * 340):
        """
        crank 行为：返回 (heading_cmd, speed_cmd)
        """
        height_cmd = 7e3 - ego_height
        heading_cmd = np.clip(delta_psi + pi / 4, -pi / 2, pi / 2)
        # temp = 0.4 * (delta_psi + pi / 4) / (pi / 4) * 2
        # heading_cmd = np.clip(temp, -0.4, 0.4)

        return np.array([height_cmd, heading_cmd, speed_cmd])

    def wander_behavior(self, speed_cmd = 300):
        """
        wander 随机漫步行为：返回 (alt_cmd, heading_cmd, speed_cmd)
        """
        alt_cmd = 3000 * np.random.uniform(-1, 1)
        heading_cmd = np.random.normal(0, 25 * pi / 180)

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
            # print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")
        return 1
    else:
        return 0


def launch_missile_immediately(env, side='r', tabu=0):
    """
    立即发射导弹
    """
    new_missile_id = None
    if side == 'r':
        uav = env.RUAV
        ally_missiles = env.Rmissiles
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        ally_missiles = env.Bmissiles
        target = env.RUAV

    ego_state = env.get_state(uav.side)
    ATA = ego_state["target_information"][4]
    distance = ego_state["target_information"][3]
    AA_hor = ego_state["target_information"][6]
    target_locked = ego_state["target_locked"]

    # 发射导弹
    if uav.ammo>0 and not uav.dead:
        if not tabu or\
                target_locked and ego_state["weapon"]>=0.1 and ATA<=60 *pi/180:
            new_missile = uav.launch_missile(target, env.t, missile_class)
            uav.ammo -= 1

            # 记录导弹发射瞬间的 ATA、distance 和 AA_hor
            uav.launch_states_order = ['ATA', 'distance', 'AA_hor', 'target_locked', 't_go']
            uav.launch_states.append(np.array([ATA, distance, AA_hor, target_locked, ego_state["weapon"]]))

            new_missile.side = 'r' if side == 'r' else 'b'
            new_missile_id = new_missile.id
            if side == 'r':
                env.Rmissiles.append(new_missile)
            else:
                env.Bmissiles.append(new_missile)
            env.missiles = env.Rmissiles + env.Bmissiles
            # print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")
    
    return new_missile_id


def launch_missile_with_basic_rules(env, side='r'):
    """
    立即发射导弹
    """
    if side == 'r':
        uav = env.RUAV
        ally_missiles = env.Rmissiles
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        ally_missiles = env.Bmissiles
        target = env.RUAV

    ego_state = env.get_state(uav.side)
    target_locked = ego_state["target_locked"]
    alt = ego_state["ego_main"][1]
    dist = ego_state["target_information"][3]
    ATA = ego_state["target_information"][4]
    AA_hor = ego_state["target_information"][6]
    interval = ego_state["weapon"]

    # 发射导弹
    can_shoot = 0
    should_shoot = 0
    if uav.ammo>0 and not uav.dead and target_locked and interval>=3:
        can_shoot = 1
    
    if can_shoot:
        should_shoot = 0
        if dist<=5e3:
            should_shoot = (1-ATA/(60*pi/180))**2

        elif dist<=20e3 and abs(AA_hor)>=pi/2:
            should_shoot = (1-ATA/(60*pi/180))**2

        elif dist<=80e3 and interval>=20:
            should_shoot = (1-ATA/(60*pi/180))**2
    
    if dist > 10e3 and AA_hor<pi/6:
        should_shoot = 0
    if interval <= 30 and dist > 40e3:
        should_shoot = 0
    
    if dist > 45e3:
        should_shoot = 0


    if np.random.rand() < should_shoot: # np.random.rand() 生成的是在区间 [0, 1) 上的独立均匀分布
        new_missile = uav.launch_missile(target, env.t, missile_class)
        uav.ammo -= 1
        new_missile.side = 'r' if side == 'r' else 'b'
        if side == 'r':
            env.Rmissiles.append(new_missile)
        else:
            env.Bmissiles.append(new_missile)
        env.missiles = env.Rmissiles + env.Bmissiles
        # print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")

        return 1
    else:
        return 0
