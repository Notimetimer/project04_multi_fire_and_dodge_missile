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


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Envs.MissileModel1112 import *  # 传统三自由度动力学导弹模型
# from Envs.MissileModel260123 import * # 可垂直飞行的导弹模型
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Math_calculates.Calc_dist2border import calc_intern_dist2circle, polygon_fences
from Visualize.tacview_visualize2 import *
from shapely.geometry import Point
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Utilities.LocateDirAndAgents import *

from TrainAndTests.Controls.UPolicyWrapper import *
from TrainAndTests.Controls.FlightControl_Train_dual_a_out2 import *
# 临时关闭端到端控制，加回PID控制
from Envs.UAVmodel6d import UAVModel
# from Envs.UAVmodel6d0309 import UAVModel

# 调用黑名单：删除 PPOHybrid，防止污染命名空间导致外层调用时 IDE 混淆
try:
    del PPOHybrid
except NameError:
    pass

g = 9.81
dt_maneuver = 0.2  # 0.02 0.8 0.2
dt_move = 0.02
# report_move_time_rate = int(round(dt_maneuver / dt_move))

o00 = np.array([144.7, 13.4])  # 地理原点的经纬
# t = 0
g_ = np.array([0, -g, 0])
# theta_limit = 85 * pi / 180

R_cage = 62.00e3

# min_height = 0
# max_height = 15e3

R_birth = 55e3

horizontal_center = np.array([0, 0])


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Battle(object):
    def __init__(self, args, tacview_show=0, vertices=None):
        # super(Battle, self).__init__() 
        # self.e2e_control = False
        self.vertices = vertices
        if self.vertices is not None:
            self.polygon_fences = polygon_fences(self.vertices)
        self.horizontal_center = horizontal_center
        # 加载训练好的模型
        import torch
        device = torch.device("cpu")
        self.control_env = track_env(tacview_show=0)
        action_dims_dict = {'cont': 4, 'cat': [], 'bern': 0}
        state_dim = 7+8+4
        hidden_dim = [128, 128]
        action_bound = np.array([[-1.1,1.1],[-1.1,1.1],[-1.1,1.1],[-0.2,1.2]])  # aileron, elevator, rudder, throttle
        policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        self.control_actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
        actor_path = os.path.join(project_root, "Controller/NNController", "01.pt")
        # pre_log_dir = os.path.join(project_root, "logs/control")
        # log_dir = os.path.join(pre_log_dir, "FlightControl_parallel-run-20260310-225856")
        # actor_path = load_actor_from_log(log_dir, number=None)
        sd = th.load(actor_path, map_location=device, weights_only=True)
        self.control_actor.load_state_dict(sd)

        # 正常的开始
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
        self.t = 0.0
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
        self.min_alt_safe = 1e3 # 3e3
        self.min_alt_danger = 5e2 # 2e3
        self.min_alt = 100 # 0.5e3  # 1e3
        self.R_cage0 = getattr(self.args, 'R_cage', R_cage) if hasattr(self.args, 'R_cage') else R_cage
        self.R_cage = self.R_cage0
        self.half_R_cage = self.R_cage / 2
        self.RWR_distance = 60e3 # 最大告警距离
        self.RWR_ranging_distance = self.RWR_distance # 最大告警可测距

        # # 智能体的观察空间
        key_order = [
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
            "out_locked", # 1
        ]
        self.full_obs_dim = 1*6+8+7+7+1+4+2+1

        self.RED_BIRTH_STATE = {'position': np.array([R_birth * cos(-pi/2), 8000.0, R_birth * sin(-pi/2)]),
                                        'psi': pi/2,
                                        'e2e': False
                                        }
        self.BLUE_BIRTH_STATE = {'position': np.array([R_birth * cos(pi/2), 8000.0, R_birth * sin(pi/2)]),
                                         'psi': -pi/2,
                                         'e2e': False
                                         }
        self.tacview_show = tacview_show

        # 动作平滑参数
        self.action_ema_beta = 0.006 ** (self.dt_maneuver / 2.0)  # 2秒后旧动作权重降至 0.006 0.2
        self.r_actions_ema = None
        self.b_actions_ema = None
        
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
            self.visualize_cage()
            
    def set_ego_side(self, side='b'):
        self.ego_side = side
        
    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, seed=None, options=None, ego_side='b'):  # 重置位置和状态
        self.R_cage = self.R_cage0
        if self.vertices is not None:
            self.horizontal_center = np.mean(self.vertices, axis=0)
        else:
            self.horizontal_center = horizontal_center
        self.set_ego_side(ego_side)
        # debug
        self.r_can_guide = 0
        self.b_can_guide = 0
        
        # [新增] 平均态势分变量初始化
        self.r_dist_seq = []
        self.b_dist_seq = []
        self.last_record_t = -1.0

        # 重置动作 EMA 变量
        self.r_actions_ema = None
        self.b_actions_ema = None
        
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
        self.r_hitter_id = None
        self.b_hitter_id = None
        self.win = 0
        self.lose = 0
        self.draw = 0
        # 无人机统一初始化
        for i in range(self.Rnum + self.Bnum):
            is_red = i < self.Rnum
            index_in_side = i if is_red else i - self.Rnum
            
            UAV = UAVModel(dt=self.dt_move)
            UAV.dead_time = None # 战斗机死的时间
            UAV.state_memory = None
            UAV.action_memory = np.array([0, 0, 340])
            UAV.last_state = None
            UAV.current_state = None
            UAV.launch_states = []
            UAV.launch_states_order = None
            UAV.init_ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.fired = 0
            UAV.id = (index_in_side + 1) if is_red else (index_in_side + 201)
            UAV.killer_id = None
            UAV.red = is_red
            UAV.blue = not is_red
            UAV.side = 'r' if is_red else 'b'
            UAV.color = np.array([1, 0, 0]) if is_red else np.array([0, 0, 1])
            
            birth_state = red_birth_state if is_red else blue_birth_state
            
            # 出生点 (注意：使用 copy 防止修改外部传入的字典数据)
            UAV.pos_ = birth_state['position'].copy()
            if self.vertices is not None:
                # 对长方形边界，直接分维度压进边界内
                min_x = min(v[0] for v in self.vertices) + 5e3
                max_x = max(v[0] for v in self.vertices) - 5e3
                min_z = min(v[1] for v in self.vertices) + 5e3
                max_z = max(v[1] for v in self.vertices) - 5e3
                UAV.pos_[0] = np.clip(UAV.pos_[0], min_x, max_x)
                UAV.pos_[2] = np.clip(UAV.pos_[2], min_z, max_z)
            else:
                # 不能出生在外面
                init_R = norm([UAV.pos_[0], UAV.pos_[2]])
                safe_R_cage = self.R_cage - 5e3
                if init_R > safe_R_cage:
                    scale_factor = safe_R_cage / init_R
                    UAV.pos_[0] *= scale_factor
                    UAV.pos_[2] *= scale_factor
                
            # 判断是否有自定义初始速度、theta、phi
            default_speed = 300 if is_red else (UAV.speed_max - UAV.speed_min) / 2
            UAV.speed = birth_state.get('speed', default_speed)
            
            UAV.psi = birth_state['psi']
            UAV.theta = birth_state.get('theta', 0 * pi / 180)
            UAV.gamma = birth_state.get('phi', 0 * pi / 180)
            UAV.psi = sub_of_radian(UAV.psi, 0)
            
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            UAV.got_hit = False
            UAV.escape_once = 0
            UAV.about_to_fire = 0
            
            if is_red:
                self.RUAVs.append(UAV)
                self.RUAVsTable[UAV.id] = {'entity': UAV, 'side': UAV.side, 'dead': UAV.dead}
            else:
                self.BUAVs.append(UAV)
                self.BUAVsTable[UAV.id] = {'entity': UAV, 'side': UAV.side, 'dead': UAV.dead}
                
            UAV.lock_on = 0
            UAV.last_locked_by_target_time = 0
            
        self.running = True
        self.UAVs = self.RUAVs + self.BUAVs
        self.UAVsTable = {**self.RUAVsTable, **self.BUAVsTable}
        self.UAV_ids = [UAV.id for UAV in self.UAVs]
        self.UAV_hit = [False for _ in range(len(self.UAVs))]

        # todo 1v1的残留
        self.RUAV = self.RUAVs[0]
        self.BUAV = self.BUAVs[0]


    def close(self):
        """
        关闭环境并清理资源
        """
        if self.tacview_show:
            try:
                self.tacview.socket.close()
            except:
                pass
        # 清理 UAV 和导弹引用
        self.UAVs = []
        self.missiles = []


    def step(self, r_actions, b_actions):
        # # 变步长
        # alive_missiles = [m for m in self.missiles if not m.dead]
        # if not alive_missiles:
        #     self.dt_move = 0.04 # 再高就有撞地风险了
        # else:
        #     self.dt_move = 0.04
        #     for m in alive_missiles:
        #         if m.distance < 5000: # 5km
        #             self.dt_move = 0.02
        #             break

        full_move_steps, remainder = divmod(self.dt_maneuver, self.dt_move)
        substep_dts = [self.dt_move] * int(full_move_steps)
        if remainder > 1e-12:
            substep_dts.append(remainder)
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 3)  # 保留3位小数

        actions = [r_actions] + [b_actions]

        # 动作 EMA 平滑
        heading_dead = np.deg2rad(15.0)  # 航向变化小于 15 度时不平滑
        if self.r_actions_ema is None:
            self.r_actions_ema = r_actions.copy()
        else:
            # 高度仍做 EMA
            self.r_actions_ema[0] = self.action_ema_beta * self.r_actions_ema[0] + (1 - self.action_ema_beta) * r_actions[0]
            # 航向：小指令或符号切换时直出，避免相位滞后导致蛇行
            if abs(r_actions[1]) < heading_dead or np.sign(r_actions[1]) != np.sign(self.r_actions_ema[1]):
                self.r_actions_ema[1] = r_actions[1]
            else:
                self.r_actions_ema[1] = self.action_ema_beta * self.r_actions_ema[1] + (1 - self.action_ema_beta) * r_actions[1]
        if self.b_actions_ema is None:
            self.b_actions_ema = b_actions.copy()
        else:
            # 高度仍做 EMA
            self.b_actions_ema[0] = self.action_ema_beta * self.b_actions_ema[0] + (1 - self.action_ema_beta) * b_actions[0]
            # 航向：小指令或符号切换时直出，避免相位滞后导致蛇行
            if abs(b_actions[1]) < heading_dead or np.sign(b_actions[1]) != np.sign(self.b_actions_ema[1]):
                self.b_actions_ema[1] = b_actions[1]
            else:
                self.b_actions_ema[1] = self.action_ema_beta * self.b_actions_ema[1] + (1 - self.action_ema_beta) * b_actions[1]
        # 使用平滑后的动作执行
        actions = [self.r_actions_ema] + [self.b_actions_ema]

        # 保存原始动作（用于记录或分析）
        self.r_actions = r_actions.copy()
        self.b_actions = b_actions.copy()

        # # 记录 step 开始时的“已存在”导弹 id（用于判断导弹是否为在本 step 开始时就已存在）
        # initial_alive_ids = {m.id for m in (self.Rmissiles + self.Bmissiles) if not m.dead}

        # 在这里执行保护系统计算，并覆写原本的飞行动作指令
        for UAV, action in zip(self.UAVs, actions):
            UAV.escape_once = 0
            
            delta_heading = action[1]
            target_height = action[0]
            target_speed = action[2]

            e2e = False
            if UAV.blue:
                e2e = self.BLUE_BIRTH_STATE.get('e2e', False)
            if UAV.red:
                e2e = self.RED_BIRTH_STATE.get('e2e', False)

            # 防撞地系统
            if self.shielded:
                # 临近撞地强制拉起
                if UAV.alt < self.min_alt_safe + 1e3:
                    target_height = max(self.min_alt_safe + 1e3 - UAV.alt, target_height)
                    e2e = False
                    delta_heading = np.clip(delta_heading, -pi/3, pi/3)

                # 不许超过限高
                if UAV.alt > self.max_alt_safe:
                    target_height = min(self.max_alt_safe - UAV.alt, target_height)
                    e2e = False

                # 速度过低强制加油门
                if UAV.speed/340 < 0.5:
                    if e2e:
                        UAV.target_speed = 1
                    else:
                        target_speed = max(340, target_speed)
                        UAV.target_speed = target_speed
                
            # 不准出界
            if self.no_out:
                if self.vertices is not None:
                    d_hor, left_or_right = self.polygon_fences.calc_intern_dist(UAV.pos_, UAV.psi_v)
                else:
                    d_hor, left_or_right = calc_intern_dist2circle(self.R_cage, UAV.pos_, UAV.psi_v)
                if d_hor < 8e3: # 8e3
                    if left_or_right > 0:
                        delta_heading = min(-pi/2, delta_heading)
                    else:
                        delta_heading = max(pi/2, delta_heading)

            # 将调整后的动作写回，Python引用就地修改
            action[0] = target_height
            action[1] = delta_heading
            action[2] = target_speed

            UAV.target_heading = sub_of_radian(UAV.psi + action[1], 0)
            # # 调试信息：计算相对方位角
            # other_uav = self.UAVs[1] if UAV == self.UAVs[0] else self.UAVs[0]
            # angle_to_other_rad = atan2(other_uav.pos_[2] - UAV.pos_[2], other_uav.pos_[0] - UAV.pos_[0])
            # print(f"UAV {UAV.id}({UAV.side}) target_heading: {np.degrees(UAV.target_heading):.2f} deg, angle_to_other: {np.degrees(angle_to_other_rad):.2f} deg")
            # print("--")

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        self.r_can_guide = 0
        self.b_can_guide = 0

        for substep_dt in substep_dts:
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                delta_heading = action[1]
                target_height = action[0]  # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                target_speed = action[2]  # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值

                # 计算当前子步下，实际机头指向与固定目标点之间的动态差角
                dynamic_delta_psi = sub_of_radian(UAV.target_heading, UAV.psi)

                rudder = None

                # 出界就炸
                if self.out_cage(UAV):
                    UAV.dead = 1

                e2e=1
                # 临时改动，关闭端到端控制，改为PID
                if e2e==1:
                    # 在这里插入强化学习的控制器
                    # 实时从 UAV 对象读取物理量，与训练环境 get_state() 完全对齐
                    # （不能使用 UAV.current_state 字典缓存——它在整个内层子步循环中不更新，是陈旧的旧状态）
                    _ego_main_realtime = np.array([
                        float(UAV.speed),          # 0 本机速度 m/s
                        float(UAV.alt),            # 1 本机高度 m
                        float(cos(UAV.theta)),     # 2
                        float(sin(UAV.theta)),     # 3
                        float(cos(UAV.phi)),       # 4
                        float(sin(UAV.phi)),       # 5
                        0,                         # 6 弹药量（控制器训练时始终置0）
                    ])
                    # 对齐训练环境: ego_control[4] 是目标航向与 速度矢量(psi_v) 的差角
                    _delta_psi_v = sub_of_radian(UAV.target_heading, UAV.psi_v)
                    _ego_control_realtime = np.array([
                        float(UAV.p),             # 0 p rad/s
                        float(UAV.q),             # 1 q rad/s
                        float(UAV.r),             # 2 r rad/s
                        float(UAV.theta_v),       # 3
                        float(_delta_psi_v),      # 4 目标航向与速度方向差角
                        float(UAV.alpha_air),     # 5 rad
                        float(UAV.beta_air),      # 6 rad
                        float(UAV.Ny),            # 7
                    ])
                    control_input_state = {
                        "ego_main": _ego_main_realtime,
                        "ego_control": _ego_control_realtime,
                        "flight_cmd": np.array([
                            cos(dynamic_delta_psi),
                            sin(dynamic_delta_psi),
                            np.clip(target_height, -5000, 5000),   # 与训练环境 clip 对齐
                            target_speed - UAV.speed,
                        ])
                    }
                    # 拼接完成后再从control_input_state调用self.control_env.scale_state 做缩放
                    scaled_control_input_state = self.control_env.scale_state(control_input_state)
                    # 最后按照 ["ego_main", "ego_control", "flight_cmd"] 顺序拼接为一个np.array作为control_input
                    control_input = np.concatenate([
                        scaled_control_input_state["ego_main"],
                        scaled_control_input_state["ego_control"],
                        scaled_control_input_state["flight_cmd"]
                    ])
                    # # debug
                    # if self.t > 80 and self.t % 3 == 0:
                    #     print(UAV.side)
                    #     print(control_input_state["flight_cmd"])
                    #     print()
                    # 控制器作用
                    control_action, _, _, _ = self.control_actor.get_action(control_input, explore=False)
                    aileron, elevator, rudder, throttle = control_action['cont']

                    # 弱化俯仰控制
                    elevator*=0.75
                    if UAV.mach > 0.85:
                        elevator=np.clip(elevator, -0.72, 0.72)

                    UAV.move(elevator, aileron, throttle, relevant_height=True, e2e=True, rudder=rudder, dt=substep_dt)
                else:
                    UAV.move(target_height, delta_heading, target_speed, relevant_height=True, e2e=0, rudder=rudder)

            # 导弹移动
            self.update_missile_state() # 先把存活的导弹找出来
            # self.missiles = self.Rmissiles + self.Bmissiles
            for missile in self.alive_missiles[:]:  # 使用切片创建副本以允许删除
                target = self.get_target_by_id(missile.target_id)
                missile.target = target
                
                # 1v1加快仿真速度用的，多对多得去掉
                # if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                #     missile.dead = True
                #     continue
                # elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹
                #     missile.dead = True # todo 改成missile.target = None, 并在missile类里改成丢失目标飞直线，并且无法触发hit
                #     continue
                
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
                            if uav.red:
                                self.r_can_guide = max(1, self.r_can_guide)
                            else:
                                self.b_can_guide = max(1, self.b_can_guide)
                last_vmt_, last_pmt_, _, _, _, _, _, _, _, _ = \
                    missile.step(target_info, dt=substep_dt, datalink=has_datalink)
                
                # 把导弹进入末制导的时间记录到导弹上
                if missile.A_pole_moment:
                    missile.A_pole_moment_time = self.t
                    missile.reward_function_info["time"] = self.t
                else:
                    missile.A_pole_moment_time = None

                # 毁伤判别
                vmt1 = norm(last_vmt_)
                # 导弹慢速自爆，节省计算量
                # if vmt1 < missile.speed_min \
                #     and missile.t > 0.5 + missile.stage1_time + missile.stage2_time \
                #         and last_pmt_[1] < 15e3: # 3000
                #     missile.dead = True
                if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                    missile.dead = True
                if missile.t > missile.t_max:  # 超时自爆
                    missile.dead = True
                # 超出电池工作时间引信不工作
                if 0 + substep_dt <= missile.t <= missile.t_max and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=substep_dt, kill_range=missile.kill_range)
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
                        target.killer_id = missile.id
                        self.UAV_hit[self.UAV_ids.index(target.id)] = True
                        if missile.side=='r':
                            self.r_hitter_id = missile.id
                        if missile.side=='b':
                            self.b_hitter_id = missile.id

                if missile.dead == True and not target.dead:
                    target.escape_once = 1
                    missile.miss = 1

            # 飞机接收毁伤判别信息
            for i, UAV in enumerate(self.UAVs):
                # 飞机被导弹命中判断
                if UAV.red:
                    enm = self.BUAV
                if UAV.blue:
                    enm = self.RUAV
                if self.UAV_hit[i]:
                    UAV.dead = True
                    UAV.got_hit = True
                # 其他毁伤判断
                enm = self.UAVs[1 - i]
                pt_ = enm.pos_
                L_ = pt_ - UAV.pos_
                distance = np.linalg.norm(L_)

                # 出界判别
                if self.crash(UAV):
                    UAV.dead = True
                # self.running = False

        # r_reward_n, b_reward_n = self.get_reward()
        # terminate = self.get_terminate()

        "可能是重复逻辑，需要排查"
        for UAV in self.UAVs:
            if UAV.got_hit or self.crash(UAV):  # or self.out_cage(UAV): ###
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
            ego = self.RUAV
            enm = self.BUAV

        else:  # if side=='b':
            ego = self.BUAV
            enm = self.RUAV


        alive_r_missiles, alive_b_missiles = self.alive_r_missiles, self.alive_b_missiles
        if side == 'r':
            alive_own_missiles = alive_r_missiles
            alive_enm_missiles = alive_b_missiles
        if side == 'b':
            alive_own_missiles = alive_b_missiles
            alive_enm_missiles = alive_r_missiles

        # 留下一个小bug，没有考虑双法是否存活
        ego_alive = not ego.dead
        # 目标存活标志
        target_alive = not enm.dead
        # 目标可见性标志 0 完全不可见 1 可获取角度信息 2 可获取全部信息
        target_observable = 2  # 难保不搞成one-hot的形式
        # 目标相对高度
        delta_alt_obs = enm.alt - ego.alt
        # 目标相对方位角
        L_ = enm.pos_ - ego.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        delta_psi = sub_of_radian(q_beta, ego.psi)
        # 目标相对俯仰角
        delta_theta = sub_of_radian(q_epsilon, ego.theta)
        # 目标相对距离
        dist = norm(L_)
        dist_obs = dist
        # 夹角
        v_ = ego.vel_
        vh_ = ego.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = ego.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        ATA = np.arccos(np.dot(L_, ego.point_) / (dist * norm(ego.point_) + 0.001))  # 防止计算误差导致分子>分母
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
        ammo = ego.ammo

        # 雷达可跟踪标志
        if ATA <= ego.max_radar_angle_rad and dist <= ego.max_radar_range and target_alive and ego_alive: # 留下一个小bug，没有考虑双法是否存活
            target_locked = 1
            ego.lock_on = 1
        else:
            target_locked = 0
            ego.lock_on = 0

        # 导弹中制导状态 bool 与 导弹发射间隔时间
        missile_in_mid_term = 0
        missile_t_go = 120
        missile_time_since_shoot = 120
        if not alive_own_missiles:  # len(alive_own_missiles) == 0
            pass
        else:
            time_since_shoots = np.ones(len(alive_own_missiles)) * 120
            missile_t_go = np.ones(len(alive_own_missiles)) * 120
            for i, missile in enumerate(alive_own_missiles):
                time_since_shoots[i] = missile.t
                missile_t_go[i] = missile.t_go
                if missile.guidance_stage < 3:
                    missile_in_mid_term = 1
            missile_time_since_shoot = min(time_since_shoots)
            missile_t_go = min(missile_t_go) # 预备增加观测空间

        # 首先找到所有存活的友方导弹是否由本机发射
        # 然后判断该导弹的 .guidance_stage是否<3

        # 目标雷达跟踪标志 bool
        alpha_enm = np.arccos(np.dot(-L_, enm.vel_) / (norm(enm.vel_) * dist + 0.01))  # 防止计算误差导致分子>分母
        if alpha_enm < ego.max_radar_angle_rad and dist < ego.max_radar_range and target_alive and ego_alive: # 留下一个小bug，没有考虑双法是否存活
            locked_by_target = 1
        else:
            locked_by_target = 0

        # 告警信息
        # 默认值
        warning = 0
        threat_delta_psi = pi  # pi 0
        threat_delta_theta = 0
        threat_distance = self.RWR_distance
        direct_threat = 0 # 是否受到导弹的直接威胁
        if alive_enm_missiles:
            # 存在敌导弹
            dist_closest = 200e3
            for i, missile in enumerate(alive_enm_missiles):
                distance_this_one = missile.distance
                # # Critic能看到中段导弹
                # if distance_this_one < dist_closest:
                #     dist_closest = distance_this_one
                #     threat_delta_psi = sub_of_radian(pi + missile.q_beta, ego.psi)
                #     threat_delta_theta = -missile.q_epsilon
                #     threat_distance = distance_this_one
                # 告警距离大于导弹锁定距离，只要导弹雷达开机，就给告警信息
                if missile.in_angle and missile.radar_on and distance_this_one < self.RWR_distance:
                    warning = 1
                    direct_threat = 1
                    # Critic看不到中段导弹
                    if distance_this_one < dist_closest:
                        dist_closest = distance_this_one # 这个导弹目前最近
                        threat_delta_psi = sub_of_radian(pi + missile.q_beta, ego.psi)
                        threat_delta_theta = -missile.q_epsilon
                    # 如果处于可测距范围(当作和告警距离一样远)，就报告威胁距离
                    threat_distance = min(threat_distance, missile.distance)
                else:
                    pass
                    # if locked_by_target:  # 导弹未进入告警距离但我机仍被敌机锁定
                    #     # 进入告警距离前用敌机方位作为导弹告警方位
                    #     warning = 1 # 敌机为导弹提供中制导也会触发我机告警信号
                    #     # 如果没有受到导弹的直接锁定，才报告敌机的方位
                    #     if direct_threat == 0:
                    #         threat_delta_psi = delta_psi
                    #         threat_delta_theta = delta_theta + ego.theta


        p = ego.p
        q = ego.q
        r = ego.r


        theta_v = ego.theta_v
        psi_v = ego.psi_v
        delta_psi_v = sub_of_radian(ego.target_heading, psi_v)  # 水平速度分量和目标航向之间的差角(弧度)

        alpha_air = ego.alpha_air # 弧度制
        beta_air = ego.beta_air # 弧度制

        speed_T = enm.speed

        # 目标相对方位角速度 (rad/s) / 0.35 与 目标相对俯仰角速度 (rad/s) / 0.35
        vT_ = enm.vel_

        psi_vT = atan2(vT_[2], vT_[0])
        theta_vT = atan2(vT_[1], sqrt(vT_[0] ** 2 + vT_[2] ** 2))

        # 目标水平/垂直进入角
        AA_hor = sub_of_radian(psi_vT, q_beta)  # 向右飞为正
        AA_vert = sub_of_radian(theta_vT, q_epsilon)  # 向上飞为正

        # 边界信息
        if self.vertices is not None:
            d_hor, left_or_right = self.polygon_fences.calc_intern_dist(ego.pos_, ego.psi)
        else:
            d_hor, left_or_right = calc_intern_dist2circle(self.R_cage, ego.pos_, ego.psi)

        # 脱离锁定的时间（30s）为最长
        if locked_by_target:
            ego.last_locked_by_target_time = self.t
            out_locked_time = 0.0
        else:
            out_locked_time = 0.0 # self.t - ego.last_locked_by_target_time

        
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
            "out_locked", # 1
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
                float(ego.Ny)     # 7
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
                float(left_or_right),  # 1
            ]),

            # 脱离锁定时间
            "out_locked": float(out_locked_time),
        }

        ego.current_state = one_side_states
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
        s["ego_control"][7] /= 5 # 取一个不上不下的过载量
        s["weapon"] /= 120
        s["threat"][3] /= 10e3
        s["border"][0] = min(1, s["border"][0] / 50e3)
        s["out_locked"] = min(1, s["out_locked"] / 30)
        # 全程都要看到边界的相对方位 # s["border"][1] = 0 if s["border"][0] == 1 else s["border"][1]
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
            s["ego_control"][7] = s["ego_control"][7] * 5

        if "weapon" in s and s["weapon"] is not None:
            s["weapon"] = s["weapon"] * 120

        if "threat" in s and s["threat"] is not None:
            s["threat"][3] = s["threat"][3] * 10e3

        if "border" in s and s["border"] is not None:
            s["border"][0] = s["border"][0] * 50e3
        
        if "out_locked" in s and s["out_locked"] is not None:
            s["out_locked"] = s["out_locked"] * 30
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

        # [修改] 若本帧无告警（warning==0），将导弹方位/距离信息覆盖回初始值，供actor观测
        # critic通过reward_fn=1绕过此处，可看到真实最近导弹信息
        # actor看不到中段导弹
        if (not reward_fn) and state["warning"] == 0:
            state["threat"][0] = cos(pi)   # cos(threat_delta_psi默认值pi)
            state["threat"][1] = sin(pi)   # sin(threat_delta_psi默认值pi)
            state["threat"][2] = 0.0        # threat_delta_theta默认值0
            state["threat"][3] = self.RWR_distance  # threat_distance默认值

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
        self.state_init["target_information"] = np.array([1, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0])
        self.state_init["weapon"] = 120
        self.state_init["threat"] = np.array([1, 0, 0, self.RWR_distance])
        self.state_init["border"] = np.array([50e3, 0])
        self.state_init["out_locked"] = 0.0

        if pomdp:  # 只有在部分观测情况下需要添加屏蔽
            # 1. 获取记忆 (Rolling Memory)
            if uav.state_memory is None:
                memory = copy.deepcopy(self.state_init)
            else:
                memory = uav.state_memory

            ATA = state["target_information"][4]
            dist = state["target_information"][3]

            # 【新增：提取之前的绝对姿态】
            if "inertial_target_psi" in memory:
                mem_inertial_psi = memory["inertial_target_psi"]
                mem_inertial_theta = memory["inertial_target_theta"]
            else:
                mem_inertial_psi = sub_of_radian(uav.psi + atan2(memory["target_information"][1], memory["target_information"][0]), 0)
                mem_inertial_theta = uav.theta + memory["target_information"][2]

            # 计算在全盲情况下应该显示的基于惯性系更新后的相对角度
            blind_delta_psi = sub_of_radian(mem_inertial_psi - uav.psi, 0)
            blind_delta_theta = mem_inertial_theta - uav.theta

            # 根据条件决定是 "全覆盖" 还是 "部分覆盖"
            
            # 情况A: 超出探测距离 -> 完全不可见
            if dist > uav.max_radar_range:
                state["target_observable"] = 0
                # 整体覆盖：除更新航向补偿外其他信息都用旧的
                state["target_information"] = memory["target_information"].copy()
                state["target_information"][0] = cos(blind_delta_psi)
                state["target_information"][1] = sin(blind_delta_psi)
                state["target_information"][2] = blind_delta_theta
            
            # 情况B: 距离较近
            elif dist > 10e3:
                # B1: 角度大 且 未被锁定 -> 完全不可见
                if ATA > self.RUAV.max_radar_angle_rad and state["locked_by_target"] == 0:
                    state["target_observable"] = 0
                    # 整体覆盖
                    state["target_information"] = memory["target_information"].copy()
                    state["target_information"][0] = cos(blind_delta_psi)
                    state["target_information"][1] = sin(blind_delta_psi)
                    state["target_information"][2] = blind_delta_theta
                
                # B2: 角度大 但 被锁定 (RWR告警) -> 部分可见
                elif ATA > self.RUAV.max_radar_angle_rad and state["locked_by_target"] == 1:
                    state["target_observable"] = 1
                    # 【核心逻辑】只覆盖运动学信息 (dist, speed, AA)，保留当前真实的 RWR 信息 (角度, ATA)
                    # 因为 memory['dist'] 已经是上一步复制下来的旧值，所以这里再次复制依然是旧值
                    for idx in (3, 5):
                        state["target_information"][idx] = memory["target_information"][idx]
                        state["target_information"][6] = pi # 被锁定了，必须假设这时候目标笔直对着我
                        state["target_information"][7] = 0
                
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
            
            # 计算绝对方位角并存入记忆，确保断锁等不可见情况下能保持惯性系正确性
            current_delta_psi = atan2(state["target_information"][1], state["target_information"][0])
            current_delta_theta = state["target_information"][2]
            uav.state_memory["inertial_target_psi"] = sub_of_radian(uav.psi + current_delta_psi, 0)
            uav.state_memory["inertial_target_theta"] = uav.theta + current_delta_theta

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

    def out_cage(self, UAV):
        if self.vertices is not None:
            p = Point(np.array([UAV.pos_[0], UAV.pos_[2]]))
            out = not self.polygon_fences.poly.contains(p)
        else:
            position = UAV.pos_
            pos_h = np.array([position[0], position[2]])
            R_uav = norm(pos_h - self.horizontal_center)
            out = True
            if R_uav <= self.R_cage:
                out = False
        
        # 试验举措：保持对对手的锁定，就不算出界
        if UAV.lock_on:
            out = False

        # # 试验举措：敌机死后边界消失
        # # 敌机全都死了之后可以出界
        # # （警告，这样可能需要同步更改border观测项和奖励）
        # ego_side = UAV.side
        # enm_side = 'b' if ego_side == 'r' else 'r'
        # # 判断敌机阵营是否全灭
        # enm_uav = self.RUAV if enm_side =='r' else self.BUAV
        # if enm_uav.dead:
        #     self.R_cage = np.inf
        #     return False # 不出界
        return out


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
                    # 雷达和锁定 Beam 显示
                    if getattr(UAV, 'lock_on', 0) == 0:
                        # 正常探测雷达
                        data_to_send += (
                            f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                            f"0|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                            f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange=100000, RadarHorizontalBeamwidth=120, RadarVerticalBeamwidth=20\n"
                        )
                    else:
                        # 锁定时的细光束波束
                        target = self.BUAV if UAV.side == 'r' else self.RUAV
                        delta_pos = target.pos_ - UAV.pos_
                        dist = norm(delta_pos)
                        q_beta = atan2(delta_pos[2], delta_pos[0])
                        q_epsilon = atan2(delta_pos[1], sqrt(delta_pos[0]**2 + delta_pos[2]**2))
                        
                        data_to_send += (
                            f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                            f"0|{q_epsilon * 180 / pi:.6f}|{q_beta * 180 / pi:.6f},"
                            f"Type=Beam, Color={color},Visible=0.7,Radius=0.0,RadarMode=1,RadarRange={dist:.1f}, RadarHorizontalBeamwidth=3, RadarVerticalBeamwidth=3\n"
                        )
                    



                    # # 绘制目标期望点 (Carrot)
                    # target_heading = getattr(UAV, 'target_heading', UAV.psi)
                    # # set_height 在无人机 move 时会存在，若无则使用当前高度
                    # set_height_target = getattr(UAV, 'set_height', loc_LLH[2])
                    # # print("target_heading of UAV", UAV.id, "is", target_heading*180/pi)
                    # # print("----")
                    # # 将期望高度控制偏差映射为类似训练代码中的预期俯仰角
                    # delta_h_clipped = np.clip(set_height_target - UAV.alt, -5000, 5000)
                    # theta_req = delta_h_clipped / 5000 * (pi / 2)
                    # N, U, E = LLH2NUE(loc_LLH[0], loc_LLH[1], loc_LLH[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
                    # delta_N = 5e3 * cos(theta_req) * cos(target_heading)
                    # delta_U = 5e3 * sin(theta_req)
                    # delta_E = 5e3 * cos(theta_req) * sin(target_heading)
                    # lon_T, lat_T, _ = NUE2LLH(N + delta_N, U + delta_U, E + delta_E, lon_o=o00[0], lat_o=o00[1], h_o=0)
                    # data_to_send += (
                    #     f"#{send_t:.2f}\n"
                    #     f"{UAV.id + 1000},T={lon_T:.6f}|{lat_T:.6f}|{set_height_target:.6f},"
                    #     f"Name=Carrot,Color={color}\n"
                    # )

                else:
                    data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                    # # data_to_send += f"#-{UAV.id+1000}\n"
                    # data_to_send += f"#{send_t:.2f}\n-{UAV.id + 1000}\n"

            # 传输导弹信息
            for missile in self.missiles:
                if hasattr(missile, 'dead') and missile.dead:
                    data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
                    # 同步移除已经死掉导弹的雷达波束
                    data_to_send += f"#{send_t:.2f}\n-{missile.id+1000}\n"
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
                                    f"Name={missile.WeaponType},Color={color}\n"
                                    )
                    # 导弹雷达波束显示
                    if getattr(missile, 'radar_on', 0):
                        if getattr(missile, 'lock_on', 0):
                            target = self.BUAV if missile.side == 'r' else self.RUAV
                            delta_pos = target.pos_ - missile.pos_
                            dist = norm(delta_pos)
                            q_beta = atan2(delta_pos[2], delta_pos[0])
                            q_epsilon = atan2(delta_pos[1], sqrt(delta_pos[0]**2 + delta_pos[2]**2))
                            
                            data_to_send += (
                                f"{missile.id+1000},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                                f"0|{q_epsilon * 180 / pi:.6f}|{q_beta * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.7,Radius=0.0,RadarMode=1,RadarRange={dist:.1f}, RadarHorizontalBeamwidth=5, RadarVerticalBeamwidth=5\n"
                            )
                        else:
                            data_to_send += (
                                f"{missile.id+1000},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                                f"0|{missile.theta * 180 / pi:.6f}|{missile.psi * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange={missile.detect_range:.1f}, \
                                    RadarHorizontalBeamwidth={np.degrees(missile.sight_angle_max)}, RadarVerticalBeamwidth={np.degrees(missile.sight_angle_max)}\n"
                            )
                    else:
                        # 导弹存活但雷达关闭，移除可能残留的雷达波束可视
                        data_to_send += f"#{send_t:.2f}\n-{missile.id+1000}\n"
                    # # 没电了没雷达
                    # if missile.t > missile.t_max:
                    #     data_to_send += f"#{send_t:.2f}\n-{missile.id+1000}\n"

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            for UAV in self.UAVs:
                data_to_send += f"#{send_t:.2f}\n-{UAV.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{UAV.id+1000}\n"
            for missile in self.missiles:
                data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{missile.id+1000}\n"
            self.tacview.send_data_to_client(data_to_send)
        else:
            pass

    def end_render(self,):
        if self.tacview_show:
            self.tacview.end_render()
        else:
            pass

    def visualize_cage(self, ):
        if self.vertices is not None:
            # 航路点画法（多边形边界）
            cage = np.zeros((len(self.vertices), 3))
            cage_dot_id = 10000
            data_to_send = ''
            for i, v in enumerate(self.vertices):
                cage[i][:] = NUE2LLH(v[0], 300, v[1], lon_o=o00[0], lat_o=o00[1], h_o=0)
            
            for i in range(len(self.vertices)):
                data_to_send += (
                    f"{cage_dot_id},Type=Navaid+Static+Waypoint,"
                    f"T={cage[i][0]:.6f}|{cage[i][1]:.6f}|{cage[i][2]:.6f},Name=WP{i+1},Color=Red,"
                    f"Next={cage_dot_id+1 if i < len(self.vertices)-1 else 10000}\n"
                )
                cage_dot_id += 1
                
        else:
            # 雷达画法（圆形边界）
            data_to_send = (
                # 外圈（纯白）
                f"10000,T={o00[0]}|{o00[1]}|{300}"
                f",Type=Beam,ShortName=Cage,Color=#FFFFFF,Visible=1,Radius=0.0,RadarMode=1"
                f",RadarRange={self.R_cage},RadarHorizontalBeamwidth=360,RadarVerticalBeamwidth=0\n"
            # # 内圈（浅灰，颜色更深/更暗）
            # f"10001,T={o00[0]}|{o00[1]}|{300}"
            # f",Type=Beam,ShortName=Cage,Color=#AAAAAA,Visible=1,Radius=0.0,RadarMode=1"
            # f",RadarRange={self.half_R_cage},RadarHorizontalBeamwidth=360,RadarVerticalBeamwidth=0\n"
            )

        self.tacview.send_data_to_client(data_to_send)
        print('cage set')

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

    def has_ammo_to_fire(self, side='r'):
        if side == 'r':
            ego = self.RUAV
        else:  # side == 'b'
            ego = self.BUAV
        if ego.ammo>0 and not ego.dead:
            return 1
        else:
            return 0

def launch_missile_immediately(env, side='r', tabu=0, action_label=None):
    """
    立即发射导弹 (受俯仰角限制)
    """
    new_missile_id = None
    if side == 'r':
        uav = env.RUAV
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        target = env.RUAV

    if action_label is not None and hasattr(env, 'maneuver14LR'):
        action_array = env.maneuver14LR(uav, action_label)
        delta_target_height = action_array[0]

        desired_theta = (min(delta_target_height, env.max_alt_safe-uav.alt) / 5000.0) * (pi / 2) # 不能再爬升了，就得降低期望俯仰角
        if (desired_theta - uav.theta > (15 * pi / 180)) and uav.alt < 7000: # 7000m以上很难再维持大爬升角，高抛延迟开火仅对7000m以下生效
            return None

    ego_state = env.get_state(uav.side)
    ATA = ego_state["target_information"][4]
    distance = ego_state["target_information"][3]
    AA_hor = ego_state["target_information"][6]
    target_locked = ego_state["target_locked"]

    # 发射导弹
    if uav.ammo>0 and not uav.dead:
        if (not tabu) or\
                target_locked and ego_state["weapon"]>=0.1 and ATA<=env.RUAV.max_radar_angle_rad:
            new_missile = uav.launch_missile(target, env.t, missile_class)
            uav.ammo -= 1
            uav.fired += 1

            # 记录导弹发射瞬间的 ATA、distance 和 AA_hor
            uav.launch_states_order = ['ATA', 'distance', 'AA_hor', 'target_locked', 't_go']
            uav.launch_states.append(np.array([ATA, distance, AA_hor, target_locked, ego_state["weapon"]]))

            # 记录发射时刻，用于开火间隔计算
            if not hasattr(uav, 'launch_times'):
                uav.launch_times = []
            uav.launch_times.append(env.t)
            # 在导弹里面记录发射时间
            new_missile.shoot_time = env.t
            new_missile.reward_function_info["time"] = env.t

            new_missile.side = 'r' if side == 'r' else 'b'
            new_missile_id = new_missile.id
            if side == 'r':
                env.Rmissiles.append(new_missile)
            else:
                env.Bmissiles.append(new_missile)
            env.missiles = env.Rmissiles + env.Bmissiles

    if hasattr(uav, 'about_to_fire'):
        uav.about_to_fire = 0

    return new_missile_id

