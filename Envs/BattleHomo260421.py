'''出生点改在外面指定'''

'''多对多需要真正启用cat动作空间的多维度功能。情况涉及到雷达开关'''

import numpy as np
from random import random
import random
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
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Math_calculates.Calc_dist2border import calc_intern_dist2circle, polygon_fences
from Visualize.tacview_visualize2 import *
from shapely.geometry import Point
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Utilities.LocateDirAndAgents import *
from Controller.F16PIDController2_1 import *
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

R_cage = 100e3

# min_height = 0
# max_height = 15e3

R_birth = 40e3

horizontal_center = np.array([0, 0])


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Battle(object):
    def __init__(self, args, tacview_show=0, vertices=None):
        self.vertices = vertices
        if self.vertices is not None:
            self.polygon_fences = polygon_fences(self.vertices)
        self.horizontal_center = horizontal_center
        # 加载PID控制器
        self.PIDController = F16PIDController()
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
        self.dt_maneuver = dt_maneuver
        self.dt_move = dt_move
        self.UAV_ids = None
        self.UAV_shot = None
        self.args = args
        self.RUAVsTable = None
        self.BUAVsTable = None
        self.UAVsTable = None
        self.t = None
        self.game_time_limit = self.args.max_episode_len  # None
        self.running = None
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
        self.RWR_distance = 60e3 # 最大告警距离
        self.RWR_ranging_distance = self.RWR_distance # 最大告警可测距


        self.RED_BIRTH_STATE = {'position': np.array([-R_birth * cos(0), 8000.0, -R_birth * sin(0)]),
                                        'psi': 0,
                                        'e2e': False
                                        }
        self.BLUE_BIRTH_STATE = {'position': np.array([-R_birth * cos(pi), 8000.0, -R_birth * sin(pi)]),
                                         'psi': pi,
                                         'e2e': False
                                         }
        # 默认观测填充模板
        self.state_init_mod = self._generate_state_init_mod()
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
            self.visualize_cage()           
    
    # 动态属性，把函数伪装成属性，self.RUAVs会调用函数返回dict的value列表
    # 警告：如果同名的属性被注册，property会失效
    @property
    def RUAVs(self):
        return list(self.RUAVsTable.values())

    @property
    def BUAVs(self):
        return list(self.BUAVsTable.values())

    @property
    def UAVs(self):
        # Python 3.9+ 支持字典合并: list({**self.RUAVsTable, **self.BUAVsTable}.values())
        # self.RUAVs + self.BUAVs
        return list(self.UAVsTable.values())

    @property
    def Rmissiles(self):
        return list(self.RMissileTable.values())

    @property
    def Bmissiles(self):
        return list(self.BMissileTable.values())

    @property
    def missiles(self):
        return self.Rmissiles + self.Bmissiles


    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, seed=None, options=None, ego_side='b'):  # 重置位置和状态
        self.R_cage = self.R_cage0
        self.horizontal_center = horizontal_center

        self.ego_side = ego_side
        # debug
        
        # [新增] 如果需要支持随机种子控制，可以在这里设置
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        if red_birth_state is None:
            red_birth_state = self.RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.BLUE_BIRTH_STATE

        # # 随机对调红蓝双方的出生位置和朝向，消除物理不对称性带来的长期偏置
        # if np.random.rand() > 0.5:
        #     red_birth_state, blue_birth_state = blue_birth_state, red_birth_state

        self.BLUE_BIRTH_STATE = blue_birth_state
        self.RED_BIRTH_STATE = red_birth_state
        self.dt_maneuver = dt_maneuver  # simulation interval，1 second
        self.t = 0
        # 初始化无人机
        self.Rnum = 2 # test
        self.Bnum = 2
        self.RUAVsTable = {}
        self.BUAVsTable = {}
        self.UAVsTable = {}
        self.alive_r_missiles = []
        self.alive_b_missiles = []
        self.alive_missiles = []
        self.win = 0
        self.lose = 0
        self.draw = 0
        # 无人机统一初始化
        for i in range(self.Rnum + self.Bnum):
            is_red = i < self.Rnum
            index_in_side = i if is_red else i - self.Rnum
            
            UAV = UAVModel(dt=self.dt_move)
            UAV.state_memory = None
            UAV.action_memory = np.array([0, 0, 340])
            UAV.last_state = None
            UAV.current_state = None
            UAV.launch_states = []
            UAV.launch_states_order = None
            UAV.init_ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.id = (index_in_side + 1) if is_red else (index_in_side + 201)
            UAV.red = is_red
            UAV.blue = not is_red
            UAV.side = 'r' if is_red else 'b'
            UAV.color = np.array([1, 0, 0]) if is_red else np.array([0, 0, 1])
            
            birth_state = red_birth_state if is_red else blue_birth_state
            
            # 出生点 (注意：使用 copy 防止修改外部传入的字典数据)
            # 构造一个0, 1, -1, 2，-2 的函数
            # temp = (-1)**index_in_side * int((index_in_side+1)/2)
            # 1/2, -1/2, 3/2, -3/2, 5/2, -5/2...
            temp = (-1)**index_in_side * (np.floor(index_in_side/2) + 1) / 2
            UAV.pos_ = birth_state['position'].copy() + np.array([1, 0, 0]) * temp * 20e3 # 出生点间距10e3，南北排开
            
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
            # # 不能出生在外面
            # init_R = norm([UAV.pos_[0], UAV.pos_[2]])
            # safe_R_cage = self.R_cage - 5e3
            # if init_R > safe_R_cage:
            #     scale_factor = safe_R_cage / init_R
            #     UAV.pos_[0] *= scale_factor
            #     UAV.pos_[2] *= scale_factor
                
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
            UAV.radar_on = 1
            UAV.radar_center = np.array([0, 0]) # 雷达中心相对方位的归一化值，先水平后垂直
            UAV.escape_once = 0
            
            if is_red:
                self.RUAVsTable[UAV.id] = UAV
            else:
                self.BUAVsTable[UAV.id] = UAV
                
            UAV.lock_on = 0
        self.running = True
        self.UAVsTable = {**self.RUAVsTable, **self.BUAVsTable}
        self.UAV_ids = [UAV.id for UAV in self.UAVs]
        self.UAV_shot = [False for _ in range(len(self.UAVs))]
        self.r_missile_count = 0  # 红方累计发射导弹数
        self.b_missile_count = 0  # 蓝方累计发射导弹数
        self.RUAV_ids = [UAV.id for UAV in self.RUAVs]
        self.BUAV_ids = [UAV.id for UAV in self.BUAVs]
        self.RMissileTable = {}
        self.BMissileTable = {}
        self.MissileTable = {}
        
    def update_missile_state(self):
        alive_r_missiles = [m for m in self.Rmissiles if not m.dead]
        alive_b_missiles = [m for m in self.Bmissiles if not m.dead]
        self.alive_r_missiles = alive_r_missiles
        self.alive_b_missiles = alive_b_missiles
        self.alive_missiles = alive_r_missiles + alive_b_missiles

    def update_uav_state(self):
        alive_r_uavs = [u for u in self.RUAVs if not u.dead]
        alive_b_uavs = [u for u in self.BUAVs if not u.dead]
        self.alive_r_uavs = alive_r_uavs
        self.alive_b_uavs = alive_b_uavs
        self.alive_uavs = alive_r_uavs + alive_b_uavs
        
    # 根据id查找实体
    def find_uav_with_id(self, id):
        uav = self.UAVsTable[id]
        return uav
    
    def find_missile_with_id(self, id):
        missile = self.MissileTable[id]
        return missile

    def index2id(self, ego_id, target_index, relation='enm'):
        """
        将神经网络视角的 index 转换为环境中的真实实体 id
        """
        ego = self.find_uav_with_id(ego_id)
        
        if relation == 'enm':
            # 查敌机：直接取对侧的 id 列表
            enm_ids = self.BUAV_ids if ego.side == 'r' else self.RUAV_ids
            if target_index < len(enm_ids):
                return enm_ids[target_index]
        elif relation == 'ally':
            # 查队友：取同侧的 id 列表，并剔除自己
            ally_ids = self.RUAV_ids if ego.side == 'r' else self.BUAV_ids
            ally_ids_without_ego = [uid for uid in ally_ids if uid != ego_id]
            if target_index < len(ally_ids_without_ego):
                return ally_ids_without_ego[target_index]
        return None

    def id2index(self, ego_id, target_id, relation='enm'):
        """
        将环境中的真实实体 id 转换为神经网络视角的 index
        """
        ego = self.find_uav_with_id(ego_id)
        if relation == 'enm':
            enm_ids = self.BUAV_ids if ego.side == 'r' else self.RUAV_ids
            if target_id in enm_ids:
                return enm_ids.index(target_id)
        elif relation == 'ally':
            ally_ids = self.RUAV_ids if ego.side == 'r' else self.BUAV_ids
            ally_ids_without_ego = [uid for uid in ally_ids if uid != ego_id]
            if target_id in ally_ids_without_ego:
                return ally_ids_without_ego.index(target_id)
        return None


    def close(self):
        """
        关闭环境并清理资源
        """
        if self.tacview_show:
            try:
                self.tacview.socket.close()
            except:
                pass
        

    def launch_1_missile(self, source_id, target_id):
        """
        source_id: 发射导弹的无人机 id
        target_id: 被击中的无人机 id
        """
        uav = self.find_uav_with_id(source_id)
        target = self.find_uav_with_id(target_id)
        side = uav.side

        # 发射导弹
        if uav.ammo>0 and not uav.dead:
            new_missile = uav.launch_missile(target, self.t, missile_class)
            uav.ammo -= 1
            new_missile.side = side
            new_missile.guider_id = source_id
            if side == 'r':
                self.Rmissiles.append(new_missile)
                self.r_missile_count += 1
                new_missile.id = 101 + self.r_missile_count
                self.RMissileTable[new_missile.id] = new_missile
            else:
                self.Bmissiles.append(new_missile)
                self.b_missile_count += 1
                new_missile.id = 301 + self.b_missile_count
                self.BMissileTable[new_missile.id] = new_missile
            
            
            self.MissileTable = {**self.RMissileTable, **self.BMissileTable}

            return new_missile
        return None
            

    # todo 输入动作打包为大小不确定的量
    def step(self, flight_cmds):
        "flight_cmds 为一个list, 用id顺序排列所有飞机"
        # "actions_of_ids: {id: action}"
        report_move_time_rate = int(round(self.dt_maneuver / self.dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        # flight_cmds = [actions_of_ids[id] for id in self.UAV_ids]

        # 在整个 maneuver step 开始时只重置一次 escape_once（不要在内层子步或导弹循环中再次重置）
        for UAV in self.UAVs:
            UAV.escape_once = 0

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, flight_cmds):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                # print(action)
                target_delta_height = action[0]  # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                target_delta_heading = action[1]  # 相对方位(弧度)
                target_speed = action[2]  # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
                if len(action)==4:
                    rudder = action[3]
                else:
                    rudder = None
                # print('target_delta_height',target_delta_height)
                
                # 多对多彻底放弃端到端控制
                e2e = False

                # 防撞地系统
                if self.shielded:
                    # 临近撞地强制拉起
                    if UAV.alt < self.min_alt_safe + 1e3:
                        target_delta_height = max(self.min_alt_safe + 1e3 - UAV.alt, target_delta_height)
                        e2e = False
                        target_delta_heading = np.clip(target_delta_heading, -pi/3, pi/3)

                    # 不许超过限高
                    if UAV.alt > self.max_alt_safe:
                        target_delta_height = min(self.max_alt_safe - UAV.alt, target_delta_height)
                        e2e = False

                    # 速度过低强制加油门
                    if UAV.speed/340 < 0.5:
                        if e2e:
                            UAV.target_speed = 1
                        else:
                            UAV.target_speed = max(340, target_speed)
                
                # 不准出界
                if self.no_out:
                    if self.vertices is not None:
                        d_hor, left_or_right = self.polygon_fences.calc_intern_dist(UAV.pos_, UAV.psi_v)
                    else:
                        d_hor, left_or_right = calc_intern_dist2circle(self.R_cage, UAV.pos_, UAV.psi_v)
                    if d_hor < 8e3: # 8e3
                        if left_or_right > 0:
                            target_delta_heading = min(-pi/2, target_delta_heading)
                        else:
                            target_delta_heading = max(pi/2, target_delta_heading)
                
                # d_hor, left_or_right = calc_intern_dist2circle(self.R_cage, UAV.pos_, UAV.psi_v)
                # # 不准出界
                # if self.no_out:
                #     if d_hor < 8e3:
                #         if left_or_right == 1:
                #             target_delta_heading = min(-pi/2, target_delta_heading)
                #         if left_or_right == -1:
                #             target_delta_heading = max(pi/2, target_delta_heading)

                # 将调整后的动作写回，Python引用就地修改
                action[0] = target_delta_height
                action[1] = target_delta_heading
                action[2] = target_speed

                UAV.target_heading = sub_of_radian(UAV.psi + action[1], 0)
                # # 调试信息：计算相对方位角
                # other_uav = self.UAVs[1] if UAV == self.UAVs[0] else self.UAVs[0]
                # angle_to_other_rad = atan2(other_uav.pos_[2] - UAV.pos_[2], other_uav.pos_[0] - UAV.pos_[0])
                # print(f"UAV {UAV.id}({UAV.side}) target_heading: {np.degrees(UAV.target_heading):.2f} deg, angle_to_other: {np.degrees(angle_to_other_rad):.2f} deg")
                # print("--")

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中


        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, flight_cmds):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                target_delta_height = action[0]  # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                target_delta_heading = action[1]
                target_speed = action[2]  # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值

                # # 动作平滑，用一阶惯性环节
                # target_height_changing_limit = 5000/(pi/2) * pi / 180 * 60 if UAV.alt>self.min_alt_safe else np.inf # 每秒限制改变俯仰角度数
                # target_psi_changing_limit = pi / 180 * 90 # 每秒限制航向误差改变度数
                # target_speed_changing_limit = np.inf
                # action[0] = np.clip(action[0], 
                #     UAV.action_memory[0]-target_height_changing_limit*self.dt_move,
                #     UAV.action_memory[0]+target_height_changing_limit*self.dt_move,
                #     )
                # action[1] = np.clip(action[1], 
                #     UAV.action_memory[1]-target_psi_changing_limit*self.dt_move,
                #     UAV.action_memory[1]+target_psi_changing_limit*self.dt_move,
                #     )
                # action[2] = np.clip(action[2], 
                #     UAV.action_memory[2]-target_speed_changing_limit*self.dt_move,
                #     UAV.action_memory[2]+target_speed_changing_limit*self.dt_move,
                #     )
                # UAV.action_memory = np.array([
                #     action[0],
                #     action[1],
                #     action[2],
                # ])

                # 计算当前子步下，实际机头指向与固定目标点之间的动态差角
                dynamic_delta_psi = sub_of_radian(UAV.target_heading, UAV.psi)

                rudder = None

                # 出界就炸
                if self.out_cage(UAV):
                    UAV.dead = 1
                    continue
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
                            np.clip(target_delta_height, -5000, 5000),   # 与训练环境 clip 对齐
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
                    UAV.move(elevator, aileron, throttle, relevant_height=True, e2e=True, rudder=rudder, dt=self.dt_move)
                else:
                    UAV.move(target_delta_height, target_delta_heading, target_speed, relevant_height=True, e2e=0, rudder=rudder)

            # 导弹移动
            self.update_missile_state() # 先把存活的导弹找出来
            for missile in self.alive_missiles[:]:  # 使用切片创建副本以允许删除
                target = self.find_uav_with_id(missile.target_id)
                # todo 干扰或更换目标逻辑，写在这里，箔条设置为从属于UAV的属性 
                # track_target = f(target, missile, self.UAVs, self.chaffs)

                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹
                    missile.dead = True # todo 多对多环境中改成missile.target = None, 并在missile类里改成丢失目标飞直线，并且无法触发hit
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
                # 导弹慢速自爆，节省计算量
                if vmt1 < missile.speed_min \
                    and missile.t > 0.5 + missile.stage1_time + missile.stage2_time \
                        and last_pmt_[1] < 15e3: # 3000
                    missile.dead = True
                if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                    missile.dead = True
                if missile.t > missile.t_max:  # 超时自爆
                    missile.dead = True
                if missile.t >= 0 + self.dt_move and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=self.dt_move, kill_range=missile.kill_range)
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
                        target.radar_on = 0
                        self.UAV_shot[self.UAV_ids.index(target.id)] = True

                if missile.dead == True and not target.dead:
                    target.escape_once = 1
                    # 目标逃脱
                # else:
                #     target.escape_once = 0

            # 飞机接收毁伤判别信息
            for i, UAV in enumerate(self.UAVs):
                # 飞机被导弹命中判断
                if self.UAV_shot[i]:
                    UAV.dead = True
                    UAV.got_hit = True
                    UAV.radar_on = 0

                # 出界判别
                if self.crash(UAV):
                    UAV.dead = True
                    UAV.radar_on = 0
                # self.running = False

        # r_reward_n, b_reward_n = self.get_reward()
        # terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or self.crash(UAV):  # or self.out_cage(UAV): ###
                UAV.dead = True
                UAV.radar_on = 0
                # self.running = False

        # 更新UAV信息
        self.update_uav_state()  # 更新飞机存活信息

        return 0, 0, 0, 0, 0  # 废弃不再使用了

    def _generate_state_init_mod(self):
        # 部分观测填充值模板
        state_init_mod = {}
        # 只有敌机、队友、已发射的导弹、来袭导弹需要这套模板
        state_init_mod["ally_info"] = np.array([
            1, # alive
            1, # radar_on
            -1, # cos(dealta_psi)
            0, # sin(delta_psi)
            0, # delta_tehta
            100e3, # dist
            300, # speed
            0, # delta_psi_ally
            0, # theta_ally
        ], dtype='float64')
        state_init_mod["enm_info"] = np.array([
            1, # alive
            1, # radar_on
            2, # observable
            0, # locked
            0, # be locked
            1, # cos(dealta_psi)
            0, # sin(delta_psi)
            0, # delta_tehta
            100e3, # dist
            0, # ATA
            300, # speed
            pi, # AA_hor
            0, # AA_vert
        ], dtype='float64')
        state_init_mod["weapon_info"] = np.array([
            120, # time
            1, # sin
            0, # sin 
            0, # mid_guide
        ], dtype='float64')
        state_init_mod["RWR_info"] = np.array([
            0, # warning
            1, # cos
            0, # sin
            0, # delta_theta
            100e3, # dist
        ], dtype='float64')
        return state_init_mod

    # todo 观测信息要考虑到多对多的情况，不再只是两架飞机
    def get_state(self, id):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        观测3层结构：
        state_of_agent = dict{
            标志位（非向量）信息，
            己方信息向量（固定），
            队友信息list[向量，按id排序，以可见性开头,如果队友死亡，将其置为空],
            己方导弹信息list[向量，按id排序，以可见性开头,如果导弹死亡/不可见，将其置为空],
            敌机信息向量list[向量，按id排序，以可见性开头,如果敌机死亡/不可见，将其置为空],
            敌方导弹信息list[向量，按id排序，以可见性开头,如果敌方导弹死亡/不可见，将其置为空],
        }
        具体某个非自己的向量信息索引方法： state_of_agent["key"][x] 到向量，后再接一个 [y] 索引具体维度

        原始环境提供的状态信息，通过列表给出，供检查用，不可以直接输入进神经网络
        需要使用外挂状态输入结构转换（先填充为最大可处理数量的信息，未存在的导弹也要用填充值填充）
        '''
        # 原先将所有量打包成一个 numpy array，这里改为 dict 结构
        self.key_order = [
            # 我机信息
            "ego_alive",  # 长度为1
            "ego_main",  # 固定长度
            "ego_control",  # 固定长度
            "ego_ammo",  # 剩余弹量
            # 雷达信息
            "Radar_info",  # 雷达开关状态，指向相对方位和俯仰（惯性系下）
            "border", # 边界信息
            # 队友信息
            "ally_infos",  # [存活、雷达开关状态、相对方位、相对俯仰角、相对距离、速度、航向角、俯仰角]*数量
            # 敌机信息
            "enm_infos",  # [存活、雷达开关状态、可见性、是否能锁定、是否锁定我、相对方位、相对俯仰角、相对距离、速度、hot/cold]*数量
            # 导弹信息（不区分发射机，预留接力制导空间）
            "weapon_infos",  # [飞行时间、相对方位、是否可引导]*数量
            # 告警信息
            "RWR_infos",  # [距离, 相对方位、相对俯仰]*数量
        ]

        ego = self.find_uav_with_id(id)
        ego_side = ego.side

        # 先自己、然后队友，最后敌机，队友和敌机内部按照id排序
        # 0. 标识符
        ego_alive = not ego.dead  # 自己活着没

        # 边界信息
        if self.vertices is not None:
            d_hor, left_or_right = self.polygon_fences.calc_intern_dist(ego.pos_, ego.psi)
        else:
            d_hor, left_or_right = calc_intern_dist2circle(self.R_cage, ego.pos_, ego.psi)

        #------------------------
        # 1. 己方信息向量（固定结构）
        #------------------------
        v_ = ego.vel_
        vh_ = ego.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = ego.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        # 速度观测量
        v_ego = v
        # 本机高度
        h_ego = ego.alt
        # 本机俯仰角
        sin_theta = sin(ego.theta)
        cos_theta = cos(ego.theta)
        # 本机滚转角
        sin_phi = sin(ego.phi)
        cos_phi = cos(ego.phi)

        # 剩余导弹量
        ammo = ego.ammo
        
        # 死了关雷达
        if ego.dead: ego.radar_on=0
        
        # 雷达信息
        radar_info = np.array([ego.radar_on, 0, 0]) # 波束中心功能先空置

        ego_main = np.array([
            float(v_ego),  # 0本机速度 m/s
            float(h_ego),  # 1本机高度 m
            float(cos_theta),  # 2
            float(sin_theta),  # 3
            float(cos_phi),  # 4
            float(sin_phi),  # 5
        ])

        ego_control = np.array([
            float(ego.p),  # 0 p rad/s act1_last
            float(ego.q),  # 1 q rad/s act2_last
            float(ego.r),  # 2 r rad/s act3_last
            float(ego.theta_v),  # 3
            float(ego.psi_v),  # 4
            float(ego.alpha_air),  # 5 rad
            float(ego.beta_air),  # 6 rad
            float(ego.Ny),      # 7 overload
        ])

        #------------------------
        # 2. 队友信息向量（所有队友，不管死活，都按id排序，如果队友死亡，照传不误）
        # todo，在get_state之外，提供生成虚拟目标信息token形状所需的参数）
        #------------------------
        ego_side_uavs = self.RUAVs if ego_side =='r' else self.BUAVs
        ego_side_uav_ids = self.RUAV_ids if ego_side =='r' else self.BUAV_ids

        ally_infos = []
        # 所有队友:
        # [存活、相对方位、相对俯仰角、相对距离、速度、航向角、俯仰角]*数量
        ally_uavs = [uav for uav in ego_side_uavs if uav is not ego]
        for ally in ally_uavs:
            # 相对方位角
            L_ = ally.pos_ - ego.pos_
            q_beta = atan2(L_[2], L_[0])
            L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
            L_v = L_[1]
            q_epsilon = atan2(L_v, L_h)
            delta_psi = sub_of_radian(q_beta, ego.psi)
            # 相对俯仰角
            delta_theta = sub_of_radian(q_epsilon, ego.theta)
            # 相对距离
            dist = norm(L_)
            # 相对航向角（队友航向角-我机航向角）
            delta_psi_ally = sub_of_radian(ally.psi, ego.psi)
            # 队友速度
            v_ally = ally.speed
            # 队友俯仰角
            theta_ally = ally.theta
            
            if ally.dead: ally.radar_on = 0

            one_ally_info = np.array([
                float(not ally.dead), # 0
                float(ally.radar_on), # 1
                float(cos(delta_psi)),  # 2
                float(sin(delta_psi)),  # 3
                float(delta_theta),   # 4
                float(dist),          # 5
                float(v_ally),        # 6
                float(delta_psi_ally),# 7
                float(theta_ally),    # 8
                # # 惯性系下的位置关系，队友相对位置信息不需要暂不记忆
                # float(dist), # 9
                # float(q_beta), # 10
                # float(q_epsilon), # 11
            ])

            ally_infos.append(one_ally_info)

        #------------------------
        # 3. 敌机信息:
        # [存活、可见性、是否能锁定、是否锁定我、相对方位、相对俯仰角、相对距离、速度、hot/cold]*数量
        #------------------------
        ego.lock_on = 0
        enm_infos = []
        enm_uavs = self.RUAVs if ego_side =='b' else self.BUAVs
        for enm in enm_uavs:
            # 相对方位角
            L_ = enm.pos_ - ego.pos_
            q_beta = atan2(L_[2], L_[0])
            L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
            L_v = L_[1]
            q_epsilon = atan2(L_v, L_h)
            # 相对方位角
            delta_psi = sub_of_radian(q_beta, ego.psi)
            # 相对俯仰角
            delta_theta = sub_of_radian(q_epsilon, ego.theta)
            # 相对距离
            dist = norm(L_)
            # 夹角
            ATA = np.arccos(np.dot(L_, ego.point_) / (dist * norm(ego.point_) + 0.00001))
            
            # 目标速度
            enm_speed = norm(enm.speed)

            # 水平进入角
            # 目标相对方位角速度 (rad/s) / 0.35 与 目标相对俯仰角速度 (rad/s) / 0.35
            vT_ = enm.vel_

            psi_vT = atan2(vT_[2], vT_[0])
            theta_vT = atan2(vT_[1], sqrt(vT_[0] ** 2 + vT_[2] ** 2))

            # 目标水平/垂直进入角
            AA_hor = sub_of_radian(psi_vT, q_beta)  # 向右飞为正
            AA_vert = sub_of_radian(theta_vT, q_epsilon)  # 向上飞为正

            # 数据是否可信（有队友视野就算完全可观测）
            enm_observable = 2  # 不在这里算

            # 是否锁定该敌机
            if ego.radar_on and \
                ATA < ego.max_radar_angle_rad and dist < ego.max_radar_range:
                enm_locked = 1
                ego.lock_on = 1
            else:
                enm_locked = 0

            # 该敌机是否锁定我
            alpha_enm = np.arccos(np.dot(-L_, enm.vel_) / (norm(enm.vel_) * dist + 0.00001))  # 防止计算误差导致分子>分母
            if alpha_enm < ego.max_radar_angle_rad and dist < ego.max_radar_range and enm.radar_on:
                locked_by_enm = 1
            else:
                locked_by_enm = 0

            if enm.dead: enm.radar_on=0
            enm_infos.append(np.array([
                float(not enm.dead), # 0
                float(enm.radar_on), # 1
                float(enm_observable), # 2
                float(enm_locked), # 3
                float(locked_by_enm), # 4
                float(cos(delta_psi)), # 5
                float(sin(delta_psi)), # 6
                float(delta_theta), # 7
                float(dist), # 8
                float(ATA), # 9
                float(enm_speed), # 10
                float(AA_hor), # 11
                float(AA_vert), # 12
                # 惯性系下的位置关系，进入记忆，而非神经网络get_obs时候需要裁剪掉后面
                float(q_beta), # 13
                float(q_epsilon), # 14
            ]))


        #------------------------
        # 导弹信息向量，协同制导我还没想清楚要如何做，先按不做的来
        #------------------------
        alive_ally_missiles = self.alive_r_missiles if ego_side=='r' else self.alive_b_missiles
        alive_ego_missiles = [m for m in alive_ally_missiles if m.guider_id == ego.id]
        ally_missile_info = []
        # 看不到队友发射的导弹，只看到自己的，协同制导时改变处理方式
        for missile in alive_ego_missiles:
            # 相对方位角
            guide_line_ = missile.pos_ - ego.pos_
            q_beta = atan2(guide_line_[2], guide_line_[0])
            q_epsilon = atan2(guide_line_[1], np.sqrt(guide_line_[0] ** 2 + guide_line_[2] ** 2))
            delta_psi = sub_of_radian(q_beta, ego.psi)
            if missile.guidance_stage < 3:
                missile_in_mid_term = 1
            else:
                missile_in_mid_term = 0
            ally_missile_info.append(np.array([
                float(missile.t),        # 飞行时间
                float(cos(delta_psi)),  # 相对方位 cos
                float(sin(delta_psi)),  # 相对方位 sin
                float(missile_in_mid_term), # 中制导标志
                # # get_obs时候需要裁剪掉后面
                # float(q_beta), # 3
                # float(q_epsilon), # 4
            ]))

        #------------------------
        # 告警信息向量, 不在环境里面做填充，有多少放多少
        #------------------------
        alive_enm_missiles = self.alive_b_missiles if ego_side=='r' else self.alive_r_missiles
        RWR_infos = []
        for enm_missile in alive_enm_missiles:
            # 导弹雷达开启后告警启动，否则不启动
            if enm_missile.radar_on:
                threat_line_ = enm_missile.pos_ - ego.pos_
                threat_distance = norm(threat_line_)
                threat_q_beta = atan2(threat_line_[2], threat_line_[0])
                threat_q_epsilon = atan2(threat_line_[1], np.sqrt(threat_line_[0] ** 2 + threat_line_[2] ** 2))
                threat_delta_psi = sub_of_radian(threat_q_beta, ego.psi)
                threat_theta = threat_q_epsilon
                warning = threat_distance < enm_missile.detect_range  # 假设导弹的探测距离正好就是飞机的告警距离
                RWR_infos.append(np.array([
                    float(warning), # 0
                    float(cos(threat_delta_psi)), # 1
                    float(sin(threat_delta_psi)), # 2
                    float(threat_theta), # 3
                    float(threat_distance), # 4
                    # # get_obs时候需要裁剪掉后面
                    # float(threat_q_beta), # 5
                    # float(threat_q_epsilon), # 6
                ]))
       
        one_state = {
            "ego_alive": ego_alive,
            "ego_main": ego_main,
            "ego_control": ego_control,
            "ego_ammo": ammo,
            "Radar_info": radar_info,
            "border": np.array([
                float(d_hor),  # 0
                int(left_or_right),  # 1
                float(norm([
                    ego.pos_[0]-self.horizontal_center[0], 
                    ego.pos_[2]-self.horizontal_center[1]
                ])),  # 2 距离圈中心距离
            ]),
            "ally_infos": np.array(ally_infos),
            "enm_infos": np.array(enm_infos),
            "weapon_infos": np.array(ally_missile_info),
            "RWR_infos": np.array(RWR_infos),
        }

        ego.current_state = one_state
        return one_state
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"][0] /= 340
            s["ego_main"][1] /= 5e3
        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"][0] /= (2 * pi)
            s["ego_control"][1] /= (2 * pi)
            s["ego_control"][2] /= (2 * pi)
            s["ego_control"][7] /= 5 
        if "border" in s and s["border"] is not None:
            s["border"][0] = min(1, s["border"][0] / 50e3)
            s["border"][2] /= self.R_cage
        if "ally_infos" in s and len(s["ally_infos"]) > 0:
            s["ally_infos"][:, 5] /= 50e3
            s["ally_infos"][:, 6] /= 340
        if "enm_infos" in s and len(s["enm_infos"]) > 0:
            s["enm_infos"][:, 8] /= 50e3
            s["enm_infos"][:, 10] /= 340
            s["enm_infos"] = s["enm_infos"][:, :13]
        if "weapon_infos" in s and len(s["weapon_infos"]) > 0:
            s["weapon_infos"][:, 0] /= 120.0
        if "RWR_infos" in s and len(s["RWR_infos"]) > 0:
            s["RWR_infos"][:, 4] /= 50e3
        return s
    
    def unscale_state(self, obs_input):
        """把 scale_state 的缩放还原。仅判断 key 是否存在（不再检查长度）。"""
        s = copy.deepcopy(obs_input)
        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"][0] = s["ego_main"][0] * 340
            s["ego_main"][1] = s["ego_main"][1] * 5e3
        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"][0] = s["ego_control"][0] * (2 * pi)
            s["ego_control"][1] = s["ego_control"][1] * (2 * pi)
            s["ego_control"][2] = s["ego_control"][2] * (2 * pi)
            s["ego_control"][7] *= 5
        if "border" in s and s["border"] is not None:
            s["border"][0] = s["border"][0] * 50e3
            s["border"][2] *= self.R_cage
        if "ally_infos" in s and len(s["ally_infos"]) > 0:
            s["ally_infos"][:, 5] *= 50e3
            s["ally_infos"][:, 6] *= 340
        if "enm_infos" in s and len(s["enm_infos"]) > 0:
            s["enm_infos"][:, 8] *= 50e3
            s["enm_infos"][:, 10] *= 340
            # s["enm_infos"] = s["enm_infos"][:, :13] # 裁剪惯性系信息
        if "weapon_infos" in s and len(s["weapon_infos"]) > 0:
            s["weapon_infos"][:, 0] *= 120
        if "RWR_infos" in s and len(s["RWR_infos"]) > 0:
            s["RWR_infos"][:, 4] *= 50e3
        return s
    
    def base_obs(self, id, pomdp=0, reward_fn=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        ego = self.find_uav_with_id(id)
        state = self.get_state(id)
        # 增加时间戳，用于状态管理
        state['t'] = self.t

        # [新增] 智能的状态快照更新逻辑：防止同一时间步多次调用导致 last_state 被覆盖
        if ego.current_state is None:
            # 初始化
            ego.last_state = copy.deepcopy(state)
        else:
            # 通过比对时间戳判断是否为新的仿真步
            # 只有当时间推进了，才把旧的 current_state 归档为 last_state
            if abs(self.t - ego.current_state['t']) > 0.5 * self.dt_maneuver:
                ego.last_state = copy.deepcopy(ego.current_state)
        ego.current_state = copy.deepcopy(state)
        # 开局瞬间会给完全信息观测，GRU也改为残差形式，不再需要默认值了
        
        if pomdp and (not reward_fn): 
            # 对每个敌机分别处理可见性问题
            """
            不太高明的做法（但曾经有用）：
            冻结看不到的敌机信息，而不是用空token代替
            """
            for i, enm_info in enumerate(state["enm_infos"]): 
                # 1. 获取记忆 (Rolling Memory)
                if ego.state_memory is None:
                    memory = copy.deepcopy(state)
                else:
                    memory = copy.deepcopy(ego.state_memory)
                # 2、观测限制，记忆重放逻辑
                ATA = enm_info[9]
                dist = enm_info[8]
                
                mem_inertial_beta = enm_info[13]
                mem_inertial_epsilon = enm_info[14]
                
                blind_delta_psi = sub_of_radian(mem_inertial_beta, ego.psi)
                blind_delta_theta = sub_of_radian(mem_inertial_epsilon, ego.theta)
                
                # 情况A: 超出探测距离 -> 完全不可见
                if dist > ego.max_radar_range:
                    # 加载记忆
                    state["enm_infos"][i] = copy.deepcopy(memory["enm_infos"][i])
                    state["enm_infos"][i][2] = 0 # 可见性评级
                    state["enm_infos"][i][5] = cos(blind_delta_psi)
                    state["enm_infos"][i][6] = sin(blind_delta_psi)
                    state["enm_infos"][i][7] = blind_delta_theta
                # 情况B: 中距离
                elif dist > 10e3:
                    locked_by_enm = state["enm_infos"][i][4]
                    # B1: 角度大 且 未被锁定 -> 完全不可见
                    if ATA > self.RUAV.max_radar_angle_rad and (not locked_by_enm):
                        # 加载记忆
                        state["enm_infos"][i] = copy.deepcopy(memory["enm_infos"][i])
                        state["enm_infos"][i][2] = 0 # 完全不可见
                        state["enm_infos"][i][5] = cos(blind_delta_psi)
                        state["enm_infos"][i][6] = sin(blind_delta_psi)
                        state["enm_infos"][i][7] = blind_delta_theta
                    # B2: 角度大 但 被锁定 (RWR告警) -> 方位信息可见
                    elif ATA > self.RUAV.max_radar_angle_rad and locked_by_enm:
                        state["enm_infos"][i][2] = 1 # 部分可见
                        for idx in (8, 10):
                            state["enm_infos"][i][idx] = memory["enm_infos"][i][idx]
                            # 假设敌机正对着我
                            state["enm_infos"][i][11] = pi
                            state["enm_infos"][i][12] = 0
                    # B3: 角度合适 -> 完全可见
                    else:
                        state["enm_infos"][i][2] = 2 # 完全可见
                # 情况C：近距
                else:
                    state["enm_infos"][i][2] = 2 # 完全可见  
        else:
            # 不是pomdp，那记忆也没有意义了
            pass
        
        if reward_fn:
            pass # 不更新记忆
        else:
            # 更新记忆
            ego.state_memory = copy.deepcopy(state)
            # 告警信息不可能冻结，队友信息也不需要冻结，所以只有敌机信息需要冻结

        # 裁剪掉需要裁剪的部分
        state["enm_infos"][:] = state["enm_infos"][:13] # 裁剪掉惯性系信息
        # 剩余部分用作观测函数输出
        
        # 在把 state 传入 scale_state 之前移除时间戳 't'
        if 't' in state:
            del state['t']

        observation = self.scale_state(state)
        return observation

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
        
        # # 敌机全都死了之后可以出界
        # # （警告，这样可能需要同步更改border观测项和奖励）
        # ego_side = UAV.side
        # enm_side = 'b' if ego_side == 'r' else 'r'
        # # 判断敌机阵营是否全灭
        # enm_uavs = self.RUAVs if enm_side =='r' else self.BUAVs
        # if all(u.dead for u in enm_uavs):
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
                        pilot = 'red'+str(UAV.id)
                    elif UAV.side == 'b':
                        color = 'Blue'
                        pilot = 'blue'+str(UAV.id)
                    else:
                        color = 'Black'
                        pilot = 'invader'+str(UAV.id)

                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{UAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{UAV.phi * 180 / pi:.6f}|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Pilot={pilot},Color={color}\n"
                    )
                    # 雷达和锁定 Beam 显示
                    if getattr(UAV, 'radar_on', 0)==0:
                        data_to_send += f"#{send_t:.2f}\n-{UAV.id+1000}\n"
                    else:
                        if getattr(UAV, 'lock_on', 0) == 0:
                            # 正常探测雷达
                            data_to_send += (
                                f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                                f"0|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange=100000, RadarHorizontalBeamwidth=120, RadarVerticalBeamwidth=20\n"
                            )
                        else:
                            # 锁定时的细光束波束
                            target = self.find_uav_with_id(UAV.current_target_id)
                            delta_pos = target.pos_ - UAV.pos_
                            dist = norm(delta_pos)
                            q_beta = atan2(delta_pos[2], delta_pos[0])
                            q_epsilon = atan2(delta_pos[1], sqrt(delta_pos[0]**2 + delta_pos[2]**2))
                            
                            data_to_send += (
                                f"{UAV.id+1000},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                                f"0|{q_epsilon * 180 / pi:.6f}|{q_beta * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.7,Radius=0.0,RadarMode=1,RadarRange={dist:.1f}, RadarHorizontalBeamwidth=5, RadarVerticalBeamwidth=5\n"
                            )

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
                                    f"Name=AIM-120C,Color={color}\n"
                                    )
                    # 导弹雷达波束显示
                    if getattr(missile, 'radar_on', 0):
                        # 导弹锁定哪个目标的判别还没有，先不画锁定
                        data_to_send += (
                            f"{missile.id+1000},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                            f"0|{missile.theta * 180 / pi:.6f}|{missile.psi * 180 / pi:.6f},"
                            f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange={missile.detect_range:.1f}, RadarHorizontalBeamwidth=120, RadarVerticalBeamwidth=120\n"
                        )
                    else:
                        # 导弹存活但雷达关闭，移除可能残留的雷达波束可视
                        data_to_send += f"#{send_t:.2f}\n-{missile.id+1000}\n"

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
