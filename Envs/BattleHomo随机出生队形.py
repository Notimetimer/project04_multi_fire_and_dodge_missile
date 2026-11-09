'''出生点改在外面指定'''

'''多对多需要真正启用cat动作空间的多维度功能。情况涉及到雷达开关'''

import numpy as np
from random import random
import random
import copy
import jsbsim
import sys
import os
import json
import importlib
import copy
from numpy.linalg import norm


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))

from Envs.MissileModel_aim120 import *  # 传统三自由度动力学导弹模型
from Envs.MissileModel_aim9x import aim9x  # 近距格斗弹模型
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
from Envs.UAVmodel6dF16 import UAVModel
# from Envs.UAVmodel6d0309 import UAVModel

# 调用黑名单：删除 PPOHybrid，防止污染命名空间导致外层调用时 IDE 混淆
try:
    del PPOHybrid
except NameError:
    pass

g = 9.81
dt_maneuver = 2 # 0.2
dt_move = 0.02
# report_move_time_rate = int(round(dt_maneuver / dt_move))
RL_controller = 1 # 是否启用RL控制器(否则使用PID)

o00 = np.array([144.7, 13.4])  # 地理原点的经纬
# t = 0
g_ = np.array([0, -g, 0])
# theta_limit = 85 * pi / 180

R_cage = 100e3

# min_height = 0
# max_height = 15e3

R_birth = 40e3

horizontal_center = np.array([0, 0])

# ==================== 个体观测空间字段命名（规范见 Envs/接口.py） ====================
EGO_MAIN_KEYS = ['v_ego', 'h_ego', 'cos_theta', 'sin_theta', 'cos_phi', 'sin_phi']
EGO_CONTROL_KEYS = ['p', 'q', 'r', 'theta_v', 'psi_v', 'alpha_air', 'beta_air', 'Ny']
RADAR_INFO_KEYS = ['radar_on']
BORDER_KEYS = ['d_hor2border', 'border_LR', 'dist_from_center']
ALLY_KEYS = ['radar_on', 'cos_ally_delta_psi', 'sin_ally_delta_psi', 'ally_delta_theta',
             'ally_dist', 'v_ally', 'cos_ally_rel_heading', 'sin_ally_rel_heading', 'theta_ally']
ENM_KEYS = ['enm_observable', 'enm_locked', 'locked_by_enm', 'cos_enm_delta_psi',
            'sin_enm_delta_psi', 'enm_delta_theta', 'enm_dist', 'enm_ATA', 'v_enm', 'cos_enm_AA', 'sin_enm_AA']
ENM_SUPPLEMENT_KEYS = ['enm_q_beta', 'enm_q_epsilon']
WEAPON_KEYS = ['missile_t_fly', 'cos_msl_delta_psi', 'sin_msl_delta_psi', 'mid_term']
RWR_KEYS = ['warning', 'cos_threat_delta_psi', 'sin_threat_delta_psi', 'threat_theta', 'threat_distance']

# fixed 部分展平时的 (顶层key, 子字段序)，子字段序为 None 表示标量
FIXED_OBS_SPEC = [
    # ('ego_alive', None),
    ('ego_main', EGO_MAIN_KEYS),
    ('ego_control', EGO_CONTROL_KEYS),
    ('radar_info', RADAR_INFO_KEYS),
    ('ego_ammo', None), # None表示标量
    ('border', BORDER_KEYS),
    ('weapon_infos', WEAPON_KEYS),
    ('RWR_infos', RWR_KEYS),
]


def _obs_items(x):
    """list-of-dicts 或单个 dict 统一为可迭代的 dict 列表"""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def dicts2array(dicts, keys):
    """list-of-dicts -> (N, len(keys)) 的2D数组，供神经网络输入边界使用"""
    if len(dicts) == 0:
        return np.zeros((0, len(keys)), dtype=np.float32)
    return np.array([[d[k] for k in keys] for d in dicts], dtype=np.float32)


def flatten_fixed_obs(obs):
    """按 FIXED_OBS_SPEC 顺序把 fixed 部分的 dict/标量 展平为一维数组"""
    parts = []
    for key, subkeys in FIXED_OBS_SPEC:
        v = obs[key]
        if subkeys is None:
            parts.append(float(v)) # 标量直接加上
        else:
            for d in _obs_items(v):
                parts.extend(float(d[k]) for k in subkeys)
    return np.array(parts, dtype=np.float32)


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
        self.shielded = True # 默认开启防撞地
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
            if tacview_show == 2:
                self.tacview = Tacview_s()
            else:
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

    def _wall_birth_positions(self, birth_state, num, random_pos, shuffle_seed, ally_dist_init=5e3):
        """wall 队形：各机间距 ally_dist_init，南北一字排开；random_pos 时打乱队内顺序"""
        positions = []
        for index_in_side in range(num):
            # 出生点 (注意：使用 copy 防止修改外部传入的字典数据)
            # 构造一个0, 1, -1, 2，-2 的函数
            # temp = (-1)**index_in_side * int((index_in_side+1)/2)
            # 1/2, -1/2, 3/2, -3/2, 5/2, -5/2...
            temp = (-1)**(index_in_side) * (np.floor(index_in_side/2 + 0.5))
            # 出生点间距ally_dist_init，南北排开
            positions.append(birth_state['position'].copy() + np.array([1, 0, 0]) * temp * ally_dist_init)
        if random_pos:
            # 红蓝双方分别用不同的随机种子打乱队内初始位置排序（局部Random实例，None则从系统熵播种）
            if shuffle_seed is None:
                random.shuffle(positions)
            else:
                random.Random(shuffle_seed).shuffle(positions)
        return positions

    def _wedge_birth_positions(self, birth_state, num, random_pos, shuffle_seed, ally_dist_init=5e3):
        """wedge 队形出生点列表（接口预留，待实现）"""
        positions = []
        # 小id永远更靠近中心，大id更靠后面
        if birth_state['position'][2] > 0: 
            side_word = -1 # 东侧出生，id小则靠西
        else:
            side_word = 1 # 西侧出生，id小则靠东
        for index_in_side in range(num):
            positions.append(birth_state['position'].copy() + np.array([0, 0, side_word]) * ally_dist_init * (
                num - (index_in_side+1)
            ))
        return positions

    def _v_birth_positions(self, birth_state, num, random_pos, shuffle_seed, ally_dist_init=5e3):
        """v 队形出生点列表（接口预留，待实现）"""
        positions = []
        # 小id永远更靠近中心，大id更靠后面
        if birth_state['position'][2] > 0: 
            side_word = -1 # 东侧出生，id小则靠西
        else:
            side_word = 1 # 西侧出生，id小则靠东
        if random_pos:
            temp = random.choice([0,1]) # 提供南北的随机性
        else:
            temp = 0
        for index_in_side in range(num):
            # 出生点 (注意：使用 copy 防止修改外部传入的字典数据)
            # 构造一个0, 1, -1, 2，-2 的函数
            # temp = (-1)^(t)*floor((t+1)/2)
            temp_NS = (-1)**(index_in_side + temp) * np.ceil(index_in_side/2)
            temp_EW = np.ceil((num-1)/2) - np.ceil(index_in_side/2)
            # 出生点间距ally_dist_init，南北排开
            positions.append(birth_state['position'].copy() + ally_dist_init/np.sqrt(2) * \
                (np.array([1, 0, 0]) * temp_NS + np.array([0, 0, side_word]) * temp_EW))
        return positions

    def reset(self, red_birth_state=None, blue_birth_state=None, 
              red_init_num=2, blue_init_num=2,
              red_init_ammo=4, blue_init_ammo=4, 
              red_alive_num=None, blue_alive_num=None,
              red_init_formation = 'random', blue_init_formation = 'random', # 初始队形
              pomdp=1, seed=None, options={"random_pos": True},
              ego_side='r'):  # 重置位置和状态
        # 初始编队合法类型：随机、墙、楔、v形
        check_formation_legal_list = ['wall', 'wedge', 'v'] # random
        
        # 随机初始化处理
        if red_init_formation == 'random': # not in check_formation_legal_list
            red_init_formation = random.choice(check_formation_legal_list)
        if blue_init_formation == 'random':
            blue_init_formation = random.choice(check_formation_legal_list)
        
        # 红方已取得分配的蓝方id清零
        self.allocatedIDs_r2b = set()
        # 蓝方已取得分配的红方id清零
        self.allocatedIDs_b2r = set()
        # 胜负情况
        self.r_outcome = None
        self.b_outcome = None
        self.R_cage = self.R_cage0
        if self.vertices is not None:
            self.horizontal_center = np.mean(self.vertices, axis=0)
        else:
            self.horizontal_center = horizontal_center

        self.ego_side = ego_side
        
        self.last_record_t = -10
        
        # [新增] 如果需要支持随机种子控制，可以在这里设置
        # seed 仅用于队内初始位置排序打乱：int 或 (red_seed, blue_seed)，红蓝各自独立；
        # 用局部 Random 实例，不播种全局随机流（np.random 噪声序列不受 reset 影响）
            
        if red_birth_state is None:
            red_birth_state = self.RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.BLUE_BIRTH_STATE
        self.BLUE_BIRTH_STATE = blue_birth_state
        self.RED_BIRTH_STATE = red_birth_state
        
        self.dt_maneuver = dt_maneuver  # simulation interval，1 second
        self.t = 0
        # get_state 缓存：以 self.t 为代次，t 推进时整体失效（每个 maneuver 步重建一次）
        self._state_cache = {}
        self._state_cache_t = None
        # 初始化无人机
        self.Rnum = red_init_num
        self.Bnum = blue_init_num
        
        # 原始可见性矩阵（0=不可见, 1=仅方位可见, 2=完全可见），必须用 int 才能区分 1 和 2
        self.prim_r2b_visable = np.ones((self.Rnum, self.Bnum), dtype=np.int8)
        self.prim_b2r_visable = np.ones((self.Bnum, self.Rnum), dtype=np.int8)
        # 组合视野可见性矩阵（队内共享后，取值同上）
        self.r2b_visable = np.ones((self.Rnum, self.Bnum), dtype=np.int8)
        self.b2r_visable = np.ones((self.Bnum, self.Rnum), dtype=np.int8)
        # id -> 队内位置索引（逆向索引可由 {v:k for k,v in d.items()} 得到，不再单独维护）
        self.r_id2indexinside={}
        self.b_id2indexinside={}
        
        self.RUAVsTable = {}
        self.BUAVsTable = {}
        self.UAVsTable = {}
        self.alive_r_missiles = []
        self.alive_b_missiles = []
        self.alive_missiles = []
        self.win = 0
        self.lose = 0
        self.draw = 0
        # 打乱编队内位置顺序（目前只在 wall 队形内部生效）
        formation_position_funcs = {
            'wall': self._wall_birth_positions,
            'wedge': self._wedge_birth_positions,
            'v': self._v_birth_positions,
        }
        assert red_init_formation in formation_position_funcs, f'非法红方初始队形: {red_init_formation}'
        assert blue_init_formation in formation_position_funcs, f'非法蓝方初始队形: {blue_init_formation}'
        # 红蓝双方分别用不同的随机种子（局部Random实例，None则从系统熵播种）
        if seed is None:
            red_seed, blue_seed = None, None
        else:
            red_seed, blue_seed = seed if isinstance(seed, (tuple, list)) else (seed, seed + 1)
        red_birth_positions = formation_position_funcs[red_init_formation](
            red_birth_state, self.Rnum, options.get("random_pos"), red_seed)
        blue_birth_positions = formation_position_funcs[blue_init_formation](
            blue_birth_state, self.Bnum, options.get("random_pos"), blue_seed)
        
        # 无人机统一初始化
        self.alive_r_uav_avg_pos_ = None # 平均位置
        self.alive_b_uav_avg_pos_ = None # 平均位置
        for i in range(self.Rnum + self.Bnum):
            is_red = i < self.Rnum
            index_in_side = i if is_red else i - self.Rnum
            # 初始化组合视野过程中id对位置的映射
            if is_red:
                self.r_id2indexinside[(index_in_side + 1)] = index_in_side
            else:
                self.b_id2indexinside[(index_in_side + 201)] = index_in_side
            UAV = UAVModel(dt=self.dt_move)
            UAV.state_memory = None
            UAV.action_memory = np.array([0, 0, 340])
            UAV.last_state = None
            UAV.current_state = None
            UAV.launch_states = []
            UAV.launch_states_order = None
            UAV.init_ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.ammo = red_init_ammo if is_red else blue_init_ammo
            UAV.last_ammo_whole_cycle = UAV.ammo
            UAV.id = (index_in_side + 1) if is_red else (index_in_side + 201)
            UAV.red = is_red
            UAV.blue = not is_red
            UAV.side = 'r' if is_red else 'b'
            UAV.color = np.array([1, 0, 0]) if is_red else np.array([0, 0, 1])
            UAV.launched_aim120s = []
            birth_state = red_birth_state if is_red else blue_birth_state
            
            if is_red:
                UAV.pos_ = red_birth_positions[index_in_side]
            else:
                UAV.pos_ = blue_birth_positions[index_in_side]
            
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
            # 红蓝两侧统一初速（原实现红 300 / 蓝 (speed_max-speed_min)/2，存在阵营不对称）
            default_speed = 300
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
            UAV.fire_hit_records = []  # 记录所有导弹的命中情况
            UAV.got_hit = False
            UAV.chosen_by_opp = 0 # 是否被对手选中
            UAV.radar_on = 1
            UAV.radar_center = np.array([0, 0]) # 雷达中心相对方位的归一化值，先水平后垂直
            UAV.escape_once = 0 # 逃脱导弹攻击
            UAV.got_a_miss = 0 # 发射的导弹脱靶
            UAV.got_a_kill = 0 # 发射的导弹命中目标
            UAV.can_let_go = 0 # 发射的导弹锁定目标
            
            if is_red:
                self.RUAVsTable[UAV.id] = UAV
            else:
                self.BUAVsTable[UAV.id] = UAV
                
            UAV.lock_on = 0
            UAV.current_target_id = None
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
        # alive_num 优先级：reset 显式传入 > init_states.json 读取
        if red_alive_num is None or blue_alive_num is None:
            # 按 init_states.json 的 alive_num 随机击杀差额飞机，直到该方剩余 alive_num 架存活
            init_states_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TrainAndTests', 'Combats', 'init_states.json')
            with open(init_states_path, 'r', encoding='utf-8') as f:
                init_states = json.load(f)
            if red_alive_num is None:
                red_alive_num = init_states.get('red_alive_num', self.Rnum)
            if blue_alive_num is None:
                blue_alive_num = init_states.get('blue_alive_num', self.Bnum)
        # 击杀选取改为确定性：按 id 升序保留、id 最大的先杀，存活结果与随机种子无关
        for side_uavs, alive_num in ((self.RUAVs, red_alive_num),
                                     (self.BUAVs, blue_alive_num)):
            kill_num = len(side_uavs) - alive_num
            if kill_num > 0:
                for uav in sorted(side_uavs, key=lambda u: u.id, reverse=True)[:kill_num]:
                    uav.dead = 1
                    uav.radar_on = 0
        # reset后、首个step前就填充存活列表，保证决策循环可访问 alive_*_uavs
        self.update_entity_alives()
        
        # 初始化 last_dead 属性，防止死亡惩罚重复计算
        for ego in self.UAVsTable.values():
            ego.last_dead = False
            # 周期事件缓存：瞬时标志 -> 整周期标志 -> 周期边界统一结算
            ego.whole_cycle_escape_once = 0
            ego.whole_cycle_got_a_miss = 0
            ego.whole_cycle_got_a_kill = 0
            ego.whole_cycle_can_let_go = 0
            ego.cycle_events = []          # [(t, id, 事件内容)]
            ego.last_cycle_t = -1e9        # 本机上次结算周期边界的时间戳
        
        # 初始化包围角
        self.last_r2b_surround_angle = None
        self.last_b2r_surround_angle = None

        # [确认存在/修改] 确保每个 Episode 开始时重置 last_obs
        self.last_obs = None 
        self.pomdp = pomdp   

        # 开场数据支持，至少背对背决斗时需要知道“对手在后面”
        for id, uav in self.UAVsTable.items():
            uav.state_memory = copy.deepcopy(self.get_state(id))
    
        
    # 更新存活
    def update_entity_alives(self,):
        # 更新飞机存活状态
        alive_r_uavs = [u for u in self.RUAVs if not u.dead]
        alive_b_uavs = [u for u in self.BUAVs if not u.dead]
        self.alive_r_uav_avg_pos_ = np.zeros(3)
        self.alive_b_uav_avg_pos_ = np.zeros(3)
        for u in alive_r_uavs:
           self.alive_r_uav_avg_pos_ += u.pos_ / len(alive_r_uavs)
        for u in alive_b_uavs:
           self.alive_b_uav_avg_pos_ += u.pos_ / len(alive_b_uavs)
        self.alive_r_uavs = alive_r_uavs
        self.alive_b_uavs = alive_b_uavs
        self.alive_uavs = alive_r_uavs + alive_b_uavs
        # 更新导弹存活状态
        alive_r_missiles = [m for m in self.Rmissiles if not m.dead]
        alive_b_missiles = [m for m in self.Bmissiles if not m.dead]
        self.alive_r_missiles = alive_r_missiles
        self.alive_b_missiles = alive_b_missiles
        self.alive_missiles = alive_r_missiles + alive_b_missiles
        return alive_r_uavs, alive_b_uavs
    
    # 更新所有实体状态
    def update_entity_states(self,):
        alive_r_uavs, alive_b_uavs = self.update_entity_alives()
        # 统计未发射导弹数
        self.unused_r_missile_num = 0
        self.unused_b_missile_num = 0
        for uav in self.RUAVs:
            self.unused_r_missile_num += 0 if uav.dead else uav.ammo
            # 更新观测有效性矩阵
            self.get_primitive_obs(uav.id)
        for uav in self.BUAVs:
            self.unused_b_missile_num += 0 if uav.dead else uav.ammo
            # 更新观测有效性矩阵
            self.get_primitive_obs(uav.id)
        # 队内共享视野
        for i, uav in enumerate(self.BUAVs):
            if np.sum(self.prim_r2b_visable[:, i]) >= 2:
                # 不管是单架飞机有全信息还是多星测角定位，都有完全信息
                self.r2b_visable[:, i] = 2
            else:
                # 按照原始定位来
                self.r2b_visable[:, i] = self.prim_r2b_visable[:, i]
        for i, uav in enumerate(self.RUAVs):
            if np.sum(self.prim_b2r_visable[:, i]) >= 2:
                # 不管是单架飞机有全信息还是多星测角定位，都有完全信息
                self.b2r_visable[:, i] = 2
            else:
                # 按照原始定位来
                self.b2r_visable[:, i] = self.prim_b2r_visable[:, i]
        
        # 导弹重新搜索目标 120只能在目标死后重新搜索目标
        for missile in self.alive_missiles:
            if missile.WeaponType == "AIM-120C":
                # 目标没死不许换目标
                target = self.find_uav_with_id(missile.target_id)
                if not target.dead:
                    continue
                # 雷达没开，靠载机数据链更换目标（无接力制导）
                if not missile.radar_on:
                    launcher_id = missile.launcher_id
                    launcher_uav = self.find_uav_with_id(launcher_id)
                    # 载机死了就别想换目标
                    if launcher_uav.dead:
                        continue
                    # 载机没死，载机锁谁我跟谁
                    # 如果载机能锁我就换，载机锁不了我continue
                    if getattr(launcher_uav, 'lock_on', 0) == 1 and getattr(launcher_uav, 'current_target_id', None) is not None:
                        target_id = launcher_uav.current_target_id
                        missile.target_id = target_id
                        continue
                    else:
                        continue
            # 导引头开了，数据链靠边站
            speed = norm(missile.vel_)
            if speed == 0:
                continue
            # 靠自身雷达/红外导引头重搜索目标
            replacement_id = None
            # replacement_distance = float('inf')
            replacement_off_axis = missile.sight_angle_max
            enemy_uavs = alive_b_uavs if missile.side == 'r' else alive_r_uavs
            for uav in enemy_uavs:
                line_of_sight = uav.pos_ - missile.pos_
                distance = norm(line_of_sight)
                # 超出探测距离不许换
                if distance == 0 or distance > missile.detect_range:
                    continue
                # 探测距离之内找角度偏移最小的
                sight_cos = np.dot(missile.vel_, line_of_sight) / (speed * distance)
                if sight_cos > np.cos(replacement_off_axis):
                    replacement_id = uav.id
                    # replacement_distance = distance
                    replacement_off_axis = np.arccos(sight_cos)
            if replacement_id is not None:
                missile.target_id = replacement_id
        
    # 根据飞机id查找飞机类实例
    def find_uav_with_id(self, id):
        uav = self.UAVsTable[id]
        return uav
    # 根据导弹id查找导弹类实例
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
        
    # 发射中距弹
    def launch_120(self, source_id, target_id):
        """
        source_id: 发射导弹的无人机 id
        target_id: 被击中的无人机 id
        """
        uav = self.find_uav_with_id(source_id)
        target = self.find_uav_with_id(target_id)
        side = uav.side
        # 发射导弹
        if uav.ammo>0 and not uav.dead:
            new_missile = uav.launch_120(target, self.t, aim120)
            uav.ammo -= 1
            uav.launched_aim120s.append(new_missile) # 无人机类自己也记录一份发射列表
            new_missile.side = side
            new_missile.launcher_id = source_id
            new_missile.target_id = target_id
            new_missile.init_target_id = target_id
            new_missile.hit_supposed = False
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
    
    # 发射近距弹
    def launch_9x(self, source_id, target_id):
        uav = self.find_uav_with_id(source_id)
        target = self.find_uav_with_id(target_id)
        side = uav.side
        # 发射导弹
        if uav.ammo_9x>0 and not uav.dead:
            new_missile = uav.launch_9x(target, self.t, aim9x)
            if new_missile is None:
                return None
            new_missile.side = side
            new_missile.launcher_id = source_id
            new_missile.target_id = target_id
            new_missile.init_target_id = target_id
            new_missile.hit_supposed = False
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

    # TODO 输入动作打包为大小不确定的量
    def step(self, flight_cmds):
        # allocatedIDs_r2b/b2r 不在 step 内清零：此处清零会让随后的 reward 永远读到空集，
        # 清理由外部决策脚本按决策轮负责（每轮决策时覆盖写入）
        "flight_cmds 为一个list, 用id顺序排列所有飞机"
        # "actions_of_ids: {id: action}"
        report_move_time_rate = int(round(self.dt_maneuver / self.dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        # flight_cmds = [actions_of_ids[id] for id in self.UAV_ids]

        # 在整个 maneuver step 开始时只重置一次 escape_once（不要在内层子步或导弹循环中再次重置）
        for UAV in self.UAVs:
            UAV.escape_once = 0 # 逃脱导弹攻击
            UAV.got_a_miss = 0 # 发射的导弹脱靶
            UAV.got_a_kill = 0 # 发射的导弹命中目标
            UAV.can_let_go = 0 # 发射的导弹锁定目标
            UAV.chosen_by_opp = 0

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
                        d_hor, border_side = self.polygon_fences.calc_intern_dist(UAV.pos_, UAV.psi_v)
                    else:
                        d_hor, border_side = calc_intern_dist2circle(self.R_cage, UAV.pos_, UAV.psi_v)
                    if d_hor < 8e3: # 8e3
                        if border_side > 0:
                            target_delta_heading = min(-pi/2, target_delta_heading)
                        else:
                            target_delta_heading = max(pi/2, target_delta_heading)
                
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
                e2e = RL_controller
                # 临时改动，关闭端到端控制，改为PID
                if e2e == 1:
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
            self.update_entity_states() # 先把存活的导弹找出来
            for missile in self.alive_missiles[:]:  # 使用切片创建副本以允许删除
                target = self.find_uav_with_id(missile.target_id)
                # todo 干扰或更换目标逻辑，写在这里，箔条设置为从属于UAV的属性 
                # track_target = f(target, missile, self.UAVs, self.chaffs)

                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                # elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹
                #     missile.dead = True # todo 多对多环境中改成missile.target = None, 并在missile类里改成丢失目标飞直线，并且无法触发hit
                #     continue
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
                if (not missile.dead) and \
                    (missile.last_guidance_stage == 2 and missile.guidance_stage == 3):
                    # 谁发射的导弹锁到人了
                    shooter = self.find_uav_with_id(missile.launcher_id)
                    shooter.can_let_go = 1

                if missile.t >= 0 + self.dt_move and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=self.dt_move, kill_range=missile.kill_range)
                    if hit:
                        print(target.side, 'is hit')
                        missile.dead = True
                        missile.hit = True
                        if target.id == missile.init_target_id:
                            missile.hit_supposed = True
                        missile.pos_ = point_m
                        missile.vel_ = last_vmt_
                        target.pos_ = point_t
                        target.vel_ = last_vtt_
                        target.dead = True
                        target.got_hit = True
                        target.radar_on = 0
                        self.UAV_shot[self.UAV_ids.index(target.id)] = True
                        # 谁扔的刀扎到人了
                        shooter = self.find_uav_with_id(missile.launcher_id)
                        shooter.got_a_kill = 1

                if missile.dead == True and not target.dead:
                    # 目标逃脱
                    target.escape_once = 1
                    # 谁的导弹错过了目标
                    shooter = self.find_uav_with_id(missile.launcher_id)
                    shooter.got_a_miss = 1


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
        self.update_entity_alives()  # 更新飞机存活信息

        return None

    def _generate_state_init_mod(self):
        # 部分观测填充值模板（字典结构，字段命名见 Envs/接口.py）
        state_init_mod = {}
        # 只有敌机、队友、已发射的导弹、来袭导弹需要这套模板
        state_init_mod["ally_info"] = {
            'radar_on': 1,
            'cos_ally_delta_psi': -1,
            'sin_ally_delta_psi': 0,
            'ally_delta_theta': 0,
            'ally_dist': 100e3,
            'v_ally': 0,
            'cos_ally_rel_heading': 0,
            'sin_ally_rel_heading': 0,
            'theta_ally': 0,
        }
        state_init_mod["enm_info"] = {
            'enm_observable': 0,
            'enm_locked': 0,
            'locked_by_enm': 0,
            'cos_enm_delta_psi': 1,
            'sin_enm_delta_psi': 0,
            'enm_delta_theta': 0,
            'enm_dist': 100e3,
            'enm_ATA': 0,
            'v_enm': 0,
            'cos_enm_AA': -1,
            'sin_enm_AA': 0,
        }
        state_init_mod["enm_info_supplement"] = {
            'enm_q_beta': 0,
            'enm_q_epsilon': 0,
        }
        state_init_mod["weapon_info"] = {
            'missile_t_fly': 120,
            'cos_msl_delta_psi': 1,
            'sin_msl_delta_psi': 0,
            'mid_term': 0,
        }
        state_init_mod["RWR_info"] = {
            'warning': 0,
            'cos_threat_delta_psi': 1,
            'sin_threat_delta_psi': 0,
            'threat_theta': 0,
            'threat_distance': 100e3,
        }
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
        # 带 t 标记的缓存：同一仿真时刻（同一 maneuver 步）内的重复调用直接复用
        # 出口统一返回副本：调用方会就地修改 state，隔离写入以保护缓存与 ego.current_state
        if self._state_cache_t == self.t and id in self._state_cache:
            return copy.deepcopy(self._state_cache[id])
        # 原先将所有量打包成一个 numpy array，这里改为 dict 结构（字段命名见 Envs/接口.py）
        self.key_order = [
            # 我机信息
            "ego_alive",  # 长度为1
            "radar_info", 
            "ego_main",  # 固定长度dict
            "ego_control",  # 固定长度dict
            "ego_ammo",  # 剩余弹量
            "border", # 边界信息dict
            # 队友信息
            "ally_infos",  # list[dict]
            # 敌机信息
            "enm_infos",  # list[dict]
            "enm_info_supplement",  # list[dict]，惯性系方位，供记忆/规则用，不进神经网络
            # 导弹信息（不区分发射机，预留接力制导空间）
            "weapon_infos",  # list[dict]
            # 告警信息
            "RWR_infos",  # list[dict]
        ]

        ego = self.find_uav_with_id(id)
        ego_side = ego.side

        # 先自己、然后队友，最后敌机，队友和敌机内部按照id排序
        # 0. 标识符
        ego_alive = not ego.dead  # 自己活着没

        # 边界信息
        if self.vertices is not None:
            d_hor, border_side = self.polygon_fences.calc_intern_dist(ego.pos_, ego.psi)
        else:
            d_hor, border_side = calc_intern_dist2circle(self.R_cage, ego.pos_, ego.psi)

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
        radar_info = {
            "radar_on": ego.radar_on
        }
        
        ego_main = {
            'v_ego': float(v_ego),        # 本机速度 m/s
            'h_ego': float(h_ego),        # 本机高度 m
            'cos_theta': float(cos_theta),
            'sin_theta': float(sin_theta),
            'cos_phi': float(cos_phi),
            'sin_phi': float(sin_phi),
        }

        ego_control = {
            'p': float(ego.p),            # p rad/s act1_last
            'q': float(ego.q),            # q rad/s act2_last
            'r': float(ego.r),            # r rad/s act3_last
            'theta_v': float(ego.theta_v),
            'psi_v': float(ego.psi_v),
            'alpha_air': float(ego.alpha_air),  # rad
            'beta_air': float(ego.beta_air),    # rad
            'Ny': float(ego.Ny),          # overload
        }

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
            # 死队友直接塞填充值，不浪费计算量
            if ally.dead: 
                ally.radar_on = 0
                ally_infos.append(self.state_init_mod["ally_info"].copy())
                continue
            
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

            one_ally_info = {
                'radar_on': float(ally.radar_on),
                'cos_ally_delta_psi': float(cos(delta_psi)),
                'sin_ally_delta_psi': float(sin(delta_psi)),
                'ally_delta_theta': float(delta_theta),
                'ally_dist': float(dist),
                'v_ally': float(v_ally),
                'cos_ally_rel_heading': float(cos(delta_psi_ally)),
                'sin_ally_rel_heading': float(sin(delta_psi_ally)),
                'theta_ally': float(theta_ally),
            }

            ally_infos.append(one_ally_info)

        #------------------------
        # 3. 敌机信息:
        # [存活、可见性、是否能锁定、是否锁定我、相对方位、相对俯仰角、相对距离、速度、hot/cold]*数量
        #------------------------
        ego.lock_on = 0
        enm_infos = []
        enm_info_supplement = []
        enm_uavs = self.RUAVs if ego_side =='b' else self.BUAVs
        for enm in enm_uavs:
            # 死目标直接塞填充值，不浪费计算量
            if enm.dead:
                enm.radar_on=0
                enm_infos.append(self.state_init_mod["enm_info"].copy())
                enm_info_supplement.append(self.state_init_mod["enm_info_supplement"].copy())
                continue
            
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

            
            enm_infos.append({
                'enm_observable': float(enm_observable),
                'enm_locked': float(enm_locked),
                'locked_by_enm': float(locked_by_enm),
                'cos_enm_delta_psi': float(cos(delta_psi)),
                'sin_enm_delta_psi': float(sin(delta_psi)),
                'enm_delta_theta': float(delta_theta),
                'enm_dist': float(dist),
                'enm_ATA': float(ATA),
                'v_enm': float(enm_speed),
                'cos_enm_AA': float(cos(AA_hor)),
                'sin_enm_AA': float(sin(AA_hor))
            })
            # 惯性系下的位置关系，进入记忆/规则，不进神经网络
            enm_info_supplement.append({
                'enm_q_beta': float(q_beta),
                'enm_q_epsilon': float(q_epsilon),
            })

        #------------------------
        # 导弹信息向量，协同制导我还没想清楚要如何做，先按不做的来
        #------------------------
        alive_ally_missiles = self.alive_r_missiles if ego_side=='r' else self.alive_b_missiles
        alive_ego_missiles = [m for m in alive_ally_missiles if m.launcher_id == ego.id]
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
            ally_missile_info.append({
                'missile_t_fly': float(missile.t),        # 飞行时间
                'cos_msl_delta_psi': float(cos(delta_psi)),
                'sin_msl_delta_psi': float(sin(delta_psi)),
                'mid_term': float(missile_in_mid_term),   # 中制导标志
            })

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
                RWR_infos.append({
                    'warning': float(warning),
                    'cos_threat_delta_psi': float(cos(threat_delta_psi)),
                    'sin_threat_delta_psi': float(sin(threat_delta_psi)),
                    'threat_theta': float(threat_theta),
                    'threat_distance': float(threat_distance),
                })
       
        one_state = {
            "ego_main": ego_main,
            "ego_control": ego_control,
            "ego_ammo": ammo,
            "radar_info": radar_info,
            "border": {
                'd_hor2border': float(d_hor),
                'border_LR': int(border_side),
                'dist_from_center': float(norm([
                    ego.pos_[0]-self.horizontal_center[0],
                    ego.pos_[2]-self.horizontal_center[1]
                ])),
            },
            "ally_infos": ally_infos,
            "enm_infos": enm_infos,
            "enm_info_supplement": enm_info_supplement,
            "weapon_infos": ally_missile_info,
            "RWR_infos": RWR_infos,
        }

        if self._state_cache_t != self.t:  # t 已推进，旧代次整体作废
            self._state_cache = {}
            self._state_cache_t = self.t
        # 缓存存独立副本，'t' 时间戳由缓存副本携带（供 base_obs 归档逻辑比对，scale 前会被移除）
        self._state_cache[id] = copy.deepcopy(one_state)
        self._state_cache[id]['t'] = self.t
        # current_state 指向缓存副本：既保证归档逻辑能读到 't'，又不会被调用方就地修改污染
        ego.current_state = self._state_cache[id]
        return one_state
    
    # 尺度缩放（按字段名索引）
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"]["v_ego"] /= 340
            s["ego_main"]["h_ego"] /= 5e3
        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"]["p"] /= (2 * pi)
            s["ego_control"]["q"] /= (2 * pi)
            s["ego_control"]["r"] /= (2 * pi)
            s["ego_control"]["Ny"] /= 5
        if "border" in s and s["border"] is not None:
            s["border"]["d_hor2border"] = min(1, s["border"]["d_hor2border"] / 50e3)
            s["border"]["dist_from_center"] /= self.R_cage
        if "ally_infos" in s:
            for ally in s["ally_infos"]:
                ally["ally_dist"] /= 50e3
                ally["v_ally"] /= 340
        if "enm_infos" in s:
            for enm in s["enm_infos"]:
                enm["enm_dist"] /= 50e3
                enm["v_enm"] /= 340
        if "weapon_infos" in s:
            for w in s["weapon_infos"]:
                w["missile_t_fly"] /= 120.0
        if "RWR_infos" in s:
            for rwr in s["RWR_infos"]:
                rwr["threat_distance"] /= 50e3
        return s

    def unscale_state(self, obs_input):
        """把 scale_state 的缩放还原。仅判断 key 是否存在（不再检查长度）。"""
        s = copy.deepcopy(obs_input)
        if "ego_main" in s and s["ego_main"] is not None:
            s["ego_main"]["v_ego"] *= 340
            s["ego_main"]["h_ego"] *= 5e3
        if "ego_control" in s and s["ego_control"] is not None:
            s["ego_control"]["p"] *= (2 * pi)
            s["ego_control"]["q"] *= (2 * pi)
            s["ego_control"]["r"] *= (2 * pi)
            s["ego_control"]["Ny"] *= 5
        if "border" in s and s["border"] is not None:
            s["border"]["d_hor2border"] *= 50e3
            s["border"]["dist_from_center"] *= self.R_cage
        if "ally_infos" in s:
            for ally in s["ally_infos"]:
                ally["ally_dist"] *= 50e3
                ally["v_ally"] *= 340
        if "enm_infos" in s:
            for enm in s["enm_infos"]:
                enm["enm_dist"] *= 50e3
                enm["v_enm"] *= 340
        if "weapon_infos" in s:
            for w in s["weapon_infos"]:
                w["missile_t_fly"] *= 120
        if "RWR_infos" in s:
            for rwr in s["RWR_infos"]:
                rwr["threat_distance"] *= 50e3
        return s
    
    def get_primitive_obs(self, id):
        # 判断是否具有原始观测
        ego = self.find_uav_with_id(id)
        if ego.side == 'r':
            ego_id2indexinside = self.r_id2indexinside
            enm_id2indexinside = self.b_id2indexinside
            our_prim_visable = self.prim_r2b_visable
        else:
            ego_id2indexinside = self.b_id2indexinside
            enm_id2indexinside = self.r_id2indexinside
            our_prim_visable = self.prim_b2r_visable
            
        # 我在原始可见性矩阵中在第几行
        ego_indexinside = ego_id2indexinside[id]
        
        state = self.get_state(id)
        # 对每个敌机分别处理可见性问题
        for i, enm_info in enumerate(state["enm_infos"]):
            # 观测限制，记忆重放逻辑
            ATA = enm_info["enm_ATA"]
            dist = enm_info["enm_dist"]
            # 情况A: 超出探测距离 -> 完全不可见
            if dist > ego.max_radar_range:
                our_prim_visable[ego_indexinside, i] = 0  # 看不见
            # 情况B: 中距离
            elif dist > 10e3:
                locked_by_enm = state["enm_infos"][i]["locked_by_enm"]
                # B1: 角度大 且 未被锁定 -> 完全不可见
                if ATA > ego.max_radar_angle_rad and (not locked_by_enm):
                    our_prim_visable[ego_indexinside, i] = 0  # 看不见
                # B2: 角度大 但 被锁定 (RWR告警) -> 方位信息可见
                elif ATA > ego.max_radar_angle_rad and locked_by_enm:
                    our_prim_visable[ego_indexinside, i] = 1  # 看得到角度
                # B3: 角度合适 -> 完全可见
                else:
                    our_prim_visable[ego_indexinside, i] = 2  # 完全能看见
            # 情况C：近距
            else:
                our_prim_visable[ego_indexinside, i] = 2 # 完全能看见
            # D: 拒绝泉水指挥官
            if ego.dead:
                our_prim_visable[ego_indexinside, i] = 0  # 死了看不见
                
    
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
            # 根据组合视野可见性矩阵决定记忆使用策略，不再重复计算 ATA 和 dist
            # 可见性等级：0=完全不可见(全用记忆), 1=仅方位可见(距离/速度用记忆), 2=完全可见(不用记忆)
            if ego.side == 'r':
                vis_row = self.r2b_visable[self.r_id2indexinside[id], :]
            else:
                vis_row = self.b2r_visable[self.b_id2indexinside[id], :]

            # 记忆只加载一次，避免每个敌机都 deepcopy
            memory = copy.deepcopy(ego.state_memory) if ego.state_memory is not None else copy.deepcopy(state)

            for i, enm_info in enumerate(state["enm_infos"]):
                vis_level = int(vis_row[i])

                if vis_level == 0:
                    # 完全不可见 -> 全部用记忆，并用惯性系位置推算当前相对方位
                    state["enm_infos"][i] = copy.deepcopy(memory["enm_infos"][i])
                    state["enm_infos"][i]["enm_observable"] = 0
                    mem_inertial_beta = state["enm_info_supplement"][i]["enm_q_beta"]
                    mem_inertial_epsilon = state["enm_info_supplement"][i]["enm_q_epsilon"]
                    state["enm_infos"][i]["cos_enm_delta_psi"] = cos(sub_of_radian(mem_inertial_beta, ego.psi))
                    state["enm_infos"][i]["sin_enm_delta_psi"] = sin(sub_of_radian(mem_inertial_beta, ego.psi))
                    state["enm_infos"][i]["enm_delta_theta"] = sub_of_radian(mem_inertial_epsilon, ego.theta)

                elif vis_level == 1:
                    # 仅方位可见（RWR告警）-> 距离和速度用记忆，假设敌机正对着我
                    state["enm_infos"][i]["enm_observable"] = 1
                    state["enm_infos"][i]["enm_dist"] = memory["enm_infos"][i]["enm_dist"]
                    state["enm_infos"][i]["v_enm"] = memory["enm_infos"][i]["v_enm"]
                    state["enm_infos"][i]["cos_enm_AA"] = -1
                    state["enm_infos"][i]["sin_enm_AA"] = 0

                else:  # vis_level == 2
                    # 完全可见 -> 不用记忆
                    state["enm_infos"][i]["enm_observable"] = 2
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
        del state["enm_info_supplement"] # 惯性系信息不进神经网络观测
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
            # 多边形边界计算
            # p = Point(np.array([UAV.pos_[0], UAV.pos_[2]]))
            # out = not self.polygon_fences.poly.contains(p)
            # 方形边界快速判定，替代 shapely contains
            out = not self.polygon_fences.is_inside(UAV.pos_)
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
                        pilot = str(UAV.id)#'red'+str(UAV.id)
                    elif UAV.side == 'b':
                        color = 'Blue'
                        pilot = str(UAV.id)#'blue'+str(UAV.id)
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
                        if getattr(UAV, 'lock_on', 0) == 0 or getattr(UAV, 'current_target_id', None) is None:
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
                                    f"Name={missile.WeaponType},Color={color}\n"
                                    )
                    # 导弹雷达波束显示（9x为红外弹，radar_on只表示视场，不画波束）
                    if getattr(missile, 'radar_on', 0) and not isinstance(missile, aim9x):
                        lock_target = None
                        if getattr(missile, 'lock_on', 0) and missile.target_id in self.UAVsTable:
                            lock_target = self.find_uav_with_id(missile.target_id)
                            if lock_target.dead:
                                lock_target = None
                        if lock_target is not None:
                            # 已锁定：指向目标的细光束，与飞机雷达锁定同款画法
                            delta_pos = lock_target.pos_ - missile.pos_
                            dist = norm(delta_pos)
                            q_beta = atan2(delta_pos[2], delta_pos[0])
                            q_epsilon = atan2(delta_pos[1], sqrt(delta_pos[0]**2 + delta_pos[2]**2))
                            data_to_send += (
                                f"{missile.id+1000},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                                f"0|{q_epsilon * 180 / pi:.6f}|{q_beta * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.7,Radius=0.0,RadarMode=1,RadarRange={dist:.1f}, RadarHorizontalBeamwidth=5, RadarVerticalBeamwidth=5\n"
                            )
                        else:
                            # 未锁定：正常探测波束
                            data_to_send += (
                                f"{missile.id+1000},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|"
                                f"0|{missile.theta * 180 / pi:.6f}|{missile.psi * 180 / pi:.6f},"
                                f"Type=Beam, Color={color},Visible=0.3,Radius=0.0,RadarMode=1,RadarRange={missile.detect_range:.1f}, RadarHorizontalBeamwidth=80, RadarVerticalBeamwidth=80\n"
                            )
                    else:
                        # 导弹存活但雷达关闭/红外弹，移除可能残留的雷达波束可视
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
