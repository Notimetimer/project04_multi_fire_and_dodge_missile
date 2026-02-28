'''
先过程奖励后结果奖励

动作空间更改，crank从30度和60度的区分改为左右的区分
'''

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
from math import *
import torch

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from Envs.battle6dof1v1_missile0919 import *
from Envs.Tasks.AttackManeuverEnv import *
from Envs.Tasks.CrankManeuverEnv import *
from Envs.Tasks.EscapeManeuverEnv import *
from Algorithms.Rules import *


# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

# 参考论文中的动作划分
action_options = {
                    0: "track",
                    1: "30track",
                    2: "60track",
                    3: "-30track",
                    4: "-60track",
                    5: "+-30crank",
                    6: "+-60crank",
                    7: "snake",
                    8: "splitS",
                    9: "39",
                    10: "slowTurn",
                    11: "fastTurn",
                    12: "-30turn",
                    13: "-60turn",
                }
# 区分左右的动作划分
action_optionsLR = {
                    0: "track",
                    1: "30track",
                    2: "60track",
                    3: "-30track",
                    4: "-60track",
                    5: "L60crank",
                    6: "R60crank",
                    7: "snake",
                    8: "splitS",
                    9: "3",
                    10: "9",
                    11: "fastTurn",
                    12: "-30turn",
                    13: "-60turn",
                }

class ChooseStrategyEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.key_order_1v1 = [
            "target_alive", # 1
            "target_observable", # 1
            "target_locked", # 1
            "missile_in_mid_term",  # 1
            "locked_by_target",  # 1
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "weapon", # 1
            "threat",  # 4
            "border",  # 2
        ]
        self.obs_dim = 1*6+8+7+1+4+2
        self.fly_act_dim = [14]
        self.fire_dim = 1
        
        # [新增] 初始化 last_obs 属性，用于记录上一帧状态以计算瞬时奖励
        self.last_obs = None

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=0, ego_side='b'):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo, ego_side=ego_side)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        
        # [新增] 初始化 last_dead 属性，防止死亡惩罚重复计算
        self.RUAV.last_dead = False
        self.BUAV.last_dead = False

        # [确认存在/修改] 确保每个 Episode 开始时重置 last_obs
        self.last_obs = None 
        
        self.pomdp = pomdp   
    
    def obs_1v1(self, side, pomdp=0, reward_fn=0):
        pre_full_obs = self.base_obs(side, pomdp, reward_fn)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.key_order_1v1}
        
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order_1v1)
        return flat_obs, full_obs
    
    def obs2obs_check(self, obs):
        """
        输入: (Dim,) 或 (Batch, Dim)
        输出: 
            - 单样本: 返回 1 个 dict
            - Batch: 返回 [dict, dict, ...] (List of Dicts)
        这样可以直接丢进 for 循环里处理
        """
        # 转 numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().detach().numpy()
            
        # 定义维度映射
        key_dims = {
            "target_alive": 1, "target_observable": 1, "target_locked": 1,
            "missile_in_mid_term": 1, "locked_by_target": 1, "warning": 1,
            "target_information": 8, "ego_main": 7, "weapon": 1,
            "threat": 4, "border": 2,
        }
        
        # 判断是否为 Batch
        is_batch = (obs.ndim > 1)
        batch_size = obs.shape[0] if is_batch else 1
        
        # 预先切分好所有数据
        # sliced_data 结构: {key: array_values}
        sliced_data = {}
        ptr = 0
        for key in self.key_order_1v1:
            dim = key_dims.get(key, 0)
            if dim == 0: continue
            
            if is_batch:
                val = obs[:, ptr : ptr + dim] # (B, dim)
            else:
                val = obs[ptr : ptr + dim]    # (dim,)
            
            sliced_data[key] = val
            ptr += dim

        # --- 核心修改：构建输出 ---
        
        if not is_batch:
            # === 单样本模式 (返回 Dict) ===
            single_dict = {}
            for key, val in sliced_data.items():
                if key_dims[key] == 1:
                    single_dict[key] = val[0] # 标量化
                else:
                    single_dict[key] = val    # 保持一维数组
            return single_dict
            
        else:
            # === Batch 模式 (返回 List[Dict]) ===
            list_of_dicts = []
            for i in range(batch_size):
                sample_dict = {}
                for key, val in sliced_data.items():
                    # val 是 (B, dim)
                    sample_val = val[i] # 取第 i 行 -> (dim,)
                    
                    if key_dims[key] == 1:
                        sample_dict[key] = sample_val[0] # 标量化
                    else:
                        sample_dict[key] = sample_val    # 保持一维数组
                list_of_dicts.append(sample_dict)
            return list_of_dicts
        
    def obs2obs_check(self, obs):
        """
        将扁平化的 obs (numpy array) 还原为 check_obs (dict)。
        该 check_obs 处于 scale 后的状态 (即可以直接输入 unscale_state)。
        
        Args:
            obs: (Dim, ) 或 (Batch, Dim) 的 numpy 数组或 tensor
        
        Returns:
            check_obs: 字典形式的状态
        """
        # 1. 确保 obs 是 numpy 数组
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().detach().numpy()
            
        # 2. 定义各字段的维度 (必须与 one_side_states 里的定义完全一致)
        # 注意：这里仅包含 key_order_1v1 中出现的键
        key_dims = {
            "target_alive": 1,        # bool -> 1
            "target_observable": 1,   # int -> 1
            "target_locked": 1,       # bool -> 1
            "missile_in_mid_term": 1, # bool -> 1
            "locked_by_target": 1,    # bool -> 1
            "warning": 1,             # bool -> 1
            "target_information": 8,  # np.array length 8
            "ego_main": 7,            # np.array length 7
            "weapon": 1,              # float -> 1
            "threat": 4,              # np.array length 4
            "border": 2,              # np.array length 2
        }
        
        # 3. 确定是否为 Batch 模式
        # 假设 obs 维度是 (28,) 或 (B, 28)
        # 如果是 (28,), ndim=1; 如果是 (B, 28), ndim=2
        is_batch = (obs.ndim > 1)
        
        obs_dict = {}
        ptr = 0
        
        # 4. 按顺序切分重构
        for key in self.key_order_1v1:
            if key not in key_dims:
                # 理论上 key_order_1v1 里的 key 都应该有定义，防御性编程
                continue
                
            dim = key_dims[key]
            
            # 切片
            if is_batch:
                val = obs[:, ptr : ptr + dim]
            else:
                val = obs[ptr : ptr + dim]
            
            ptr += dim
            
            # 5. 格式还原 (Scalar vs Array)
            # 这里的逻辑是为了让 unscale_state 能正常工作
            # unscale_state 中:
            #   s["weapon"] *= 120 (标量或数组均可)
            #   s["target_information"][3] *= ... (必须是数组/可索引对象)
            
            # 对于维度为 1 的标量字段 (如 weapon, target_alive)
            # 如果是单样本模式，还原为标量 (float/bool)
            # 如果是 Batch 模式，保持 (B, 1) 或 Flatten 为 (B,) 取决于后续需求，
            # 通常保持 (B, 1) 对矩阵运算更安全，但这里为了模仿 one_side_states 的原始结构 (scalar)，
            # 单样本时取 val[0]。
            
            if dim == 1:
                if not is_batch:
                    val = val[0] # 还原为标量
                # Batch 模式下通常保留 (B, 1) 维度以便后续处理
            
            obs_dict[key] = val

        # 注意：ego_control 不在 key_order_1v1 中，因此无法恢复。
        # unscale_state 函数中有 `if "ego_control" in s` 的检查，所以这是安全的。
            
        return obs_dict
    
    # 旧动作空间（无左右分别）
    def maneuver14(self, UAV, action):
        # 输入动作与动力运动学状态
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])

        move_action = np.zeros(3)

        # 水平跟踪
        if action == 0:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 0
            speed_cmd = 400

        # 30°爬升加速
        if action == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 2500
            speed_cmd = 400

        # 60°爬升加速
        if action == 2:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000
            speed_cmd = 400

        # -30°俯冲跟踪
        if action == 3:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # -60°俯冲跟踪
        if action == 4:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400

        # ±30°水平偏移
        if action == 5:
            delta_psi_cmd = sub_of_radian(delta_psi - np.sign(delta_psi)*pi/6, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # ±60°水平偏移
        if action == 6:
            delta_psi_cmd = sub_of_radian(delta_psi - np.sign(delta_psi) * 55 * pi/180, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 水平蛇形机动
        if action == 7:
            if delta_psi > 50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            if delta_psi < -50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            elif UAV.phi>=0:
                delta_psi_cmd = sub_of_radian(delta_psi+pi/3, 0)
            else:
                delta_psi_cmd = sub_of_radian(delta_psi-pi/3, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 破s
        if action == 8:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi, pi)
            delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
            speed_cmd = 300

        # 水平三九线机动
        if action == 9:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp - np.sign(delta_psi)*pi/2, 0)
            delta_height_cmd = 0
            speed_cmd = 400

        # 水平慢置尾
        if action == 10:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = -np.sign(delta_psi)*np.clip((1-abs(delta_psi_temp)/pi) * 2, 0, 1) * 10*pi/180
            delta_height_cmd = 0
            speed_cmd = 600

        # 水平快置尾
        if action == 11:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
            speed_cmd = 400

        # 水平快置尾后-30°俯冲
        if action == 12:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # 水平快置尾后-60°俯冲
        if action == 13:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])
    
    # 新动作空间（区分左右）
    def maneuver14LR(self, UAV, action):        
        # 输入动作与动力运动学状态
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])

        move_action = np.zeros(3)

        # 水平跟踪
        if action == 0:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 0
            speed_cmd = 400

        # 30°爬升加速
        if action == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 2500
            speed_cmd = 400

        # 60°爬升加速
        if action == 2:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000
            speed_cmd = 400

        # -30°俯冲跟踪
        if action == 3:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # -60°俯冲跟踪
        if action == 4:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400

        # 左60°水平偏移
        if action == 5:
            delta_psi_cmd = sub_of_radian(delta_psi - 55 * pi/180, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 右60°水平偏移
        if action == 6:
            delta_psi_cmd = sub_of_radian(delta_psi + 55 * pi/180, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 水平蛇形机动
        if action == 7:
            if delta_psi > 50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            if delta_psi < -50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            elif UAV.phi>=0:
                delta_psi_cmd = sub_of_radian(delta_psi+pi/3, 0)
            else:
                delta_psi_cmd = sub_of_radian(delta_psi-pi/3, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 破s
        if action == 8:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi, pi)
            delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
            speed_cmd = 300

        # 水平3线机动
        if action == 9:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
            delta_height_cmd = 0
            speed_cmd = 400

        # 水平9线机动
        if action == 10:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
            delta_height_cmd = 0
            speed_cmd = 400

        # 水平快置尾
        if action == 11:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
            speed_cmd = 400

        # 水平快置尾后-30°俯冲
        if action == 12:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # 水平快置尾后-60°俯冲
        if action == 13:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])
    

    def combat_terminate_and_reward(self, side, action_label, action_shoot, action_cycle_multiplier=30):
        # [重构] 将辅助奖励的缩放系数集中到此字典中，所有值为正数
        reward_weights = {
            'base_survival': 0.0,
            'missile_guidance': 0.04,
            'target_locked': 0.06,
            'locked_by_target': 0.05,
            'missile_warning': 0.06,
            'enemy_gets_warning': 0.05,
            'alt_limit_penalty': 1.0,
            'border_penalty_scale': 0.2,
            'border_reward': 1.0,
            'angle_advantage': 1.0,
            'height_advantage': 0.1,
            'defensive_angle_close': 0.5,
            'defensive_run_close': 0.5,
            'defensive_angle_far': 0.2,
            'defensive_crank_penalty': 0.3,
            'aoa_penalty': 0.02,
            'pitch_penalty': 0.02,
        }

        self.close_range_kill() # 允许跑刀
        self.update_missile_state()
        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            alive_enm_missiles = self.alive_b_missiles
            alive_ally_missiles = self.alive_r_missiles
        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_ally_missiles = self.alive_b_missiles

        # 如果对手和对手的所有导弹都没了，且己方飞机还在，判定为胜
        if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
            self.win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方飞机还在，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            self.lose = 1
            done = 1
        # 如果友方和敌方打光导弹且都存活，或双方飞机都没了，判定为平
        elif ego.ammo == 0 and enm.ammo == 0 and not ego.dead and not enm.dead or \
                ego.dead and enm.dead:
            self.draw = 1
            done = 1
        else:
            done = 0

        if self.t > self.game_time_limit:
            done = 1
            # 如果超时，我方打光导弹，导弹全自爆，对手导弹还有剩，且存活，判定为负
            if ego.ammo + len(alive_ally_missiles) == 0 and \
                enm.ammo + len(alive_enm_missiles) > 0 and not enm.dead:
                self.lose = 1

            # 如果超时，对手打光导弹，导弹全自爆，我方导弹还有剩，且存活，判定为胜
            elif enm.ammo + len(alive_enm_missiles) == 0 and \
                ego.ammo + len(alive_ally_missiles) > 0 and not ego.dead:
                self.win = 1                

            # 如果超时，双方均未打光导弹/仍有导弹在空中飞，且双方均存活, 或者双方都死，判定为平
            else:
                self.draw = 1

        # ego_states = self.get_state(side)
        # enm_states = self.get_state(enm.side)
        ego_states = ego.current_state
        enm_states = enm.current_state
        
        delta_theta = ego_states["target_information"][2]
        distance = ego_states["target_information"][3]
        speed = ego_states["ego_main"][0]
        alt = ego_states["ego_main"][1]
        cos_delta_psi = ego_states["target_information"][0]
        sin_delta_psi = ego_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = ego_states["target_information"][4]
        warning = ego_states["warning"]
        missile_in_mid_term = ego_states["missile_in_mid_term"]
        missile_time_since_shoot = ego_states["weapon"]
        AA_hor = ego_states["target_information"][-2]
        
        cos_delta_psi_threat = ego_states["threat"][0]
        sin_delta_psi_threat = ego_states["threat"][1]
        threat_distance = ego_states["threat"][3]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)
        delta_theta_threat = ego_states["threat"][2]

        d_hor = ego_states["border"][0]
        # [新增] 计算对手视角下的ATA，用于判定严格的“被锁定”条件
        # 注意：这里计算的是 enm 的速度矢量 与 enm指向ego的连线 之间的夹角
        L_enm2ego = ego.pos_ - enm.pos_
        dist_enm2ego = norm(L_enm2ego)
        cos_ATA_enm = np.dot(enm.vel_, L_enm2ego) / (norm(enm.vel_) * dist_enm2ego + 0.001)
        ATA_enm = np.arccos(np.clip(cos_ATA_enm, -1, 1))
        # [新增] 严格的被锁定判定：系统判定被锁 + 距离限制 + 角度限制
        strict_locked_by_target = ego_states["locked_by_target"] and (dist_enm2ego <= 80e3) and (ATA_enm <= pi/3)
        # [新增] 初始化或获取上一时刻的状态字典
        if self.last_obs is None:
            self.last_obs = {'r': {}, 'b': {}}
        last_info = self.last_obs.get(side, {})
        # 获取上一时刻的关键状态，默认为0 (False)
        last_target_locked = last_info.get("target_locked", 0)
        last_strict_locked_by_target = last_info.get("strict_locked_by_target", 0)
        last_warning = last_info.get("warning", 0)
        last_enm_warning = last_info.get("enm_warning", 0)

        ego_last_state = ego.last_state
        last_threat_distance = ego_last_state["threat"][3]
        
        reward_event = 0  # 结果奖励
        reward_assisted = reward_weights['base_survival']  # 辅助奖励
        
        wasted = 0
        # 发射导弹
        shoot = action_shoot
        # 如果死了，就把剩余导弹损耗的惩罚一并加上，禁止自杀套利
        
        # [修改] 引入 last_dead 判定，确保死亡惩罚只扣一次
        is_dead_now = ego.dead or self.out_range(ego)
        if is_dead_now and not getattr(ego, 'last_dead', False):
            shoot = ego.ammo
            wasted = ego.ammo
            ego.last_dead = True
        elif is_dead_now and getattr(ego, 'last_dead', False):
            # 已经死过一次了，不再重复扣除剩余导弹的惩罚
            shoot = 0
            wasted = 0
            
        

        # [修改] 为导弹提供制导
        if missile_in_mid_term:
            reward_assisted += reward_weights['missile_guidance']

        # [修改] 锁定目标
        if ego_states["target_locked"]:
            reward_assisted += reward_weights['target_locked']

        # [修改] 被目标锁定
        if strict_locked_by_target:
            reward_assisted -= reward_weights['locked_by_target']

        # [修改] 收到导弹警告
        if warning:
            reward_assisted -= reward_weights['missile_warning']

        # [修改] 导弹锁定目标 (对手收到警告)
        if enm_states["warning"]:
            reward_assisted += reward_weights['enemy_gets_warning']

        # 逃脱导弹 (硬编码)
        if ego.escape_once:
            reward_event += 20

        # 导弹被逃脱 (硬编码)
        escape_penalty = 20
        if enm.escape_once:
            reward_event -= escape_penalty

        # 死了也当剩下导弹全被逃脱处理
        if wasted>0:
            reward_event -= escape_penalty*wasted
        
        # # 发射惩罚 (硬编码)
        should_fire_missile = False
        if distance < 60e3 and alpha < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
            if missile_time_since_shoot >= 20 and not missile_in_mid_term and not (distance>12e3 and abs(AA_hor) < 30*pi/180):
                should_fire_missile = True
        
        reward_shoot = 0
        if shoot == 1:
            if should_fire_missile:
                reward_shoot += 10
            else:
                reward_shoot -= 10
        if shoot == 0:
            if should_fire_missile:
                reward_shoot -= 10
            else:
                reward_shoot += 0.01
        reward_assisted += reward_shoot

        if should_fire_missile==1:
            nobody = 1
        if should_fire_missile==0 and shoot==1:
            nobody = 2

        # 高度限制奖励/惩罚
        reward_assisted += ((alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * reward_weights['alt_limit_penalty']

        # 靠近边界惩罚
        o002ego_ = np.array([ego.pos_[0], ego.pos_[2]]) # 北，东
        ego_vh_ = np.array([ego.vel_[0], ego.vel_[2]])
        if d_hor<=50e3:
            reward_assisted -= (1-d_hor/50e3) * np.dot(ego_vh_, o002ego_)/norm(o002ego_ + 1e-3)/340 * reward_weights['border_penalty_scale']
        else:
            reward_assisted += reward_weights['border_reward']
        
        # if d_hor < 20e3:
        #     reward_assisted += (d_hor/20e3-1) * 2

        # [新增] 在计算完奖励后，更新上一时刻的状态记录，供下一步使用
        self.last_obs[side] = {
            "target_locked": float(ego_states["target_locked"]),
            "strict_locked_by_target": float(strict_locked_by_target),
            "warning": float(warning),
            "enm_warning": float(enm_states["warning"])
        }

        # 没发射导弹时alpha越小越好
        if len(alive_ally_missiles) == 0 and ego.ammo > 0 and not warning:
            # 角度优势度
            reward_assisted += (ATA_enm / pi - alpha / pi) * reward_weights['angle_advantage']
            # 高度优势度
            reward_assisted += (ego.alt-enm.alt)/5000 * reward_weights['height_advantage']
        
        
        # # 有warning时alpha越大越好
        if warning:
            threat_directio_n = np.array([cos(delta_theta_threat)*cos(delta_psi_threat), 
                                         sin(delta_theta_threat), 
                                         cos(delta_theta_threat)*sin(delta_psi_threat)])
            if threat_distance<=30e3:
                reward_assisted += reward_weights['defensive_angle_close'] * abs(delta_psi_threat) / pi
                reward_assisted += np.dot(ego.vel_,threat_directio_n)/340 * reward_weights['defensive_run_close']
            else:
                reward_assisted += reward_weights['defensive_angle_far'] * abs(delta_psi_threat) / pi
                if missile_in_mid_term:
                    reward_assisted -= reward_weights['defensive_crank_penalty'] * abs(abs(delta_psi)-pi/3)/(pi/3) # alpha-pi/3
            

        # 迎角惩罚
        reward_assisted -= reward_weights['aoa_penalty'] * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))
        reward_assisted -= reward_weights['pitch_penalty'] * (abs(ego.theta)/pi*2)  # 俯仰角惩罚
        

        if done:
            print('回合结束')
            if self.out_range(ego):
                print('出界')
            
        # if self.out_range(ego) or self.out_range(enm):
        #     print()
        
        # 补齐自杀惩罚
        time_left = self.game_time_limit-self.t
        # 没走完的步数
        steps_left = time_left / (action_cycle_multiplier * self.dt_maneuver/0.2)

        # 计算所有 shaping 权重之和（用于步数缩放）
        total_shaping_weight = float(sum(reward_weights.values()))

        # 回合结束奖励 (硬编码)
        if self.win:
            reward_event += 100 + steps_left * total_shaping_weight + ego.ammo * 20 # 增加节省弹药的奖励
        if self.lose:
            reward_event += -100 - steps_left * total_shaping_weight
            if self.out_range(ego) or ego.alt < self.min_alt:
                reward_event += -50
        if self.draw:
            reward_event += -50

        return done, reward_event+reward_assisted, reward_assisted

    # 重写近距杀方法（加了print）
    def close_range_kill(self,):
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
                        print('近距双杀')
                    # 单杀
                    if cos_ATA_r >= cos(pi / 3) and cos_ATA_b < cos(pi / 3):
                        buav.dead = True
                        buav.got_hit = True
                        print('近距单杀')
                    if cos_ATA_r < cos(pi / 3) and cos_ATA_b >= cos(pi / 3):
                        ruav.dead = True
                        ruav.got_hit = True
                        print('近距单杀')
