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

from Envs.battle6dof1v1_missile0309_hierarchical import *


# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

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

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=1, ego_side='b'):
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

        # 开场数据支持，至少背对背决斗时需要知道“对手在后面”
        self.RUAV.state_memory = copy.deepcopy(self.get_state('r'))
        self.BUAV.state_memory = copy.deepcopy(self.get_state('b'))
    
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
            delta_height_cmd = 135
            speed_cmd = 340

        # 30°爬升加速
        if action == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000/3
            speed_cmd = 340

        # 60°爬升加速
        if action == 2:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000*2/3
            speed_cmd = 340

        # -30°俯冲跟踪
        if action == 3:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 340

        # -60°俯冲跟踪
        if action == 4:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 340

        # 左60°水平偏移
        if action == 5:
            delta_psi_cmd = sub_of_radian(delta_psi - 61 * pi/180, 0)
            delta_height_cmd = 0
            speed_cmd = 340

        # 右60°水平偏移
        if action == 6:
            delta_psi_cmd = sub_of_radian(delta_psi + 61 * pi/180, 0)
            delta_height_cmd = 0
            speed_cmd = 340

        # 占领中心机动
        if action == 7:
            line2center = np.array([self.horizontal_center[0]-UAV.pos_[0], self.horizontal_center[1]-UAV.pos_[2]])
            psi2center = np.arctan2(line2center[1], line2center[0])
            dist2center = norm(line2center)
            delta_psi_cmd = sub_of_radian(psi2center, UAV.psi) * dist2center/self.R_cage
            # 保持在安全高度
            if UAV.alt < self.min_alt_safe:
                delta_height_cmd = 300
            elif UAV.alt > self.max_alt_safe:
                delta_height_cmd = -300
            else:
                delta_height_cmd = 0
            speed_cmd = 340

        # 破s
        if action == 8:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp, pi)
            delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
            speed_cmd = 300

        # 水平3线机动
        if action == 9:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
            delta_height_cmd = 0
            speed_cmd = 340

        # 水平9线机动
        if action == 10:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
            delta_height_cmd = 0
            speed_cmd = 340

        # 水平快置尾
        if action == 11:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
            delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
            speed_cmd = 340

        # 水平快置尾后-30°俯冲
        if action == 12:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 340

        # 水平快置尾后-60°俯冲
        if action == 13:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 340
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])

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
