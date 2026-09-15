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
from Math_calculates.coord_rotations import RodRot

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
    def __init__(self, args, tacview_show=0, vertices=None):
        super().__init__(args, tacview_show, vertices=vertices)
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
            "out_locked", # 1
        ]
        self.obs_dim = self.full_obs_dim - 7 # 砍去ego_control
        self.fly_act_dim = [5, 7] # [5,7] 14
        self.fly_act_dim_circ = [5, 6]
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
        self.middle_hold_score = 0

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
            "out_locked": 1,          # float -> 1
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


    def maneuverContinuous(self, UAV, action):
        # 输入动作与动力运动学状态, action直接给弧度
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        theta = asin(uav_obs["ego_main"][3])
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])

        move_action = np.zeros(3)

        # 动作空间正交化
        action_v, action_h = action

        # 速度指令，锁定340
        speed_cmd = 340

        # 垂直
        theta_desired = np.clip(theta+delta_theta + 
            np.clip(action_v, np.radians(-75), np.radians(45)), 
            np.radians(-75), np.radians(60))

        # 不能出安全高度范围
        delta_height_cmd = np.clip(theta_desired/pi*2*5000, 
                                   self.min_alt_safe-UAV.alt, 
                                   self.max_alt_safe-UAV.alt)

        # 水平
        delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
        delta_psi_cmd = sub_of_radian(delta_psi_temp + action_h, 0)

        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])


    # 新动作空间（区分左右）
    def maneuver14LR(self, UAV, action, type='primitive'):        
        # 输入动作与动力运动学状态
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        theta = asin(uav_obs["ego_main"][3])
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])
        RWR = uav_obs["warning"]

        move_action = np.zeros(3)

        # 动作空间正交化
        action_v, action_h = action

        # # 针对环形动作分布改键位
        # if type == "circular":
        #     mapping = {
        #         0: 0, 
        #         1: 1, 
        #         2: 3, 
        #         3: 5, 
        #         4: 4, 
        #         5: 2}
        #     action_h = mapping.get(int(action_h), action_h)

        # 速度指令，锁定340
        speed_cmd = 340

        # 垂直方向指令
        if action_v == 0: # 比目标高45°
            theta_desired = np.clip(theta+delta_theta+np.radians(45), -pi/2, pi/2)
        if action_v == 1: # 比目标高20°
            theta_desired = np.clip(theta+delta_theta+np.radians(20), -pi/2, pi/2)
        if action_v == 2: # 纯追踪
            theta_desired = np.clip(theta+delta_theta+np.radians(1), -pi/2, pi/2)
        if action_v == 3: # 比目标低30°
            theta_desired = np.clip(theta+delta_theta+np.radians(-30), -pi/2, pi/2)
        if action_v == 4: # 急速下降
            theta_desired = -np.radians(85)  # -pi/2
        
        # 水平方向指令：
        # 追踪
        if action_h == 0:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
        # 左crank
        if action_h == 1:
            delta_psi_cmd = sub_of_radian(delta_psi - 61 * pi/180, 0)
            theta_desired = np.clip(theta_desired, -pi/4, pi/4) # 防止无效crank
        # 右crank
        if action_h == 5:
            delta_psi_cmd = sub_of_radian(delta_psi + 61 * pi/180, 0)
            theta_desired = np.clip(theta_desired, -pi/4, pi/4) # 防止无效crank
        # 3线
        if action_h == 2:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
        # 9线
        if action_h == 4:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
        # 置尾机动
        if action_h == 3:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)

        # 回中心(连续动作空间离散化禁止使用)
        if action_h == 6:
            line2center = np.array([self.horizontal_center[0]-UAV.pos_[0], self.horizontal_center[1]-UAV.pos_[2]])
            psi2center = np.arctan2(line2center[1], line2center[0])
            dist2center = norm(line2center)
            delta_psi_cmd = sub_of_radian(psi2center, UAV.psi) * dist2center/self.R_cage

        
        # # 不能出安全高度范围
        # delta_height_cmd = np.clip(theta_desired/pi*2*5000,
        #                            self.min_alt_safe-UAV.alt,
        #                            self.max_alt_safe-UAV.alt)
        
        # 处理指令，让crank的时候依然能够保持锁定
        # theta_desired = delta_height_cmd /5000*pi/2
        
        target_delta_point_ = np.array([
            cos(theta + delta_theta) * cos(delta_psi),
            sin(theta + delta_theta),
            cos(theta + delta_theta) * sin(delta_psi)
        ])
        desired_point_ = np.array([
            cos(theta_desired) * cos(delta_psi_cmd),
            sin(theta_desired),
            cos(theta_desired) * sin(delta_psi_cmd)
        ])
        
        ATA_estimated = np.arccos(np.dot(desired_point_, target_delta_point_)*0.999)
        # crank
        if action_h in[1,5]:
            crank_angle = 53 # 59 # 偏置机动角度
            if ATA_estimated > np.radians(crank_angle):
                axis_ = np.cross(target_delta_point_, desired_point_)
                axis_ = axis_ / (norm(axis_) + 1e-6)
                target_delta_point_ = RodRot(target_delta_point_, axis_, np.radians(crank_angle))
                theta_desired = np.arcsin(target_delta_point_[1])
                delta_psi_cmd = np.arctan2(target_delta_point_[2], target_delta_point_[0])
        
        # # 39机动 有bug，屏蔽处理
        # if action_h in[2,4]:
        #     if RWR:
        #         crank_angle = 90
        #     else: # 未收到告警保持锁定
        #         crank_angle = 90 # 59 # 偏置机动角度
        #     if ATA_estimated != np.radians(crank_angle):
        #         axis_ = np.cross(target_delta_point_, desired_point_)
        #         axis_ = axis_ / (norm(axis_) + 1e-6)
        #         target_delta_point_ = RodRot(target_delta_point_, axis_, np.radians(crank_angle))
        #         theta_desired = np.arcsin(target_delta_point_[1])
        #         delta_psi_cmd = np.arctan2(target_delta_point_[2], target_delta_point_[0])
        
        # 不能出安全高度范围
        delta_height_cmd = np.clip(theta_desired/pi*2*5000,
                                   self.min_alt_safe-UAV.alt,
                                   self.max_alt_safe-UAV.alt)
        
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])

    # 重写近距杀方法（加了print）
    def close_range_kill(self,):
        WVR = 0
        for RUAV in self.RUAVs:
            if RUAV.dead:
                continue
            for BUAV in self.BUAVs:
                if BUAV.dead:
                    continue
                elif norm(RUAV.pos_ - BUAV.pos_) >= 8e3:
                    continue
                else:
                    Lbr_ = RUAV.pos_ - BUAV.pos_
                    Lrb_ = BUAV.pos_ - RUAV.pos_
                    dist = norm(Lbr_)
                    # 求解hot-cold关系
                    cos_ATA_r = np.dot(Lrb_, RUAV.vel_) / (dist * RUAV.speed)
                    cos_ATA_b = np.dot(Lbr_, BUAV.vel_) / (dist * BUAV.speed)
                    # 角度优势杀
                    if cos_ATA_r >= cos(pi / 6) and cos_ATA_b < cos(pi / 6):
                        BUAV.dead = True
                        print('近距单杀')
                        WVR = 1
                    elif cos_ATA_r < cos(pi / 6) and cos_ATA_b >= cos(pi / 6):
                        RUAV.dead = True
                        print('近距单杀')
                        WVR = 1
                    # 都在可攻击角度
                    elif cos_ATA_r >= cos(pi / 6) and cos_ATA_b >= cos(pi / 6):
                        RUAV.dead = True
                        BUAV.dead = True
                        print('近距双杀')
                        WVR = 1

                    # 更复杂，但不一定好用的判定逻辑
                    #     # 看高度
                    #     if BUAV.alt - RUAV.alt > 1500:
                    #         # 低于对面，近距处于劣势
                    #         RUAV.dead = True
                    #         RUAV.got_hit = True
                    #         print('近距单杀')
                    #     elif RUAV.alt - BUAV.alt > 1500:
                    #         # 高于对面，近距处于优势
                    #         BUAV.dead = True
                    #         BUAV.got_hit = True
                    #         print('近距单杀')
                    #     else:
                    #         # 速度落后80m/s
                    #         if BUAV.speed - RUAV.speed > 80:
                    #             RUAV.dead = True
                    #             RUAV.got_hit = True
                    #             print('近距单杀')
                    #         elif RUAV.speed - BUAV.speed > 80:
                    #             BUAV.dead = True
                    #             BUAV.got_hit = True
                    #             print('近距单杀')
                    #         else:
                    #             RUAV.dead = True
                    #             BUAV.dead = True
                    #             RUAV.got_hit = True
                    #             BUAV.got_hit = True
                    #             print('近距双杀')
                        
                    else: # 都不在可攻击角度
                        pass  # 无法杀
        return WVR


# # 水平跟踪
# if action == 0:
#     delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
#     delta_height_cmd = 135
#     speed_cmd = 340

# # 30°爬升加速
# if action == 1:
#     delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
#     delta_height_cmd = 5000/3
#     speed_cmd = 340

# # 60°爬升加速
# if action == 2:
#     delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
#     delta_height_cmd = 5000*2/3
#     speed_cmd = 340

# # -30°俯冲跟踪
# if action == 3:
#     delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
#     delta_height_cmd = -5000/3
#     speed_cmd = 340

# # -60°俯冲跟踪
# if action == 4:
#     delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
#     delta_height_cmd = -5000/3*2
#     speed_cmd = 340

# # 左60°水平偏移
# if action == 5:
#     delta_psi_cmd = sub_of_radian(delta_psi - 61 * pi/180, 0)
#     delta_height_cmd = 0
#     speed_cmd = 340

# # 右60°水平偏移
# if action == 6:
#     delta_psi_cmd = sub_of_radian(delta_psi + 61 * pi/180, 0)
#     delta_height_cmd = 0
#     speed_cmd = 340

# # 占领中心机动
# if action == 7:
#     line2center = np.array([self.horizontal_center[0]-UAV.pos_[0], self.horizontal_center[1]-UAV.pos_[2]])
#     psi2center = np.arctan2(line2center[1], line2center[0])
#     dist2center = norm(line2center)
#     delta_psi_cmd = sub_of_radian(psi2center, UAV.psi) * dist2center/self.R_cage
#     # 保持在安全高度
#     if UAV.alt < self.min_alt_safe:
#         delta_height_cmd = 300
#     elif UAV.alt > self.max_alt_safe:
#         delta_height_cmd = -300
#     else:
#         delta_height_cmd = 0
#     speed_cmd = 340

# # 破s
# if action == 8:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = sub_of_radian(delta_psi_temp, pi)
#     delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
#     speed_cmd = 300

# # 水平3线机动
# if action == 9:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
#     delta_height_cmd = 0
#     speed_cmd = 340

# # 水平9线机动
# if action == 10:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
#     delta_height_cmd = 0
#     speed_cmd = 340

# # 水平快置尾
# if action == 11:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
#     delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
#     speed_cmd = 340

# # 水平快置尾后-30°俯冲
# if action == 12:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
#     delta_height_cmd = -5000/3
#     speed_cmd = 340

# # 水平快置尾后-60°俯冲
# if action == 13:
#     delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
#     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
#     delta_height_cmd = -5000/3*2
#     speed_cmd = 340