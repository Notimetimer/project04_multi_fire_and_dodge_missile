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

from Envs.BattleHomo260421 import *


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
                    7: "Conquer",
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
        # self.key_order_1v1 = [
        #     "target_alive", # 1
        #     "target_observable", # 1
        #     "target_locked", # 1
        #     "missile_in_mid_term",  # 1
        #     "locked_by_target",  # 1
        #     "warning",  # 1
        #     "target_information",  # 8
        #     "ego_main",  # 7
        #     "weapon", # 1
        #     "threat",  # 4
        #     "border",  # 2
        # ]
        # self.obs_dim = 1*6+8+7+1+4+2
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
        for UAV in self.UAVsTable.values():
            UAV.last_dead = False

        # [确认存在/修改] 确保每个 Episode 开始时重置 last_obs
        self.last_obs = None 
        
        self.pomdp = pomdp   

        # 开场数据支持，至少背对背决斗时需要知道“对手在后面”
        for id, uav in self.UAVsTable.items():
            uav.state_memory = copy.deepcopy(self.get_state(id))
    
    """
    需要区分目标死活、填充未发射的导弹信息和死掉的敌机/队友信息，
    1、简单起见，告警信息机械筛选距离最近的，武器信息也用机械筛选时间最近的
    2、只有敌机和队友用变长度序列交给两组注意力机制处理
    3、输出不再是一维数组，仍然是字典包list
    4、get_obs 的输出仍然是字典，但不使用base_obs中的一长串keys，而是适配神经网络结构。
    keys:  fixed, ally[list], enm[list], ally_mask(定长度序列), enm_mask(定长度序列)
    1) 观测变量可以给出空列表[]，填充Padding交给算法端的包装器去做
    2) 掩码必须显式给出，交给注意力机制去忽略
    3) 算法包装器需要null_token,不在环境端做
    4) 两个mask也必须存入经验池。
    """
    def get_obs(self, id, pomdp=0, reward_fn=0):
        full_obs = self.base_obs(id, pomdp, reward_fn)
        full_state = self.unscale_state(full_obs)

        ego = self.find_uav_with_id(id)
        side = ego.side
        
        ego_side_num = self.Rnum if side =='r' else self.Bnum
        enm_side_num = self.Bnum if side =='r' else self.Rnum

        mask_ally = np.ones(ego_side_num-1)
        mask_enm = np.ones(enm_side_num)
        
        information_output = {'fixed':[], 'ally':[], 'enm':[]}
        mask_output = {'ally':None, 'enm':None}

        # 只缓存一次 scaled_init_mod 以兼顾性能和合法性（组装为dict进行缩放）
        if not hasattr(self, 'scaled_init_mod'):
            self.scaled_init_mod = {}
            for key, array_key in zip(
                ["enm_info", "ally_info", "weapon_info", "RWR_info"],
                ["enm_infos", "ally_infos", "weapon_infos", "RWR_infos"]
            ):
                temp_dict = {array_key: np.array([self.state_init_mod[key]])}
                self.scaled_init_mod[key] = self.scale_state(temp_dict)[array_key][0]

        # 处理敌人，对死掉的敌人写mask
        for i, enm in enumerate(full_state["enm_infos"]):
            enm_alive = enm[0]
            # 死掉的敌人，加mask和填充
            if not enm_alive:
                mask_enm[i] = 0
                full_obs["enm_infos"][i] = self.scaled_init_mod["enm_info"]

        # 处理队友，对死掉的队友写mask，
        for i, ally in enumerate(full_state["ally_infos"]):
            ally_alive = ally[0]
            # 死掉的队友，加mask和填充
            if not ally_alive:
                mask_ally[i] = 0
                full_obs["ally_infos"][i] = self.scaled_init_mod["ally_info"]

        # 处理导弹（残留处理逻辑，手工注意力）
        if len(full_obs["weapon_infos"]) == 0: # 空数组
            full_obs["weapon_infos"] = self.scaled_init_mod["weapon_info"]
        else:
            # 提取最近的发射
            min_launch_time = 120
            closest_weapon_info = None
            for i, weapon_info in enumerate(full_state["weapon_infos"]):
                launch_time = weapon_info[0]
                if launch_time < min_launch_time:
                    min_launch_time = launch_time
                    closest_weapon_info = weapon_info
            
            if closest_weapon_info is not None:
                full_obs["weapon_infos"] = self.scale_state({"weapon_infos": np.array([closest_weapon_info])})["weapon_infos"][0]
            else:
                full_obs["weapon_infos"] = self.scaled_init_mod["weapon_info"]

        # 处理威胁
        if len(full_obs["RWR_infos"]) == 0:
            full_obs["RWR_infos"] = self.scaled_init_mod["RWR_info"]
        else:
            # 提取最近的威胁
            min_dist = 100e3
            closest_RWR_info = None
            for i, RWR_info in enumerate(full_state["RWR_infos"]):
                dist = RWR_info[4]
                if dist < min_dist:
                    min_dist = dist
                    closest_RWR_info = RWR_info
            
            if closest_RWR_info is not None:
                full_obs["RWR_infos"] = self.scale_state({"RWR_infos": np.array([closest_RWR_info])})["RWR_infos"][0]
            else:
                full_obs["RWR_infos"] = self.scaled_init_mod["RWR_info"]

        ally_array = full_obs["ally_infos"]
        enm_array  = full_obs["enm_infos"]
        
        # 拼接fixed
        # 1. 简单做法
        # fixed_order = self.key_order.copy()
        # fixed_order.remove("ally_infos") # 直接调用，不管它的返回值
        # fixed_order.remove("enm_infos")
        # 2. 列表推导式
        fixed_order = [k for k in self.key_order if k not in ["ally_infos", "enm_infos"]]
        information_output['fixed'] = flatten_obs(full_obs, fixed_order)
        information_output['ally'] = ally_array
        information_output['enm'] = enm_array
        mask_output['ally'] = mask_ally
        mask_output['enm'] = mask_enm
        
        return information_output, mask_output, self.unscale_processed_obs(full_obs)
    
    # 不做策略蒸馏，删除反扁平化处理
    
    # 被压缩的观测信息反缩放需要适配原先unscale的格式
    def unscale_processed_obs(self, obs_input):

        s = copy.deepcopy(obs_input)
        # 封装为 2D 数组以适配 scale/unscale_state 中的 [:, idx] 语法
        s["weapon_infos"] = np.array([s["weapon_infos"]])
        s["RWR_infos"] = np.array([s["RWR_infos"]])

        s1 = self.unscale_state(s)

        s1["weapon_infos"]=s1["weapon_infos"][0]
        s1["RWR_infos"]=s1["RWR_infos"][0]
        return s1
    
    # 雷达动作(只有开关)
    def radar_action(self, id, action):
        ego = self.find_uav_with_id(id)
        if action and (not ego.dead):
            ego.radar_on = 1
        else:
            ego.radar_on = 0
        return ego.radar_on
    
    # 开火动作,这个函数一次只处理一枚导弹的发射
    def fire_missile(self, source_id, enm_index, tabu=0):
        ego = self.find_uav_with_id(source_id)
        ego_side = ego.side
        if ego_side =='r':
            enm_side = 'b'
            ego_num = self.Rnum
            enm_num = self.Bnum
        else:
            enm_side = 'r'
            ego_num = self.Bnum
            enm_num = self.Rnum
        
        # 输入动作与动力运动学状态
        max_enm_index = enm_num + 1 # 预留多一个槽位给null_token
        if enm_index >= max_enm_index:
            can_fire = 0 # 禁止向需token开火
            return None
        else:
            can_fire = 1

        new_missile_id = None
        ego = self.find_uav_with_id(source_id)
        enm_id = self.index2id(source_id, enm_index, relation='enm')
        enm = self.find_uav_with_id(enm_id)
        ego_state = self.get_state(source_id)
        
        # tabu 自变量
        enm_locked = ego_state["enm_infos"][enm_index][3]
        ATA = ego_state["enm_infos"][enm_index][9]
        distance = ego_state["enm_infos"][enm_index][8]
        AA_hor = ego_state["enm_infos"][enm_index][11]

        # 绝对禁止开火条件
        # 1. 我死了
        case1 = ego.dead
        # 2. 没有弹药
        case2 = (ego.ammo == 0)

        # 绝对禁止开火情况
        if case1 or case2 :
            return None

        # 部分禁止开火条件
        # tabu1. 未锁定目标
        tabu_case1 = not enm_locked
        # tabu2. 目标已死
        tabu_case2 = enm.dead
        # tabu3. 冷却时间（不建议放在环境端）

        if tabu and (tabu_case1 or tabu_case2):
            can_fire = 0
        
        # 发射导弹
        if can_fire:
            # 调用唯一注册入口，严禁直接修改self.Rmissiles或self.Bmissiles
            new_missile_id = self.launch_1_missile(source_id, enm_id)
        return new_missile_id

    # 只针对一个敌机的动作空间（不考虑队友也不考虑综合情况）
    "动作空间留多一个槽位，用来接住null_token输出的index"
    def maneuver_for_1_enm(self, id, enm_index, action, full_obs_at_the_moment):
        ego = self.find_uav_with_id(id)
        ego_side = ego.side
        if ego_side =='r':
            enm_side = 'b'
            ego_num = self.Rnum
            enm_num = self.Bnum
        else:
            enm_side = 'r'
            ego_num = self.Bnum
            enm_num = self.Rnum
        
        # 动作空间也需要知道智能体“看到什么”
        # 导弹的信息和威胁的信息被压缩
        vice_uav_states = self.unscale_processed_obs(full_obs_at_the_moment)

        # 输入动作与动力运动学状态
        max_enm_index = enm_num + 1 # 预留多一个槽位给null_token
        if enm_index >= max_enm_index:
            vice_uav_states["enm_infos"] = self.state_init_mod["enm_info"]
        else:
            # 执行动作前压缩信息
            vice_uav_states["enm_infos"] = vice_uav_states["enm_infos"][enm_index]
        enm_info = vice_uav_states["enm_infos"]

        # 队友的方位没有考虑在动作空间里，暂不做压缩
        
        # 敌机信息
        cos_delta_psi = enm_info[5]
        sin_delta_psi = enm_info[6]
        delta_theta = enm_info[7]
        distance = enm_info[8]
        speed = vice_uav_states["ego_main"][0]
        alt = vice_uav_states["ego_main"][1]
        d_hor, leftright, d2center = vice_uav_states["border"]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(vice_uav_states["RWR_infos"][2], vice_uav_states["RWR_infos"][1])

        # 动作空间正交化
        action_v, action_h = action
        # 速度指令，锁定340
        speed_cmd = 340

        # 垂直方向指令
        if action_v == 0:
            delta_height_cmd = min(5000*2/3, self.max_alt_safe-ego.alt)
        if action_v == 1:
            delta_height_cmd = min(5000/3, self.max_alt_safe-ego.alt)
        if action_v == 2:
            delta_height_cmd = 135
        if action_v == 3:
            delta_height_cmd = max(-5000/3, self.min_alt_safe-ego.alt)
        if action_v == 4:
            delta_height_cmd = max(-5000/3*2, self.min_alt_safe-ego.alt)
        if action_v == 5:
            delta_height_cmd = max(-2000, self.min_alt_safe-ego.alt)

        # 水平方向指令：
        # 回中心
        if action_h == 0:
            line2center = np.array([self.horizontal_center[0]-ego.pos_[0], self.horizontal_center[1]-ego.pos_[2]])
            psi2center = np.arctan2(line2center[1], line2center[0])
            dist2center = norm(line2center)
            delta_psi_cmd = sub_of_radian(psi2center, ego.psi) * dist2center/self.R_cage
        # 追踪
        if action_h == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
        # 左crank
        if action_h == 2:
            delta_psi_cmd = sub_of_radian(delta_psi - 61 * pi/180, 0)
        # 右crank
        if action_h == 3:
            delta_psi_cmd = sub_of_radian(delta_psi + 61 * pi/180, 0)
        # 3线
        if action_h == 4:
            delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
        # 9线
        if action_h == 5:
            delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
        # 置尾机动
        if action_h == 6:
            delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)

        # # 旧的14维动作空间，现在开始正交化分解
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
        #     line2center = np.array([self.horizontal_center[0]-ego.pos_[0], self.horizontal_center[1]-ego.pos_[2]])
        #     psi2center = np.arctan2(line2center[1], line2center[0])
        #     dist2center = norm(line2center)
        #     delta_psi_cmd = sub_of_radian(psi2center, ego.psi) * dist2center/self.R_cage
        #     # 保持在安全高度
        #     if ego.alt < self.min_alt_safe:
        #         delta_height_cmd = 300
        #     elif ego.alt > self.max_alt_safe:
        #         delta_height_cmd = -300
        #     else:
        #         delta_height_cmd = 0
        #     speed_cmd = 340

        # # 破s
        # if action == 8:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = sub_of_radian(delta_psi_temp, pi)
        #     delta_height_cmd = max(-2000, self.min_alt_safe-ego.alt)
        #     speed_cmd = 300

        # # 水平3线机动
        # if action == 9:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = sub_of_radian(delta_psi_temp - pi/2, 0)
        #     delta_height_cmd = 0
        #     speed_cmd = 340

        # # 水平9线机动
        # if action == 10:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = sub_of_radian(delta_psi_temp + pi/2, 0)
        #     delta_height_cmd = 0
        #     speed_cmd = 340

        # # 水平快置尾
        # if action == 11:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
        #     delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
        #     speed_cmd = 340

        # # 水平快置尾后-30°俯冲
        # if action == 12:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
        #     delta_height_cmd = -5000/3
        #     speed_cmd = 340

        # # 水平快置尾后-60°俯冲
        # if action == 13:
        #     delta_psi_temp = delta_psi_threat if vice_uav_states["RWR_infos"][0] else delta_psi
        #     delta_psi_cmd = np.clip(sub_of_radian(delta_psi_temp, pi), -pi/2, pi/2)
        #     delta_height_cmd = -5000/3*2
        #     speed_cmd = 340

        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])


    # 综合动作空间（考虑队友和综合情况）
    def maneuver_complex(self, id, action):
        pass

    # 近距杀方法，多对多也能用（加了print）
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
