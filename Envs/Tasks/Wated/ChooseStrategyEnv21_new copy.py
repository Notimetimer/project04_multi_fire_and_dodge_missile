'''
增加开火惩罚
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
        
        # # [新增] 初始化 last_obs 属性，用于记录上一帧状态以计算瞬时奖励
        # self.last_obs = None

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=0):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        
        # [新增] 初始化 last_dead 属性，防止死亡惩罚重复计算
        self.RUAV.last_dead = False
        self.BUAV.last_dead = False

        # # [确认存在/修改] 确保每个 Episode 开始时重置 last_obs
        # self.last_obs = None 
        
        self.pomdp = pomdp   
    
    def obs_1v1(self, side, pomdp=0, reward_fn=0):
        pre_full_obs = self.base_obs(side, pomdp, reward_fn)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.key_order_1v1}
        
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order_1v1)
        return flat_obs, full_obs


    def maneuver14(self, UAV, action):
        # # debug
        # # 动作太多？那就砍掉一些
        # if 0<= action <=4:
        #     action = 0
        # if 5<=action<=7:
        #     action = 6
        # if 8<=action<=13:
        #     action = 11
        
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
            delta_psi_cmd = delta_psi - np.sign(delta_psi)*pi/6
            delta_height_cmd = 0
            speed_cmd = 350

        # ±60°水平偏移
        if action == 6:
            delta_psi_cmd = delta_psi - np.sign(delta_psi) * 55 * pi/180
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
            delta_psi_cmd = delta_psi_temp - np.sign(delta_psi)*pi/2
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
    

    def combat_terminate_and_reward(self, side, action_label, action_shoot, action_cycle_multiplier=30):
        # --- 1. 参数初始化与状态获取 ---
        # 权重在此仅作为内部计算比例，实际整体缩放由外部 lambda 控制
        weights = {
            'missile_guidance': 0.04,
            'target_locked': 0.06,
            'strict_locked_by_target': 0.05,
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

        # --- 2. 终止判定 ---
        done = 0
        
        # --简单判定法--
        # 严格回合时间限制
        if self.t > self.game_time_limit:
            done = 1
        # 双杀
        if enm.dead and ego.dead:
            done = 1
        # 如果敌方和敌方所有导弹都没了，且我方存活，判定为胜
        if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
            self.win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方存活，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            self.lose = 1
            done = 1
        # 双杀双活时间到，就是平局
        elif done: 
            self.draw = 1
            
        # # --原有判定法--
        # if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
        #     self.win = 1
        #     done = 1
        # # 如果友方和友方的所有导弹都没了，且敌方飞机还在，判定为负
        # elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
        #     self.lose = 1
        #     done = 1
            
        # # 如果友方和敌方打光导弹且都存活，或双方飞机都没了，判定为平
        # elif ego.ammo == 0 and enm.ammo == 0 and (not ego.dead) and (not enm.dead) or \
        #         (ego.dead and enm.dead):
        #     self.draw = 1
        #     done = 1
        # else:
        #     done = 0
        # if self.t > self.game_time_limit:
        #     done = 1
        #     # 如果超时，我方打光导弹，导弹全自爆，对手导弹还有剩，且存活，判定为负
        #     if ego.ammo + len(alive_ally_missiles) == 0 and \
        #         enm.ammo + len(alive_enm_missiles) > 0 and not enm.dead:
        #         self.lose = 1
        #     # 如果超时，对手打光导弹，导弹全自爆，我方导弹还有剩，且存活，判定为胜
        #     elif enm.ammo + len(alive_enm_missiles) == 0 and \
        #         ego.ammo + len(alive_ally_missiles) > 0 and not ego.dead:
        #         self.win = 1                
        #     # 如果超时，双方均未打光导弹/仍有导弹在空中飞，且双方均存活, 或者双方都死，判定为平
        #     else:
        #         self.draw = 1

        # ego_states = self.get_state(side)
        # enm_states = self.get_state(enm.side)
        # --- 3. 基础变量计算 ---
        ego_states, enm_states = ego.current_state, enm.current_state
        dist_enm2ego = norm(ego.pos_ - enm.pos_)
        
        cos_ATA_enm = np.dot(enm.vel_, (ego.pos_ - enm.pos_)) / (norm(enm.vel_) * dist_enm2ego + 1e-3)
        ATA_enm = np.arccos(np.clip(cos_ATA_enm, -1, 1))
        delta_theta = ego_states["target_information"][2]
        distance = ego_states["target_information"][3]
        speed = ego_states["ego_main"][0]
        alt = ego_states["ego_main"][1]
        cos_delta_psi = ego_states["target_information"][0]
        sin_delta_psi = ego_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = ego_states["target_information"][4]
        # 严格被锁判定
        strict_locked_by_target = ego_states["locked_by_target"] and (dist_enm2ego <= 80e3) and (ATA_enm <= pi/3)
        
        AA_hor = ego_states["target_information"][-2]
        warning = ego_states["warning"]
        missile_in_mid_term = ego_states["missile_in_mid_term"]
        missile_time_since_shoot = ego_states["weapon"]
        
        cos_delta_psi_threat = ego_states["threat"][0]
        sin_delta_psi_threat = ego_states["threat"][1]
        threat_distance = ego_states["threat"][3]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)
        delta_theta_threat = ego_states["threat"][2]
        
        # 奖励项初始化
        r_event = 0.0      # 结果奖励
        r_constraint = 0.0 # 约束与代价
        r_shaping = 0.0    # 战术引导

        # --- 4. 约束奖励计算 (r_constraint) - 固定权重 ---
        # 物理限制
        r_constraint += ((ego.alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                        (ego.alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * weights['alt_limit_penalty']
        
        # 边界限制
        o002ego_ = np.array([ego.pos_[0], ego.pos_[2]])
        ego_vh_ = np.array([ego.vel_[0], ego.vel_[2]])
        d_hor = ego_states["border"][0]
        if d_hor <= 50e3:
            r_constraint -= (1-d_hor/50e3) * np.dot(ego_vh_, o002ego_)/norm(o002ego_ + 1e-3)/340 * weights['border_penalty_scale']
        else:
            r_constraint += weights['border_reward']
        
        # 飞行品质惩罚
        r_constraint -= weights['aoa_penalty'] * ((ego.alpha_air*180/pi > 15)*(ego.alpha_air*180/pi-15) + \
                                                 (ego.alpha_air*180/pi < -5)*(-5 - ego.alpha_air*180/pi))
        r_constraint -= weights['pitch_penalty'] * (abs(ego.theta)/pi*2)

        # 开火代价控制
        shoot = action_shoot
        wasted = 0
        is_dead_now = ego.dead or self.out_range(ego)
        if is_dead_now and not getattr(ego, 'last_dead', False):
            shoot = ego.ammo
            wasted = ego.ammo
            ego.last_dead = True
        elif is_dead_now:
            shoot = 0 # 已经死过了，不再重复扣除

        if shoot >= 1:
            # 基础开火惩罚
            r_constraint -= (4 if alpha*180/pi > 10 else 3) * shoot
            if len(alive_ally_missiles) > 1:
                r_constraint -= 10 * shoot
            
            # 战术开火质量（虽然是引导，但与资源挂钩，建议放在约束/代价中防止乱开火）
            if not ego.dead:
                r_constraint += 1.0 * (pi/3 - alpha)/(pi/3)
                r_constraint += 0.6 * (abs(AA_hor)/pi - 1)
                r_constraint += 1.0 * (np.clip(ego.theta/(pi/3), -1, 1) - 1)
                if distance > 60e3:
                    r_constraint += -5 * (distance - 60e3)/20e3

        # --- 5. 引导奖励计算 (r_shaping) - 外部随步数衰减 ---
        if missile_in_mid_term: r_shaping += weights['missile_guidance']
        if ego_states["target_locked"]: r_shaping += weights['target_locked']
        if strict_locked_by_target: r_shaping -= weights['strict_locked_by_target']
        if warning: r_shaping -= weights['missile_warning']
        if enm_states["warning"]: r_shaping += weights['enemy_gets_warning']

        # 优势度引导
        if len(alive_ally_missiles) == 0 and ego.ammo > 0 and not warning:
            # 角度优势度
            r_shaping += (ATA_enm / pi - alpha / pi) * weights['angle_advantage']
            # 高度优势度
            r_shaping += (ego.alt - enm.alt)/5000 * weights['height_advantage']

        # 防御引导
        if warning:
            threat_directio_n = np.array([cos(delta_theta_threat)*cos(delta_psi_threat), 
                                         sin(delta_theta_threat), 
                                         cos(delta_theta_threat)*sin(delta_psi_threat)])
            if threat_distance <= 30e3:
                r_shaping += weights['defensive_angle_close'] * abs(delta_psi_threat) / pi
                r_shaping += np.dot(ego.vel_,threat_directio_n)/340 * weights['defensive_run_close']
            else:
                r_shaping += weights['defensive_angle_far'] * abs(delta_psi_threat) / pi
                if missile_in_mid_term:
                    r_shaping -= weights['defensive_crank_penalty'] * abs(alpha-pi/3)/(pi/3)
                    
        # 开火引导：
        '''
        你给我搞丢了,快加回来
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
        
        '''

        # --- 6. 结果奖励计算 (r_event) - 核心稀疏奖励 ---
        if ego.escape_once: r_event += 20
        if enm.escape_once: r_event -= 20
        if wasted > 0: r_event -= 20 * wasted # 死亡导致的导弹浪费惩罚

        if done:
            time_left = self.game_time_limit - self.t
            steps_left = time_left / (action_cycle_multiplier * self.dt_maneuver/0.2)
            # 这里的 total_weight 建议只包含 shaping 权重之和，用于平滑过渡
            total_shaping_sum = sum(weights.values())

            if self.win:
                r_event += 100 + steps_left * total_shaping_sum
            elif self.lose:
                r_event -= 100 + steps_left * total_shaping_sum
                if self.out_range(ego) or ego.alt < self.min_alt:
                    r_event -= 50
            elif self.draw:
                r_event -= 50
            
            # 打印详细奖励组成，方便调试 lambda 缩放比例
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if self.win else 'Lose' if self.lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Constraint: {r_constraint:.2f} | R_Shaping: {r_shaping:.2f}")

        # # 更新历史状态
        # self.last_obs[side] = {
        #     "target_locked": float(ego_states["target_locked"]),
        #     "strict_locked_by_target": float(strict_locked_by_target),
        #     "warning": float(warning),
        #     "enm_warning": float(enm_states["warning"])
        # }

        # 返回 done 和三个分项奖励
        return done, r_event, r_constraint, r_shaping

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
