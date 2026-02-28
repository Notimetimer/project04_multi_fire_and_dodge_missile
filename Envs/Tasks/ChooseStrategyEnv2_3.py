'''
向量形式奖励
奖励函数表示为向量
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
from .ChooseStrategyEnv2_0 import ChooseStrategyEnv as BaseChooseStrategyEnv
from .ChooseStrategyEnv2_0 import action_options, action_optionsLR

# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

class ChooseStrategyEnv(BaseChooseStrategyEnv):
    """
    Inherit everything from ChooseStrategyEnv2_0 except override
    combat_terminate_and_reward (keeps other definitions/vars from base).
    """
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.reward_weights = {
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
        self.num_rewards = 13 # len(self.reward_weights.keys())
    
    def combat_terminate_and_reward(self, side, action_label, action_shoot, action_cycle_multiplier=30):
        # --- 1. 参数初始化与状态获取 ---
        # 权重在此仅作为内部计算比例，实际整体缩放由外部 lambda 控制
        reward_weights = self.reward_weights
        
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
            
        # --- 3. 基础变量计算 ---
        ego_states = ego.current_state
        enm_states = enm.current_state
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
        reward = np.zeros(self.num_rewards)
        
        r_event = 0.0      # 结果奖励
        r_constraint = 0.0 # 约束与代价
        r_shaping = 0.0    # 战术引导

        # --- 4. 约束奖励计算 (r_constraint) - 固定权重 ---
        # 高度限制奖励/惩罚
        r_constraint += ((alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                        (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * reward_weights['alt_limit_penalty']
        
        # 靠近边界惩罚
        o002ego_ = np.array([ego.pos_[0], ego.pos_[2]]) # 北，东
        ego_vh_ = np.array([ego.vel_[0], ego.vel_[2]])
        d_hor = ego_states["border"][0]
        if d_hor <= 50e3:
            r_constraint -= (1-d_hor/50e3) * np.dot(ego_vh_, o002ego_)/norm(o002ego_ + 1e-3)/340 * reward_weights['border_penalty_scale']
        else:
            r_constraint += reward_weights['border_reward']
        
        # 迎角惩罚
        r_constraint -= reward_weights['aoa_penalty'] * ((ego.alpha_air*180/pi > 15)*(ego.alpha_air*180/pi-15) + \
                                                         (ego.alpha_air*180/pi < -5)*(-5 - ego.alpha_air*180/pi))
        # 俯仰角惩罚
        r_constraint -= reward_weights['pitch_penalty'] * (abs(ego.theta)/pi*2)

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
            # 发射惩罚 (硬编码)
            if alpha*180/pi > 10:
                r_constraint -= 4 * shoot
            else:
                r_constraint -= 3 * shoot
            
            if len(alive_ally_missiles) > 1:
                r_constraint -= 10 * shoot
            
            # 发射时的态势惩罚/奖励（归类为资源使用的约束，防止乱射）
            if not ego.dead:
                r_constraint += 1.0 * (pi/3 - alpha)/(pi/3)
                r_constraint += 0.6 * (abs(AA_hor)/pi - 1)
                r_constraint += 1.0 * (np.clip(ego.theta/(pi/3), -1, 1) - 1)  # 鼓励抛射
                
                # 发射距离惩罚
                if distance > 60e3:
                    r_constraint += -5 * (distance - 60e3)/20e3

        # --- 5. 引导奖励计算 (r_shaping) - 外部随步数衰减 ---
        # 为导弹提供制导
        if missile_in_mid_term:
            r_shaping += reward_weights['missile_guidance']
            reward[0] += reward_weights['missile_guidance']

        # 锁定目标
        if ego_states["target_locked"]:
            r_shaping += reward_weights['target_locked']
            reward[1] += reward_weights['target_locked']

        # 被目标锁定
        if strict_locked_by_target:
            r_shaping -= reward_weights['locked_by_target']
            reward[2] -= reward_weights['locked_by_target']

        # 被导弹导引头锁住
        if warning and threat_distance <= 20e3:
            r_shaping -= reward_weights['missile_warning']
            reward[3] -= reward_weights['missile_warning']

        # 导弹锁定目标
        if enm_states["warning"] and enm_states["threat"][3] <= 20e3:
            r_shaping += reward_weights['enemy_gets_warning']
            reward[4] += reward_weights['enemy_gets_warning']

        # 优势度引导
        if len(alive_ally_missiles) == 0 and ego.ammo > 0 and not warning:
            # 角度优势度
            r_shaping += (ATA_enm / pi - alpha / pi) * reward_weights['angle_advantage']
            reward[5] += (ATA_enm / pi - alpha / pi) * reward_weights['angle_advantage']
            # 高度优势度
            r_shaping += (alt - enm.alt)/5000 * reward_weights['height_advantage']
            reward[6] += (alt - enm.alt)/5000 * reward_weights['height_advantage']

        # 防御引导
        if warning:
            threat_directio_n = np.array([cos(delta_theta_threat)*cos(delta_psi_threat), 
                                         sin(delta_theta_threat), 
                                         cos(delta_theta_threat)*sin(delta_psi_threat)])
            if threat_distance <= 30e3:
                r_shaping += reward_weights['defensive_angle_close'] * abs(delta_psi_threat) / pi
                reward[7] += reward_weights['defensive_angle_close'] * abs(delta_psi_threat) / pi
                r_shaping += np.dot(ego.vel_, threat_directio_n)/340 * reward_weights['defensive_run_close']
                reward[8] += np.dot(ego.vel_, threat_directio_n)/340 * reward_weights['defensive_run_close']
            else:
                r_shaping += reward_weights['defensive_angle_far'] * abs(delta_psi_threat) / pi
                reward[9] += reward_weights['defensive_angle_far'] * abs(delta_psi_threat) / pi
                if missile_in_mid_term:
                    r_shaping -= reward_weights['defensive_crank_penalty'] * abs(abs(delta_psi)-pi/3)/(pi/3)
                    reward[10] -= reward_weights['defensive_crank_penalty'] * abs(abs(delta_psi)-pi/3)/(pi/3)

        # [加回] 开火引导逻辑 (Should fire vs Shoot)
        should_fire_missile = False
        if distance < 60e3 and alpha < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
            if missile_time_since_shoot >= 20 and not missile_in_mid_term and not (distance > 12e3 and abs(AA_hor) < 30*pi/180):
                should_fire_missile = True
        
        reward_shoot_coach = 0
        if shoot == 1:
            if should_fire_missile:
                reward_shoot_coach += 10
            else:
                reward_shoot_coach -= 10
        if shoot == 0:
            if should_fire_missile:
                reward_shoot_coach -= 10
            else:
                reward_shoot_coach += 0.01
        
        r_shaping += reward_shoot_coach # 归入引导奖励
        reward[11] += reward_shoot_coach

        # --- 6. 结果奖励计算 (r_event) - 核心稀疏奖励 ---
        # 逃脱导弹
        if ego.escape_once:
            r_event += 20
        # 导弹被逃脱
        if enm.escape_once:
            r_event -= 20
            
        # 死了也当剩下导弹全被逃脱处理 (自杀代价补偿)
        if wasted > 0:
            r_event -= 20 * wasted

        if done:
            time_left = self.game_time_limit - self.t
            steps_left = time_left / (action_cycle_multiplier * self.dt_maneuver/0.2)
            total_shaping_sum = sum(reward_weights.values())

            if self.win:
                r_event += 150 # 100 + steps_left * total_shaping_sum
            elif self.lose:
                r_event -= 100 + steps_left * total_shaping_sum
                if self.out_range(ego) or ego.alt < self.min_alt:
                    r_event -= 50
            elif self.draw:
                r_event -= 50
            
            # 打印详细奖励组成，方便调试
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if self.win else 'Lose' if self.lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Constraint: {r_constraint:.2f} | R_Shaping: {r_shaping:.2f}")

        # 返回 done 和三个分项奖励

        reward[-1] = r_event + r_constraint
        return done, reward, r_event + r_constraint + r_shaping
