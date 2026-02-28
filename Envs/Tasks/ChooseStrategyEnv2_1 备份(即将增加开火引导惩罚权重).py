'''
增加开火惩罚
二元组奖励
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
            
        # [新增] 多枚导弹发射，但只有一枚能够命中，也算浪费
        if len(alive_ally_missiles) > 0 and enm.got_hit:
            reward_event -= escape_penalty * len(alive_ally_missiles)
        

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
        

        if should_fire_missile==1:
            nobody = 1
        if should_fire_missile==0 and shoot==1:
            nobody = 2

        if shoot >= 1:
            if alpha*180/pi > 10:
                reward_event -= 4*shoot
            else:
                reward_event -= 3*shoot
                
            if len(alive_ally_missiles)>1:
                reward_event -= 10*shoot # 重复发射惩罚
            
            if not ego.dead:
                reward_shoot += 1 * (pi/3-alpha)/(pi/3)
                reward_shoot += 0.6 * (abs(AA_hor)/pi-1)
                reward_shoot += 1 * (np.clip(ego.theta/(pi/3), -1, 1)-1)  # 鼓励抛射
                
                # 发射距离惩罚
                if distance > 60e3:
                    reward_shoot += -5 * (distance-60e3)/(20e3)

                # 发射间隔奖励 
                reward_shoot += 3 * np.clip((missile_time_since_shoot-30)/30, -1,1)  # 过30s发射就可以奖励了

        
        reward_assisted += reward_shoot

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
                    reward_assisted -= reward_weights['defensive_crank_penalty'] * abs(alpha-pi/3)/(pi/3)
            

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
