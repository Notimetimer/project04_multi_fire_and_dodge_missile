'''
增加开火惩罚
三元组奖励
'''

from Controller.Controller_function import sub_of_radian
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

from Envs.battle6dof1v1_missile0309_hierarchical import *
from .ChooseStrategyEnv2_0_hierarchical import ChooseStrategyEnv as BaseChooseStrategyEnv
from .ChooseStrategyEnv2_0_hierarchical import action_optionsLR

# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

class ChooseStrategyEnv(BaseChooseStrategyEnv):
    """
    Inherit everything from ChooseStrategyEnv2_0 except override
    combat_terminate_and_reward (keeps other definitions/vars from base).
    """
    
    def combat_terminate_and_reward(self, side, action_label, action_shoot, action_cycle_multiplier=30):
        # --- 1. 参数初始化与状态获取 ---
        # 权重在此仅作为内部计算比例，实际整体缩放由外部 lambda 控制
        reward_weights = {
            'missile_guidance': 0.04,
            'target_locked': 0.06,
            'locked_by_target': 0.05,
            'missile_warning': 0.06,
            'enemy_gets_warning': 0.05,
            'alt_limit_penalty': 1.0,
            'border_penalty_scale': 0.2,
            'border_reward': 0.2, # 旧的数值: 1.0, 新的数值：0.2
            'angle_advantage': 0.01, # 0.007, # 0.03
            'height_advantage': 0.01,
            'aoa_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.2
            'pitch_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.05
            'to_center_reward' : 0.005, # 0.02 占领中心点的价值
            'speed_penalty': 0.01, # 慢速惩罚
        }

        ego_win=0
        ego_lose=0
        ego_draw=0

        # --- [新增] 态势记录 ---
        # 计算是否到达记录周期
        cycle_time = self.dt_maneuver * action_cycle_multiplier
        step_idx = round(self.t / cycle_time)
        
        # 允许极小的误差，并通过 last_record_t 避免在一帧内被两个智能体调用时重复记录
        if abs(self.t - step_idx * cycle_time) < 1e-4 and (self.t - self.last_record_t) > (cycle_time * 0.5):
            r_dist = np.linalg.norm(np.array([self.RUAV.pos_[0] - self.horizontal_center[0], self.RUAV.pos_[2] - self.horizontal_center[1]]))
            b_dist = np.linalg.norm(np.array([self.BUAV.pos_[0] - self.horizontal_center[0], self.BUAV.pos_[2] - self.horizontal_center[1]]))
            self.r_dist_seq.append(r_dist)
            self.b_dist_seq.append(b_dist)
            self.last_record_t = self.t

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
            ego_win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方存活，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            ego_lose = 1
            done = 1
        # 双杀双活时间到，就是平局
        elif done: 
            ego_draw = 1


        # 回合的胜负取决于ego_side
        if ego.side == self.ego_side:
            self.win = ego_win
            self.lose = ego_lose
            self.draw = ego_draw
        else:
            self.win = ego_lose
            self.lose = ego_win
            self.draw = ego_draw
        
        # ego_states = self.get_state(side)
        # enm_states = self.get_state(enm.side)
        # --- 3. 基础变量计算 --- pomdp会导致观测污染奖励函数，必须重新获取一次状态变量
        ego_states = copy.deepcopy(self.get_state(side))
        enm_states = copy.deepcopy(self.get_state(enm.side))
        # ego_states = ego.current_state
        # enm_states = enm.current_state
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
        strict_locked_by_target = ego_states["locked_by_target"] and (dist_enm2ego <= 80e3) and (ATA_enm <= self.RUAV.max_radar_angle_rad)
        
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
        # # 高度限制奖励/惩罚
        # r_constraint += ((alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
        #                 (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * reward_weights['alt_limit_penalty']
        
        # 靠近边界惩罚
        o002ego_ = np.array([ego.pos_[0], ego.pos_[2]]) # 北，东
        ego_vh_ = np.array([ego.vel_[0], ego.vel_[2]])
        d_hor = ego_states["border"][0]
        if d_hor <= 50e3:
            r_constraint -= (1-d_hor/50e3) * np.dot(ego_vh_, o002ego_)/norm(o002ego_ + 1e-3)/340 * reward_weights['border_penalty_scale']
        else:
            r_constraint += reward_weights['border_reward']
        
        # 补充占领中心奖励
        border_side = ego_states["border"][1]
        
        line2center = np.array([self.horizontal_center[0]-ego.pos_[0], self.horizontal_center[1]-ego.pos_[2]])
        psi2center = np.arctan2(line2center[1], line2center[0])
        dist2center = norm(line2center)
        delta_psi_to_center = sub_of_radian(psi2center, ego.psi)

        # 距离中心越远，指向中心的价值越高
        r_constraint += (1 + dist2center/self.R_cage)/2 * \
                        (1 - abs(delta_psi_to_center)/pi) * \
                        reward_weights['to_center_reward']
        
        # # 迎角惩罚
        # r_constraint -= reward_weights['aoa_penalty'] * ((ego.alpha_air*180/pi > 15)*(ego.alpha_air*180/pi-15) + \
        #                                                  (ego.alpha_air*180/pi < -5)*(-5 - ego.alpha_air*180/pi))
        # # 俯仰角惩罚
        # r_constraint -= reward_weights['pitch_penalty'] * (abs(ego.theta)/pi*2)

        r_constraint *= (1-ego.dead) # 密集奖励只有在存活的时候有意义

        # 开火代价控制
        wasted = 0
        now_dead = ego.dead or self.out_cage(ego)
        if not getattr(ego, 'last_dead', False):
            if now_dead:
                shoot = ego.ammo # 死亡瞬间，记录清仓惩罚
                wasted = ego.ammo
                ego.last_dead = True
            else:
                shoot = action_shoot # 还在飞，正常记录
        else:
            shoot = 0 # 死后不再记录任何幻影开火
            wasted = 0


        # --- 态势辅助奖励 (千分位级别) ---
        # 1. 进攻态势：我方导弹是否横在两机之间 (敌机感受到的威胁距离 < 两机距离)
        enm_threat_dist = enm_states["threat"][3]
        if enm_threat_dist < distance:
            r_constraint += 0.0005 * (1 - ego.dead) # 0.001

        # 2. 防御态势：敌方导弹是否横在两机之间 (我方感受到的威胁距离 < 两机距离)
        if threat_distance < distance:
            r_constraint -= 0.0005 * (1 - ego.dead) # 0.001

        # 角度奖励
        # 进攻引导
        if len(alive_ally_missiles) == 0 and not warning:
            # 瞄准奖励
            r_constraint += cos(delta_psi) * reward_weights['angle_advantage'] * (1-ego.dead)
            # 爬高奖励
            r_constraint += 0.5 * (ego.alt/1e5) * reward_weights['height_advantage'] * (1-ego.dead)
            r_constraint += 0.5 * min(ego.theta/(pi/4), 1) * reward_weights['angle_advantage'] * (1-ego.dead)
        # crank引导
        if len(alive_ally_missiles) > 0 and not warning:
            # 开火后crank下高，误差惩罚改为“保持中制导条件下的奖励”
            r_constraint += 4 * (1 - abs(pi/3-abs(delta_psi))/(pi/3)) * reward_weights['angle_advantage'] * missile_in_mid_term * (1-ego.dead)
            r_constraint += 5 * (1 - abs(-pi/4 - ego.theta) / (pi/4)) * reward_weights['angle_advantage'] * missile_in_mid_term * (1-ego.dead)
        # 防御引导
        if warning:
            # 受到威胁应该三九线/置尾和下高
            if abs(delta_psi_threat) < pi/2:
                r_constraint += min(abs(delta_psi_threat), pi/2)/(pi/2) * reward_weights['angle_advantage'] * (1-ego.dead)
            r_constraint += (-ego.theta)/(pi/2) * reward_weights['angle_advantage'] * (1-ego.dead)
        
        # 速度惩罚
        slow_mach = 0.7
        if ego.speed < slow_mach*340:
            r_constraint -= (slow_mach-ego.speed/340) * reward_weights['speed_penalty'] * (1-ego.dead)

        
        # 密集奖励只有在agent活着的时候有意义
        # r_constraint *= (1-ego.dead)
        # r_shaping *= (1-ego.dead)

        # --- 6. 结果奖励计算 (r_event) - 核心稀疏奖励 ---
        if shoot >= 1:
            r_event -= 5 * shoot

            if len(alive_ally_missiles) > 1: # 重复开火惩罚
                r_event -= 20 * shoot * max(1-missile_time_since_shoot/60, 0) # 13
            
            # # 发射时的态势惩罚/奖励（归类为资源使用的约束，防止乱射）
            if not ego.dead:
                r_event += 1.0 * (pi/3 - abs(delta_psi))/(pi/3) # 鼓励抛射就得把alpha解耦出来
                r_event += 1.0 * (abs(AA_hor)/pi - 1) # 0.6 鼓励对头射击，惩罚追尾射击
                r_event += 1.3 * (np.clip(ego.theta/(pi/3), -1, 1) - 1)  # 鼓励抛射 # 1.0 # 高抛项太多了，都忽视速度了
                r_event -= 0.7 * max(1.0-ego.speed/340, 0)  # 开火时候的速度不能太低
                

        # 逃脱导弹
        if ego.escape_once:
            r_event += 20 * (1-ego.dead) # 活着才算逃脱，否则只是游戏机制
        # 导弹被逃脱
        if enm.escape_once:
            r_event -= 20 * (1-enm.dead)
            
        # 死了也当剩下导弹全被逃脱处理 (自杀代价补偿)
        if wasted > 0:
            r_event -= 20 * wasted

        if done:
            time_left = self.game_time_limit - self.t
            steps_left = time_left / (action_cycle_multiplier * self.dt_maneuver/0.2)
            total_shaping_sum = sum(reward_weights.values())

            if ego_win:
                r_event += 180 # 150 + 0.2 * steps_left * total_shaping_sum # 旧 150 新 145
            elif ego_lose:
                r_event -= 180 # 125 + steps_left * total_shaping_sum # 旧 100 新 125
                if self.out_cage(ego) or ego.alt < self.min_alt:
                    r_event -= 50
            elif ego_draw:
                # [修改] 不再使用常数-50奖励，而是根据平均态势分来结算
                if len(self.r_dist_seq) > 0 and len(self.b_dist_seq) > 0:
                    r_avg_dist = sum(self.r_dist_seq) / len(self.r_dist_seq)
                    b_avg_dist = sum(self.b_dist_seq) / len(self.b_dist_seq)
                    
                    if side == 'r':
                        ego_avg_dist = r_avg_dist
                        enm_avg_dist = b_avg_dist
                    else:
                        ego_avg_dist = b_avg_dist
                        enm_avg_dist = r_avg_dist
                    
                    # 赢不了，也要占据中心，并把对手逼到边上，如果赢了或者输了，都禁止加这个奖励
                    r_event += -30 - 20 * (ego_avg_dist-enm_avg_dist)/self.R_cage0
                    self.middle_hold_score = (ego_avg_dist-enm_avg_dist)/self.R_cage0
                else:
                    # 如果由于某种原因没有记录到态势数据，退回之前的默认惩罚或给0
                    pass
                
                # # “同归于尽收回导弹浪费惩罚，先死才有补偿，后死照样惩罚”
                # if enm.dead and getattr(ego, 'last_dead', False):
                #     r_event += 15 * ego.ammo
            
            # 打印详细奖励组成，方便调试
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if ego_win else 'Lose' if ego_lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Constraint: {r_constraint:.2f} | R_Shaping: {r_shaping:.2f}")

        # 返回 done 和三个分项奖励
        return done, r_event, r_constraint, r_shaping
