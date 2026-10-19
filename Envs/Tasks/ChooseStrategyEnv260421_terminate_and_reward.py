'''

'''

import numpy as np
from random import random
import random
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
from .ChooseStrategyEnv260421_action_spaces import ChooseStrategyEnv as BaseChooseStrategyEnv

# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

def sigmoid(x):
    return 1/(1+np.exp(-x))
def arcsigmoid(x):
    return -np.log(1/(x)-1)
def softplus(x):
    return np.log(1+np.exp(x))

class ChooseStrategyEnv(BaseChooseStrategyEnv):
    def combat_terminate_and_reward(self, id, action_cycle_multiplier=10):
        self.update_entity_alives() # 更新所有飞机和导弹存活情况
        # 奖励权重预置
        DENSE_REWARD_WEIGHT = 0.2   # 密集奖励参考值
        END_REWARD_WEIGHT = 30      # 结果奖励
        # 事件奖励
        DIE_WEIGHT = -10            # 被击杀
        KILL_WEIGHT = 10            # 拿到击杀
        MSL_LOCK_WEIGHT = 5         # 导弹锁定
        MISS_WEIGHT = -5            # 脱靶
        ESCAPE_ONCE_WEIGHT = 5      # 逃脱一次
        FOX3 = -5                   # 开火惩罚
        
        # 各个分项总奖励
        reward_dict = {
            "event": 0,
            "end": 0,
            "shaping": 0
        }
        # --- [新增] 态势记录 ---
        # 计算是否到达记录周期
        cycle_time = self.dt_maneuver * action_cycle_multiplier
        step_idx = round(self.t / cycle_time)
        # 我是谁
        ego = self.find_uav_with_id(id)
        side = ego.side
        # 过程事件奖励是否允许输出(时间戳保护：每架飞机各自记账，同一周期内只结算一次)
        event_reward_available = 0
        if abs(self.t - step_idx * cycle_time) < 1e-4 and \
                (self.t - getattr(ego, 'last_cycle_t', -1e9)) > (cycle_time * 0.5):
            event_reward_available = 1
            ego.last_cycle_t = self.t
            self.last_record_t = self.t

        # 瞬时事件标志逐帧并入整周期缓存（这些标志在每次step开头被清零，必须每帧都读）
        if getattr(ego, 'escape_once', 0):   ego.whole_cycle_escape_once = 1
        if getattr(ego, 'got_a_miss', 0):    ego.whole_cycle_got_a_miss = 1
        if getattr(ego, 'got_a_kill', 0):    ego.whole_cycle_got_a_kill = 1
        if getattr(ego, 'can_let_go', 0):    ego.whole_cycle_can_let_go = 1

        if side == 'r':
            # 同侧
            egoside_alive_uavs =      self.alive_r_uavs
            egoside_alive_msls =      self.alive_r_missiles
            egoside_unused_msl_num = self.unused_r_missile_num
            # 对手侧
            enmside_alive_uavs =      self.alive_b_uavs
            enmside_alive_msls =      self.alive_b_missiles
            enmside_unused_msl_num = self.unused_b_missile_num
        else: # 'b'
            # 同侧
            egoside_alive_uavs =      self.alive_b_uavs
            egoside_alive_msls =      self.alive_b_missiles
            egoside_unused_msl_num = self.unused_b_missile_num
            # 对手侧
            enmside_alive_uavs =      self.alive_r_uavs
            enmside_alive_msls =      self.alive_r_missiles
            enmside_unused_msl_num = self.unused_r_missile_num

        # 胜负平续判决
        # self.close_range_kill() # 允许跑刀
        if getattr(self, 'last_shoot_9x_t', None) != self.t: # 同一帧内多机重复调用只执行一次
            self.last_shoot_9x_t = self.t
            self.close_range_shoot() # 近距格斗弹自动发射（规则控制）
        
        # 胜1 平0 负-1 续None
        done = 0
        outcome = None # 默认继续
        if len(egoside_alive_uavs) + len(enmside_alive_uavs) == 0:
            outcome = 0 # 平
            done = 1
        # 我方飞机和导弹全死，敌方飞机有剩下
        elif len(egoside_alive_uavs) + len(egoside_alive_msls) == 0 and\
            len(enmside_alive_uavs) > 0:
            outcome = -1 # 输
            done = 1
        # 敌方飞机和导弹全死，我方飞机有剩下
        elif len(enmside_alive_uavs) + len(enmside_alive_msls) == 0 and\
            len(egoside_alive_uavs) > 0:
            outcome = 1 # 赢
            done = 1
        # 超时
        elif self.t >= self.game_time_limit:
            # 超时，按剩余飞机数量判别胜负
            outcome = np.sign(len(egoside_alive_uavs) - len(enmside_alive_uavs))
            done = 1
        # 时间内，而且双方飞机都有剩
        else:
            outcome = None # 继续
            done = 0
        
        # 回填开火记录的命中/脱靶结果 ---
        # 每次调用都扫描本侧全部已发射导弹（不论死活）；hit 优先于 dead（命中导弹同时 dead=1）
        fire_hit_records = getattr(ego, 'fire_hit_records', None)
        if fire_hit_records:
            msl_table = {m.id: m for m in ego.launched_aim120s}
            for rec in fire_hit_records:
                if rec['hit_record'] is not None:
                    continue
                m = msl_table.get(rec['msl_id'])
                if m is None:
                    continue
                if m.hit:
                    rec['hit_record'] = 1
                elif m.dead:
                    rec['hit_record'] = 0
        # 终局强制结算：done=1 时仍未分辨命中/脱靶的记录整行删除
        if done and fire_hit_records:
            ego.fire_hit_records = [rec for rec in fire_hit_records if rec['hit_record'] is not None]


        # 事件结算 + 奖励：周期边界 或 终局（防止最后一个不完整周期里的事件丢失）
        # 每架飞机在各自的调用里只结算自己，last_cycle_t 防止同一时刻重复结算
        if event_reward_available or (done and getattr(ego, 'last_cycle_t', -1e9) != self.t):
            ego.last_cycle_t = self.t
            t_now = (int(self.t/60), int(self.t%60))
            # 密集奖励
            r_shaping = 0
            # 获取信息
            _ ,_ , ego_full_state = self.get_obs(id, reward_fn=1)
            RWR_infos = ego_full_state["RWR_infos"]
            warning = RWR_infos["warning"]
            cos_threat_delta_psi = RWR_infos["cos_threat_delta_psi"]
            threat_theta = RWR_infos["threat_theta"]
            weapon_infos = ego_full_state["weapon_infos"]
            missile_t_fly = weapon_infos["missile_t_fly"]
            missile_in_mid_term = weapon_infos["mid_term"]
            ammo = ego.ammo
            
            enm_infos = ego_full_state["enm_infos"]
            ally_infos = ego_full_state["ally_infos"]
            
            # 敌机全灭或 POMDP 下不可见时 enm_infos 为空：锁定标志取 False，ATA 取中性值 pi/2（cos=0、不触发贴脸惩罚）
            if enm_infos:
                ego_has_lock = any(temp["enm_locked"] for temp in enm_infos)
                ego_is_locked = any(temp["locked_by_enm"] for temp in enm_infos)
                ata_list = [temp["enm_ATA"] for temp in enm_infos]
                min_enm_ATA = min(ata_list)
                max_enm_ATA = max(ata_list)
                mean_enm_ATA = float(np.mean(ata_list))
            else:
                ego_has_lock = False
                ego_is_locked = False
                min_enm_ATA = max_enm_ATA = mean_enm_ATA = pi / 2
            
            # 速度奖励（常驻）
            mach = ego.mach
            r_shaping += (mach-1.0)*2 * DENSE_REWARD_WEIGHT/2
            
            # 受到威胁时
            if warning:
                # 下高引导
                r_shaping += (-ego.vu/100) * np.sign(threat_theta-(-pi/4)) * DENSE_REWARD_WEIGHT
                # 三九和拖尾机动的奖励在动作mask面前很微弱，不写了
            else:
                # 锁定/被锁定
                r_shaping += (ego_has_lock - ego_is_locked) * DENSE_REWARD_WEIGHT
                
                # 制导时
                if missile_t_fly < 40:
                    # 开火后下高
                    r_shaping += (-ego.vu/100) * np.sign(threat_theta-(-pi/4)) * DENSE_REWARD_WEIGHT
                    # 开火后不要继续指着敌机飞了
                    if min_enm_ATA < np.radians(15):
                        r_shaping += - DENSE_REWARD_WEIGHT
                
                # 没有导弹处于中制导阶段
                if (not missile_in_mid_term):
                    # 奖励指向任何一个目标
                    r_shaping += cos(min_enm_ATA) * DENSE_REWARD_WEIGHT
                    if ammo > 0: # 有弹药，爬高争取发射机会
                        r_shaping += (ego.vu/100) * DENSE_REWARD_WEIGHT
                    else: # 没有弹药，飞5000m
                        r_shaping += (-ego.vu/100) * np.sign(ego.alt-5000) * DENSE_REWARD_WEIGHT
                
                # TODO 指向场地中心奖励
            
            
            reward_dict["shaping"] = r_shaping
            reward_dict["shaping"] *= (1-ego.dead) # 死了没有密集奖励

            # 结果奖励
            if outcome is not None:
                reward_dict["end"] = outcome * END_REWARD_WEIGHT
            # 事件奖励+调试输出
            events_this_cycle = []
            # 本周期内刚死（last_dead只在此处更新，上一周期结束时已死的不重复记录）
            if ego.dead and not ego.last_dead:
                events_this_cycle.append('被击杀')
                reward_dict["event"] += DIE_WEIGHT
            ego.last_dead = bool(ego.dead)
            # 整周期内出现过的瞬时事件
            if getattr(ego, 'whole_cycle_can_let_go', 0):
                events_this_cycle.append('导弹锁定目标')
                reward_dict["event"] += MSL_LOCK_WEIGHT
            if getattr(ego, 'whole_cycle_got_a_kill', 0):
                events_this_cycle.append('导弹命中目标')
                reward_dict["event"] += KILL_WEIGHT
            if getattr(ego, 'whole_cycle_got_a_miss', 0):
                events_this_cycle.append('导弹脱靶')
                reward_dict["event"] += MISS_WEIGHT
            if getattr(ego, 'whole_cycle_escape_once', 0):
                events_this_cycle.append('逃脱导弹攻击')
                reward_dict["event"] += ESCAPE_ONCE_WEIGHT
            if outcome == 1:
                events_this_cycle.append('获胜')
            if outcome == -1:
                events_this_cycle.append('落败')
            if outcome == 0:
                events_this_cycle.append('平局')
            
            ego.whole_cycle_escape_once = 0
            ego.whole_cycle_got_a_miss = 0
            ego.whole_cycle_got_a_kill = 0
            ego.whole_cycle_can_let_go = 0
            
            # 开火惩罚，通过弹药下降沿判断
            if ego.ammo < ego.last_ammo_whole_cycle:
                current_target_id = ego.launched_aim120s[-1].target_id
                events_this_cycle.append(f'向{current_target_id}开火')
                reward_dict["event"] += FOX3
                ego.last_ammo_whole_cycle = ego.ammo

                 # [新增] 开火记录：快照当前观测，登记最新发射的导弹，等待命中/脱靶回填
                if not hasattr(ego, 'fire_hit_records'):
                    ego.fire_hit_records = []
                if len(ego.launched_aim120s) > 0:
                    newest_msl = ego.launched_aim120s[-1]
                    # 死亡清仓惩罚等 shoot>=1 但本步并无新弹发射的情形，避免重复登记同一枚导弹
                    # if all(rec['msl_id'] != newest_msl.id for rec in ego.fire_hit_records):
                        # TODO 即时获取一次瞬时观测，需要适配环境
                        # ego_obs, ego_check_obs = self.obs_1v1(ego.side, pomdp=1)
                        # ego.fire_hit_records.append({
                        #     "msl_id": newest_msl.id,
                        #     "fire_obs": copy.deepcopy(ego_obs),
                        #     "hit_record": None,
                        # })

            # 每个结算时间点只写一行：无事件时事件栏为空，三类奖励如实记0
            ego.cycle_events.append((t_now, ego.id,
                                     reward_dict["shaping"], reward_dict["end"], reward_dict["event"],
                                     ';'.join(events_this_cycle)))
            
            # 胜负结果可导出
            if side == 'r':
                self.r_outcome = outcome
                if outcome is not None:
                    self.b_outcome = - outcome
            else:
                self.b_outcome = outcome
                if outcome is not None:
                    self.r_outcome = - outcome
        return done, reward_dict