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
    def combat_terminate_and_reward(self, id, action_shoots, action_cycle_multiplier=10):
        self.update_entity_states() # 更新所有飞机和导弹存活情况
        # 奖励权重预置
        DENSE_REWARD_WEIGHT = 0.5   # 密集奖励参考值
        END_REWARD_WEIGHT = 30      # 结果奖励
        # 事件奖励
        DIE_WEIGHT = -10            # 被击杀
        KILL_WEIGHT = 10            # 拿到击杀
        MSL_LOCK_WEIGHT = 5         # 导弹锁定
        MISS_WEIGHT = 5             # 脱靶
        ESCAPE_ONCE_WEIGHT = 5      # 逃脱一次
        FOX3 = -5                   # 发射一枚导弹的基础惩罚
        
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
        elif self.t > self.game_time_limit:
            # 超时，按剩余飞机数量判别胜负
            outcome = np.sign(len(egoside_alive_uavs) - len(enmside_alive_uavs))
            done = 1
        # 时间内，而且双方飞机都有剩
        else:
            # # 双方中距弹耗尽，按剩余飞机数量判别胜负
            # if len(egoside_alive_msls) + egoside_unused_msl_num + \
            #     len(enmside_alive_msls) + enmside_unused_msl_num == 0:
            #         outcome = np.sign(len(egoside_alive_uavs) - len(enmside_alive_uavs))
            #         done = 1
            # # 我方一发可用的导弹都没有，而敌方还有导弹未发射
            # elif len(egoside_alive_msls) + egoside_unused_msl_num == 0 and\
            #     enmside_unused_msl_num > 0:
            #         if len(egoside_alive_uavs) <= len(enmside_alive_uavs):
            #             outcome = -1 # 预判为输
            #             done = 1
            #         else:
            #             outcome = None # 继续
            #             done = 0
            # else:
                outcome = None # 继续
                done = 0

        # 事件结算 + 奖励：周期边界 或 终局（防止最后一个不完整周期里的事件丢失）
        # 每架飞机在各自的调用里只结算自己，last_cycle_t 防止同一时刻重复结算
        if event_reward_available or (done and getattr(ego, 'last_cycle_t', -1e9) != self.t):
            ego.last_cycle_t = self.t
            t_now = (int(self.t/60), int(self.t%60))
            # 密集奖励
            reward_dict["shaping"] = 0

            # 结果奖励
            if outcome is not None:
                reward_dict["end"] = outcome * END_REWARD_WEIGHT
            
            
            
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
            # 有发射导弹
            if ego.ammo < ego.last_ammo_whole_cycle:
                current_target_id = ego.launched_aim120s[-1].target_id
                events_this_cycle.append(f'向{current_target_id}开火')
                reward_dict["event"] += FOX3
                ego.last_ammo_whole_cycle = ego.ammo

            # 每个结算时间点只写一行：无事件时事件栏为空，三类奖励如实记0
            ego.cycle_events.append((t_now, ego.id,
                                     reward_dict["shaping"], reward_dict["end"], reward_dict["event"],
                                     ';'.join(events_this_cycle)))

        return done, reward_dict