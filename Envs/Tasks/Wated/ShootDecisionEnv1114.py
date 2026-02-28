'''出生点改在外面指定'''
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


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from Envs.battle6dof1v1_missile0919 import *


# 通过继承构建观测空间、奖励函数和终止条件

class ShootTrainEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.attack_key_order = [
            "target_observable", # 1
            "target_locked", # 1
            "missile_in_mid_term",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "weapon", # 1
        ]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # 初始化红蓝远离速度
        self.last_dist_dot = None
        self.last_dhor = None
        self.lock_time_count = 0

        self.target_hit = 0
        self.target_out = 0

    def step(self, r_actions, b_actions):
        report_move_time_rate = int(round(self.dt_maneuver / dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]
        self.r_actions = r_actions.copy()
        self.b_actions = b_actions.copy()

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                # print(action)
                target_height = action[0]  # 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                delta_heading = action[1]  # 相对方位(弧度)
                target_speed = action[2]  # 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
                # print('target_height',target_height)

                if UAV.blue:
                    # 如果 BLUE_BIRTH_STATE 包含 p2p 则使用其值，否则为 False
                    p2p = self.BLUE_BIRTH_STATE.get('p2p', False)
                if UAV.red:
                    # 对红方同样兼容 RED_BIRTH_STATE 中可能存在的 p2p 字段
                    p2p = self.RED_BIRTH_STATE.get('p2p', False)

                # 出界强制按回
                if self.out_range(UAV):
                    target_direction_ = horizontal_center - np.array(UAV.pos_[0], UAV.pos_[2])
                    delta_heading = sub_of_radian(atan2(target_direction_[1], target_direction_[0]), UAV.psi)
                    p2p = False # 只能用PID来按回


                UAV.move(target_height, delta_heading, target_speed, relevant_height=True, p2p=p2p)
                # 上一步动作
                # UAV.act_memory = np.array([action[0],action[1],action[2]])

            hitter = None

            # 导弹移动
            self.missiles = self.Rmissiles + self.Bmissiles
            for missile in self.missiles[:]:  # 使用切片创建副本以允许删除
                target = self.get_target_by_id(missile.target_id)
                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹, 在飞机1V1的时候可以节省一点计算量，不用费事处理多目标的问题
                    missile.dead = True
                    continue
                else:
                    missile.target = target
                # if not missile.dead:
                # print('目标位置', target.pos_)
                # 计算前导弹和目标位速
                last_pmt_ = missile.pos_
                last_vmt_ = missile.vel_
                last_ptt_ = target.pos_
                last_vtt_ = target.vel_
                if not missile.dead:
                    # 获取目标信息
                    target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, last_ptt_)
                    # 更新导弹制导阶段
                    has_datalink = False
                    for uav in self.UAVs:
                        # 找到载机，判断载机能否为导弹提供中制导
                        if uav.id == missile.launcher_id:
                            if uav.can_offer_guidance(missile, self.UAVs):
                                has_datalink = True
                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt=self.dt_move, datalink=has_datalink)

                vmt1 = norm(last_vmt_)
                if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                    missile.dead = True
                if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                    missile.dead = True
                if missile.t > missile.t_max:  # 超时自爆
                    missile.dead = True
                if missile.t >= 0 + self.dt_move and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=self.dt_move)
                    if hit:
                        print(target.side, 'is hit')
                        missile.dead = True
                        missile.hit = True
                        missile.pos_ = point_m
                        missile.vel_ = last_vmt_
                        target.pos_ = point_t
                        target.vel_ = last_vtt_
                        target.dead = True
                        target.got_hit = True
                        self.UAV_hit[self.UAV_ids.index(target.id)] = True

                        hitter = missile.id if hitter is None else min(hitter, missile.id)                    

                if missile.dead == True and not hit:
                    target.escape_once = 1
                    # 目标逃脱
                else:
                    target.escape_once = 0

            # 毁伤判断
            for i, UAV in enumerate(self.UAVs):
                # 飞机被导弹命中判断
                if UAV.red:
                    adv = self.BUAV
                if UAV.blue:
                    adv = self.RUAV
                if self.UAV_hit[i]:
                    UAV.dead = True
                    UAV.got_hit = True
                # 其他毁伤判断
                adv = self.UAVs[1 - i]
                pt_ = adv.pos_
                L_ = pt_ - UAV.pos_
                distance = np.linalg.norm(L_)
                # 近距杀
                #     short_range_killed = UAV.short_range_kill(adv)
                #     if short_range_killed:
                #         # self.running = False
                #         adv.got_hit = True
                # 出界判别
                if self.crash(UAV):
                    UAV.dead = True
                # self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash:  # or self.out_range(UAV): ###
                UAV.dead = True
                # self.running = False

        r_dones = False
        b_dones = False
        if self.RUAV.dead:
            r_dones = True
        if self.BUAV.dead:
            b_dones = True

        self.RUAV = self.UAVs[0]
        self.BUAV = self.UAVs[1]

        if terminate:
            self.running = False

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate, hitter
    
    # 进攻策略观测量
    def attack_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.attack_key_order}
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        # full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        # full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])
        # full_obs["threat"] = copy.deepcopy(self.obs_init["threat"])
        # full_obs["border"] = copy.deepcopy(self.obs_init["border"])

        # 将观测按顺序拉成一维数组
        # flat_obs = flatten_obs(full_obs, self.key_order)
        flat_obs = flatten_obs(full_obs, self.attack_key_order)
        return flat_obs, full_obs

    def attack_terminate_and_reward(self, side, u):  # 进攻策略训练与奖励
        # 兼容：若 u 是一维数组或列表，则取第0个元素；若为标量则直接使用（不考虑更高维度）
        if isinstance(u, (list, tuple, np.ndarray)):
            arr = np.array(u)
            if arr.ndim == 1:
                ut = arr[0]
            elif arr.ndim == 0:
                ut = arr.item()
            else:
                ut = u
        else:
            ut = u
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        AA_hor = state["target_information"][6]
        launch_interval = state["weapon"]

        missile_time_since_shoot = state["weapon"]

        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            all_ally_missiles = self.Rmissiles
            alive_ally_missiles = self.alive_r_missiles

        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            all_ally_missiles = self.Bmissiles
            alive_ally_missiles = self.alive_b_missiles

        target_alt = enm.alt
        enm_state = self.get_state(enm.side)

        # if alpha < 10*pi/180:
        #     self.lock_time_count += dt_maneuver
        # if alpha > 30*pi/180:
        #     self.lock_time_count = 0

        # 结束判断
        if self.t > self.game_time_limit:
            terminate = True
            # self.lose = 1  # 还没进入范围判定为负

        # 被对面刀了直接判负
        if dist < 10e3 and enm_state["target_information"][4]<pi/6:
            terminate = True
            self.lose = 1

        # # 导弹打光没干掉对面直接判负
        # if ego.ammo==0 and len(alive_ally_missiles)==0 and not enm.dead:
        #     terminate = True
        #     self.lose = 1

        # 命中判断
        if enm.dead or self.out_range(enm): # 驱赶也行， 新增
            terminate = True
            self.win = 1

        if enm.dead:
            self.target_hit = 1
        if self.out_range(enm):
            self.target_out = 1

        if self.win and self.lose:
            self.draw = 1

        # reward_base = 100 / (self.game_time_limit/dt_maneuver)  # 防自杀奖励

        # # 目标进入角奖励，当目标从进攻转逃逸时就给奖励
        # reward_AA = enm_state["target_information"][4]/pi * 1

        # 发射惩罚，根据 missile_time_since_shoot
        reward_shoot = 0
        if ut == 1:
            reward_shoot += np.clip((missile_time_since_shoot-30)/30, -1,1)  # 过30s发射就可以奖励了
            reward_shoot += 0.5 * abs(AA_hor)/pi-0.5  # 要把敌人骗进来杀， 进入角为负的时候可以暂缓射击
            reward_shoot -= np.clip(dist/40e3, 0, 1)

        if dist <= 20e3 and ego.ammo==6:
            reward_shoot -= 100 # 一发都不打必须重罚
        if terminate and ego.ammo<6:
            reward_shoot += 20 # 至少打了一枚

        # 重复发射导弹时惩罚, 否则有奖励
        reward_SuoHa = 0
        if len(alive_ally_missiles)>1 and ut==1: # state["missile_in_mid_term"] and ut==1:
            reward_SuoHa -= 30
        if len(alive_ally_missiles)>1 and ut==0: # state["missile_in_mid_term"] and ut==0:
            reward_SuoHa += 30

        # miss 惩罚
        reward_miss = 0
        if enm.escape_once:
            reward_miss -= 60 # 10

        # 命中奖励
        end_reward = 0
        reward_event = 0
        if self.lose:
            reward_event = -300
            
        if self.win:
            reward_event = 300 # + 200*(6-ego.ammo)/6  ## 赢了，导弹省得越多奖励越高 test 300
            end_reward = 300

        # 0.2? 0.02?
        reward = np.sum([
            # 1 * reward_base,
            # 1 * reward_AA,
            1 * reward_shoot,
            1 * reward_SuoHa,
            # 1 * reward_violate,
            1 * reward_miss,
            1 * reward_event,
        ])

        if terminate:
            self.running = False

        return terminate, reward, end_reward


def shoot_action_shield(at, distance, alpha, AA_hor, launch_interval):
    at0 = at
    # if distance > 60e3:
    #     interval_refer = 30
    # elif distance>40e3:
    #     interval_refer = 20
    # elif distance>20e3:
    #     interval_refer = 15
    # else:
    #     interval_refer = 8

    interval_refer = 0.1  # 8

    # todo 对方在我射界内向我发射了一枚导弹，我也应该向对方来一枚
    
    # # test
    # if abs(distance-55e3) < 1e3 and alt>8000:
    #     at = 1


    # if distance<30e3 and abs(AA_hor)>pi/2:
    #     at = 1

    # if distance>20e3:
    #     interval_refer = 15
    # elif distance>10e3:
    #     interval_refer = 10
    # elif distance < 10e3:
    #     interval_refer = 5
    
    if distance > 80e3 or alpha > 60*pi/180:
        at = 0
    # if distance < 10e3 and alpha < pi/12 and abs(AA_hor) > pi*3/4 and launch_interval>30:
    #     at = 1
    if launch_interval <= interval_refer:
        at = 0

    # if abs(AA_hor) < pi*1/3 and distance>12e3: ## 禁止超视距完全尾追发射 新增
    #     at=0

    same = int(bool(at0) == bool(at))
    xor  = int(bool(at0) != bool(at))  

    return at, xor
