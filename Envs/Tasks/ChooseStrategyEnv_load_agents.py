'''
出生点改在外面指定

子策略暂时使用规则智能体，留下使用神经网络的接口

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
# from Algorithms.PPOcontinues_std_no_state_with_truncs import *
# 更改导入，只导入需要的类和函数
from Algorithms.PPOcontinues_std_no_state_with_truncs import PolicyNetContinuous, take_action_from_policy


from Utilities.LocateDirAndAgents import *

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
attack_obs_dim = 8 + 7 + 2
crank_obs_dim = 1+8+7+4+2
escape_obs_dim = 1 * 2 + 8 + 7 + 4 + 2
hidden_dim = [128, 128, 128]  # 128
action_dim = 3
# 由于不再实例化PPO，以下参数仅为参考，不再直接使用
# gamma = 0.95 # 0.9
# lmbda = 0.95 # 0.9
# epochs = 10
# eps = 0.2
# actor_lr=0
# critic_lr=0

# 仅实例化策略网络 (PolicyNetContinuous)
attack_actor = PolicyNetContinuous(attack_obs_dim, hidden_dim, action_dim).to(device)
Lcrank_actor = PolicyNetContinuous(crank_obs_dim, hidden_dim, action_dim).to(device)
Rcrank_actor = PolicyNetContinuous(crank_obs_dim, hidden_dim, action_dim).to(device)
escape_actor = PolicyNetContinuous(escape_obs_dim, hidden_dim, action_dim).to(device)


# 规范化 agents 目录路径并确保目录存在（避免 './' 出现在路径中）
actors_dir = os.path.abspath(os.path.join(current_dir, "Tasks", "Current_Agents"))
os.makedirs(actors_dir, exist_ok=True)

# 正确拼接文件名：不要在文件名前加路径分隔符（'\\' 或 '/'），否则会被认为是绝对路径
attack_actor_dir = os.path.abspath(os.path.join(actors_dir, "AttackActor.pt"))
escape_actor_dir = os.path.abspath(os.path.join(actors_dir, "EscapeActor.pt"))
Lcrank_actor_dir = os.path.abspath(os.path.join(actors_dir, "LCrankActor.pt"))
Rcrank_actor_dir = os.path.abspath(os.path.join(actors_dir, "RCrankActor.pt"))

# 加载模型参数
attack_actor.load_state_dict(torch.load(attack_actor_dir, map_location=device, weights_only=True))
escape_actor.load_state_dict(torch.load(escape_actor_dir, map_location=device, weights_only=True))
Lcrank_actor.load_state_dict(torch.load(Lcrank_actor_dir, map_location=device, weights_only=True))
Rcrank_actor.load_state_dict(torch.load(Rcrank_actor_dir, map_location=device, weights_only=True))


# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

action_options = [
                    "attack",
                    "escape",
                    "left",
                    "right",
                ]

class ChooseStrategyEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.attack_key_order = [
            "target_information",  # 8
            "ego_main",  # 7
            "border",  # 2
        ]
        self.crank_key_order = [
            "target_locked",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "threat",  # 4
            "border",  # 2
        ]
        self.escape_key_order = [
            "locked_by_target",  # 1
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "threat",  # 4
            "border",  # 2
        ]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        self.last_obs = None

    # 进攻策略观测量
    def attack_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.attack_key_order}
        full_obs["ego_main"][6]=0 # ammo
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.attack_key_order)
        return flat_obs, full_obs
    def crank_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.crank_key_order}
        full_obs["ego_main"][6]=0 # ammo
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.crank_key_order)
        return flat_obs, full_obs
    def escape_obs(self, side):
        pre_full_obs = self.base_obs(side)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.escape_key_order}
        full_obs["ego_main"][6] = 0 # ammo
        full_obs["locked_by_target"] = 1
        # full_obs["target_information"]
        flat_obs = flatten_obs(full_obs, self.escape_key_order)
        return flat_obs, full_obs
        

    def step(self, r_actions, b_actions, rules_or_agents=False):
        # 这一层的action是离散的动作类型

        # 对每个动作类型，调用子策略产生连续动作指令值
        report_move_time_rate = int(round(self.dt_maneuver / dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]
        self.r_actions = r_actions  # .copy()
        self.b_actions = b_actions  # .copy()

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中
        UAV_actions = {}
        for UAV, action in zip(self.UAVs, actions):

            if UAV.dead:
                continue
            elif action == 0:
                obs, check_obs = self.attack_obs(UAV.side)
                base_obs = self.base_obs(UAV.side)
                delta_theta_obs = check_obs["target_information"][2]
                delta_theta_actual = base_obs["target_information"][2]
                distance = check_obs["target_information"][3]
                distance_actual = base_obs["target_information"][3]
                speed = check_obs["ego_main"][0]
                speed_actual = base_obs["ego_main"][0]
                alt = check_obs["ego_main"][1]
                alt_actual = base_obs["ego_main"][1]
                delta_psi_obs = atan2(check_obs["target_information"][1], check_obs["target_information"][0])
                delta_psi_actual = atan2(base_obs["target_information"][1], base_obs["target_information"][0])

                if delta_theta_obs!=delta_theta_actual or distance!=distance_actual or speed!=speed_actual or\
                    alt!=alt_actual or delta_psi_obs != delta_psi_actual:
                    print('信息获取错误')


                move_action = take_action_from_policy(attack_actor, obs, action_bound, device)
                # print(UAV.side)
                # print(move_action)
                1+1
            elif action == 1:
                obs, check_obs = self.escape_obs(UAV.side)
                move_action = take_action_from_policy(escape_actor, obs, action_bound, device)
            elif action == 2:
                obs, check_obs = self.crank_obs(UAV.side)
                move_action = take_action_from_policy(Lcrank_actor, obs, action_bound, device)
            elif action == 3:
                obs, check_obs = self.crank_obs(UAV.side)
                move_action = take_action_from_policy(Rcrank_actor, obs, action_bound, device)
            else:
                move_action = np.zeros(3)
            UAV_actions[UAV.id] = move_action
            

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                uav_obs = self.base_obs(UAV.side, pomdp=0)  ### test 部分观测的话用1
                delta_theta = uav_obs["target_information"][2]
                distance = uav_obs["target_information"][3] * 10e3
                d_hor, leftright = uav_obs["border"]
                # state = self.get_state(UAV.side)
                speed = uav_obs["ego_main"][0]
                alt = uav_obs["ego_main"][1]
                cos_delta_psi = uav_obs["target_information"][0]
                sin_delta_psi = uav_obs["target_information"][1]
                delta_psi = atan2(sin_delta_psi, cos_delta_psi)

                # move_action = np.zeros(3)

                # # 进攻机动
                # if action == 0:
                #     # 调用进攻策略
                #     # # 规则智能体
                #     # move_action = self.track_behavior(UAV.alt, delta_psi)

                #     # 训练出的智能体
                #     obs, _ = self.attack_obs(UAV.side)
                #     move_action, _ = attack_agent.take_action(obs, action_bound, explore=False)

                #     # 武器发射决策
                #     pass

                # if action == 1:
                #     # 调用escape策略
                #     # # 规则智能体
                #     cos_threat_psi = uav_obs["threat"][0]
                #     sin_threat_psi = uav_obs["threat"][1]
                #     threat_psi = atan2(sin_threat_psi, cos_threat_psi)
                #     move_action = self.escape_behavior(UAV.alt, delta_psi, uav_obs["warning"], threat_psi)

                #     # 训练出的智能体
                #     obs, _ = self.escape_obs(UAV.side)
                #     move_action, _ = escape_agent.take_action(obs, action_bound, explore=False)

                # if action == 2:
                #     # 调用Lcrank策略
                #     # # 规则智能体
                #     # move_action = self.left_crank_behavior(UAV.alt, delta_psi)

                #     # 训练出的智能体
                #     obs, _ = self.crank_obs(UAV.side)
                #     move_action, _ = Lcrank_agent.take_action(obs, action_bound, explore=False)

                # if action == 3:
                #     # 调用Rcrank策略
                #     # # 规则智能体
                #     # move_action = self.right_crank_behavior(UAV.alt, delta_psi)

                #     # 训练出的智能体
                #     obs, _ = self.crank_obs(UAV.side)
                #     move_action, _ = Rcrank_agent.take_action(obs, action_bound, explore=False)

                # if action == 4:
                #     # 转圈搜索目标
                #     # 该部分不训练智能体，改为使用规则智能体
                #     move_action[0] = np.clip(8000 - UAV.alt, self.min_alt_safe - UAV.alt,
                #                              self.max_alt_safe - UAV.alt) * 5000

                #     if leftright >= 0:
                #         move_action[1] = pi
                #     else:
                #         move_action[1] = -pi
                #     move_action[2] = 1.0 * 340

                move_action = UAV_actions[UAV.id]



                # # 避让水平边界动作修正：
                # # print(UAV.side)
                # state = self.get_state(UAV.side)
                # dist_2_hor = state["border"][0]
                # move_action = self.back_in_cage(move_action, UAV.pos_, UAV.psi)

                # # if UAV.side == 'r':
                # #     print(move_action)
                # #     print()
                # pass

                UAV.move(move_action[0], move_action[1], move_action[2], relevant_height=True)

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
                if self.out_range(UAV):
                    UAV.dead = True
                    # self.running = False

        # r_reward_n, b_reward_n = self.get_reward()
        # terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
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

        # if terminate:
        #     self.running = False

        # return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def combat_terminate_and_reward(self, side, action_label):
        # terminate = self.get_terminate()
        # done = terminate

        # todo 奖励函数调用或是重写都要在这实现
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

            # 如果超时，双方均未打光导弹/仍有导弹在空中飞，且双方均存活，判定为平
            else:
                self.draw = 1

        reward = 0
        event_reward = 0
        if self.win:
            reward += 300
        if self.lose:
            reward -= 300
        if self.draw:
            reward -= 0

        uav_states = self.get_state(side)
        enm_state = self.get_state(enm.side)
        delta_theta = uav_states["target_information"][2]
        distance = uav_states["target_information"][3]
        d_hor, leftright = uav_states["border"]
        # state = self.get_state(UAV.side)
        speed = uav_states["ego_main"][0]
        alt = uav_states["ego_main"][1]
        cos_delta_psi = uav_states["target_information"][0]
        sin_delta_psi = uav_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = uav_states["target_information"][4]
        warning = uav_states["warning"]
        missile_in_mid_term = uav_states["missile_in_mid_term"]
        missile_time_since_shoot = uav_states["weapon"]
        AA_hor = uav_states["target_information"][-2]
        
        cos_delta_psi_threat = uav_states["threat"][0]
        sin_delta_psi_threat = uav_states["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)


        # 导弹锁定目标
        if enm_state["warning"]:
            reward += 3

        # 逃脱导弹
        if ego.escape_once:
            reward += 10

        # 导弹被逃脱
        if enm.escape_once:
            reward -= 5

        if not ego.dead: # 以下是存活才能获取的奖励
            # 事件/密集奖励
            # 为导弹提供制导
            if missile_in_mid_term:
                reward += 2

            # 锁定目标
            if uav_states["target_locked"]:
                reward += 1.1

            # 被目标锁定
            if uav_states["locked_by_target"]:
                reward -= 1

            # 收到导弹警告
            if warning:
                reward -= 3

            # # 密集奖励
            if warning and action_label != 1 and len(alive_ally_missiles)>0:
                reward -= 5

            # 没发射导弹时alpha越小越好
            if len(alive_ally_missiles) == 0 and not warning:
                reward += 0.5 - alpha / pi
            
            # 导弹处于中制导阶段时alpha在±60*pi/180之间为奖励高台， 其余随alpha线性减少
            if missile_in_mid_term:
                # 目标没有跑，不该重复攻击
                if action_label==0 and abs(AA_hor)>pi*2/3:
                    reward -= 5

                # 高台奖励：alpha在[-60*pi/180, 60*pi/180]区间奖励高，其余线性递减
                if abs(alpha) < np.pi / 3:
                    reward += 1
                else:
                    reward += 1 - (alpha - pi / 3) / pi

            # 有warning时alpha越大越好
            if warning:
                reward += 8 * abs(delta_psi_threat) / pi # 这里的alpha错了，应该是和导弹的threa_delta_psi有关的
            
            # deltapsi变化率奖励 todobedontinued

            # 态势优势度 tobecontinued


        return done, reward, event_reward

    def is_action_complete(self, side, action_label):
        # if side == 'r':
        #     ego = self.RUAV
        #     enm = self.BUAV
        # if side == 'b':
        #     ego = self.BUAV
        #     enm = self.RUAV

        state = self.get_state(side)
        cos_delta_psi = state["target_information"][0]
        sin_delta_psi = state["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_theta = state["target_information"][2]
        alpha = state["target_information"][4]
        RWR = state["warning"]
        cos_delta_psi_threat = state["threat"][0]
        sin_delta_psi_threat = state["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)

        action = action_options[action_label]

        action_done = False
        if action == 'attack':
            if alpha < pi/6:
                action_done = True
        
        if action == 'escape':
            if abs(delta_psi_threat) > pi*5/6 and RWR or\
                  abs(alpha) > pi*5/6 and not RWR:
                action_done = True
        
        if action == 'left':
            if abs(delta_psi*180/pi - 50) < 10:
                action_done = True
        
        if action == 'right':
            if abs(delta_psi*180/pi + 50) < 10:
                action_done = True
            
        return action_done


