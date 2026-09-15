from numpy import argmin
import numpy as np
from math import *
import torch
import argparse
from numpy.linalg import norm
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
import re

use_tacview = 2  # 静默写入 acmi_records
action_cycle_multiplier = 10

def basic_rules(rule_inputs, mask_inputs, rules_num, p_random=0):
    '''
    rules_num = 0: 保持和目标相同高度只打进攻(maneuver_cmds = 0,1,3)，0平飞追踪 1爬升追踪 3下降追踪，攻击区内发射导弹，上一枚导弹发射后如果还在中制导，不发射新导弹
    rules_num = 1: 保持和目标相同高度进攻(0,1,3), 发射完导弹立马crank(5,6), 受到威胁立刻回转至5000m高度以下(11水平回转, 12俯冲回转), 威胁结束后回归进攻
    rules_num = 2: 保持和目标相同高度打首轮进攻(0,1,3), 在距离40km以外先爬升60°(2), 一个决策回合后射击, 否则直接射击，设计后立刻crank(6)，
        收到威胁立刻splitS(8), 威胁解除后回转进攻(0,1,3)
    '''

    # 观测信息解包
    mask_ally = mask_inputs['ally']
    mask_enm = mask_inputs['enm']

    num_mask_enm = len(mask_enm)

    # 本机信息
    speed = rule_inputs["ego_main"][0]
    alt = rule_inputs["ego_main"][1]
    sin_theta = rule_inputs["ego_main"][3]

    # 地图信息
    d_hor, leftright, d2center = rule_inputs["border"]

    # 队友信息略过
    ally_info_list = rule_inputs["ally_infos"]

    # 敌机信息
    enm_info_list = np.array(rule_inputs["enm_infos"])
    enm_distance_list = enm_info_list[:, 8]
    enm_delta_theta_list = enm_info_list[:, 7]
    enm_delta_psi_list = np.arctan2(enm_info_list[:, 6], enm_info_list[:, 5])
    enm_delta_alt_list = enm_distance_list * np.sin(enm_delta_theta_list) # 相对高度
    enm_ATA_list = enm_info_list[:, 9]
    AA_hor_list = enm_info_list[:, 11]
    enm_locked_list = enm_info_list[:, 3]

    # 武器信息
    weapon_infos = rule_inputs["weapon_infos"]
    t_fired = weapon_infos[0] # 导弹发射计时
    on_guiding = weapon_infos[3] # 中制导状态标志

    # 告警信息
    RWR_infos = rule_inputs["RWR_infos"]
    RWR_list = np.array(RWR_infos[0]) # 告警灯标志
    threat_delta_psi_list = np.arctan2(RWR_infos[2], RWR_infos[1])
    threat_distance_list = np.array(RWR_infos[4]) # 威胁距离
    
    if RWR_list.any():
        RWR = True
        threat_distance = np.min(threat_distance_list[RWR_list > 0])
    else:
        RWR = False
        threat_distance = 999e3

    # 不同策略开始逃跑的触发距离
    escape_distance_list = np.array([
                                    0, # 0
                                    45, # 1
                                    45, # 2
                                    45, # 3
                                    45, # 4
                                    45, # 5
                                    ]) * 1e3

    # 手动“注意力”，机动的时候只注意最近的活着的敌机，开火只管选中的敌机，雷达全开
    valid_index = []
    selected_enm_index = None
    current_min_distance = 200e3
    for i in range(len(mask_enm)):
        if mask_enm[i]:
            valid_index.append(i)
            if enm_distance_list[i] < current_min_distance:
                selected_enm_index = i
                current_min_distance = enm_distance_list[i]
    
    if selected_enm_index is None:
        selected_index = num_mask_enm + 1
        if RWR and threat_distance < escape_distance_list[rules_num]: # 受到威胁
            # 优先俯冲回转至5000m以下
            if alt > 5000:
                action_v = 4 # -60度俯冲
            else:
                action_v = 2 # 平飞

            # 置尾机动
            action_h = 6 # 水平方向背离

            # # 三九线机动
            # if delta_psi < 0:
            #     action_h = 5 # 9线
            # else:
            #     action_h = 4 # 3线

        else:
            action_v = 2 # 平飞
            action_h = 0 # 回中心
            # maneuver_cmds = 7

        flight_action = {'do': [action_v, action_h], 'at': selected_index}  
        # flight_action = {'do': maneuver_cmds, 'at': selected_index}
        fire_action = {'do': False, 'at': selected_index}
        radar_action = 1
        return flight_action, fire_action, radar_action

    selected_index = selected_enm_index
    
    # 提取目标信息
    distance = enm_distance_list[selected_index]
    ATA = enm_ATA_list[selected_index]
    delta_psi = enm_delta_psi_list[selected_index]
    delta_theta = enm_delta_theta_list[selected_index]
    AA_hor = AA_hor_list[selected_index]

    # 1. 计算初始的开火意图
    fire_120 = False
    case1 = distance < 0.7*70e3 and (sin_theta >= sin(30*pi/180) or alt >= 9e3) # 105, 30
    case2 = distance < 0.7*60e3 # 75e3
    if (case1 or case2) and ATA < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
        if t_fired >= 40 and not on_guiding and not (distance>12e3 and abs(AA_hor) < 30*pi/180):
            fire_120 = True

    action_v = 2 # 追踪
    action_h = 0 # 追踪
    maneuver_cmds = [action_v, action_h] # 默认执行基础进攻
    base_offensive_action = [action_v, action_h]

    # 3. 根据规则编号(rules_num)决定最终机动和开火决策
    if rules_num == 0:
        # 规则0: 纯进攻
        maneuver_cmds = base_offensive_action
        fire_missile_affirmative = fire_120

    elif rules_num == 1:
        # 规则1: 带防御机动
        if RWR: # 受到威胁
            # 优先俯冲回转至5000m以下
            if alt > 5000:
                action_v = 4 # 快速下高
            else:
                action_v = 2 # 平飞
            # 置尾机动
            action_h = 3 # 水平方向背离
            
        elif on_guiding: # 如果本回合决定发射导弹
            if delta_psi < 0:
                action_v = 2 # 平飞
                action_h = 5 # Rcrank
            else:
                action_v = 2 # 平飞
                action_h = 1 # Lcrank
        else:
            [action_v, action_h] = base_offensive_action
        fire_missile_affirmative = fire_120
        maneuver_cmds = [action_v, action_h]

    elif rules_num == 2:
        # 规则2: Loft爬升射击序列
        if RWR: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            maneuver_cmds = [action_v, action_h]
            # maneuver_cmds = 8 # 立刻 split-S
            fire_120 = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离，或上一回合是爬升
            if delta_psi < 0:
                action_v = 2 # 平飞
                action_h = 5 # Rcrank
                maneuver_cmds = [action_v, action_h]
                # maneuver_cmds = 6 # 右crank
            else:
                action_v = 2 # 平飞
                action_h = 1 # Lcrank
                maneuver_cmds = [action_v, action_h]
                # maneuver_cmds = 5 # 左crank
            # maneuver_cmds = 6 if delta_psi < 0 else 5 # random.choice([5,6]) # 立刻crank
        elif fire_120 and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:  # last_action != 2: # 如果上一动作为非爬升
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                # maneuver_cmds = 2 # 则本回合执行爬升
                fire_120 = False
                maneuver_cmds = [action_v, action_h]
            else:
                fire_120 = True
        else:
            maneuver_cmds = base_offensive_action
        fire_missile_affirmative = fire_120

    elif rules_num == 3:
        if RWR: # : # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            maneuver_cmds = [action_v, action_h]
            fire_120 = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离，或上一回合是爬升
            if delta_psi < 0:
                action_v = 3 # 下降
                action_h = 5 # Rcrank
                maneuver_cmds = [action_v, action_h]
            else:
                action_v = 3 # 下降
                action_h = 1 # Lcrank
                maneuver_cmds = [action_v, action_h]
        elif fire_120 and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:  # last_action != 2: # 如果上一动作为非爬升
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_120 = False
                maneuver_cmds = [action_v, action_h]
            else:
                fire_120 = True
        else:
            maneuver_cmds = base_offensive_action
        fire_missile_affirmative = fire_120
    
    elif rules_num == 4:
        # 规则4: 只知道进攻和防御，不知道crank
        if RWR: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            maneuver_cmds = [action_v, action_h]
            fire_120 = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离
            action_v = 2 # 平飞
            action_h = 0 # 追踪，不crank
            maneuver_cmds = [action_v, action_h]
        elif fire_120 and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_120 = False
                maneuver_cmds = [action_v, action_h]
            else:
                fire_120 = True
        else:
            maneuver_cmds = base_offensive_action
        fire_missile_affirmative = fire_120

    elif rules_num == 5:
        # 规则5: 在rule2基础上，leftright>0时左crank，否则右crank
        if RWR: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            maneuver_cmds = [action_v, action_h]
            fire_120 = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离
            if delta_psi < 0:
                action_v = 3 # 下降
                action_h = 5 # Rcrank
                maneuver_cmds = [action_v, action_h]
            else:
                action_v = 3 # 下降
                action_h = 1 # Lcrank
                maneuver_cmds = [action_v, action_h]
            # 保留战略纵深
            if d_hor < 50e3:
                if leftright > 0:
                    action_v = 2 # 平飞
                    action_h = 1 # Lcrank
                    maneuver_cmds = [action_v, action_h]
                if leftright < 0:
                    action_v = 2 # 平飞
                    action_h = 5 # Rcrank
                maneuver_cmds = [action_v, action_h]
        elif fire_120 and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_120 = False
                maneuver_cmds = [action_v, action_h]
            else:
                fire_120 = True
        else:
            maneuver_cmds = base_offensive_action
        fire_missile_affirmative = fire_120
        
    # if  np.random.rand() <= p_random:
    #     maneuver_cmds = np.random.randint(0, 13+1)
    #     maneuver_cmds = np.clip(maneuver_cmds, 0, 13)
    #     # # 不准因为随机行为而出界，效果不好，改回
    #     # if d_hor < 8e3:
    #     #     maneuver_cmds = base_offensive_action
    
    # if rules_num in [3, 4]:
    #     # 不准出界
    #     if d_hor < 8e3:
    #         maneuver_cmds = base_offensive_action
    
    maneuver_cmds = [action_v, action_h]
    flight_action = {'do': maneuver_cmds, 'at': selected_index}
    fire_action = {'do': fire_missile_affirmative,'at': selected_index}
    radar_action = 1 # 暂不考虑雷达开关

    return flight_action, fire_action, radar_action

"""
action_optionsLR = {
                    0: "track",
                    1: "30track",
                    2: "60track",
                    3: "-30track",
                    4: "-60track",
                    5: "L60crank",
                    6: "R60crank",
                    7: "snake",   "notch"
                    8: "splitS",
                    9: "3",     "left
                    10: "9",    "right
                    11: "fastTurn",
                    12: "-30turn",
                    13: "-60turn",
                }
"""

if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    # 在这里调用规则(编号)下的策略
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # # Environment
    # parser.add_argument("--max-episode-len", type=float, default=15*60,  # 8 * 60,
    #                     help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--R-cage", type=float, default=71e3,
    #                     help="")
    args = parser.parse_args()
    
    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]
    env = ChooseStrategyEnv(args, tacview_show=use_tacview, vertices=vertices)
    # test
    env.dt_move = 0.1 # 0.04 # 4 # 0.025 0.02

    device = torch.device("cpu")  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def creat_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 55e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    steps_count = 0

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    decide_steps_after_update = 0
    try:
        r_action_list = []
        b_action_list = []
        r_guide_list = []
        b_guide_list = []
        
        # 采集不同轨迹的动作
        for i_episode in range(1): # 5

            episode_return = 0
            # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],}

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=4, blue_init_ammo=4)
            r_action_label=0
            b_action_label=0
            last_decision_state = None
            
            b_reward = None

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()

            fly_action_all = {}
            fire_action_all = {}
            radar_action_all = {}

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 回合结束判断
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                agent_inputs_ = {}
                mask_inputs_ = {}
                rule_inputs_ = {}

                for id in env.UAV_ids:
                    agent_input, mask_input, rule_input  = env.get_obs(id, pomdp=0)
                    agent_inputs_[id] = agent_input
                    mask_inputs_[id] = mask_input
                    rule_inputs_[id] = rule_input

                # 记忆逻辑改到环境端进行，算法端不再处理原始数据的记忆

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了

                    # if steps_of_this_eps > 0:
                    #     transition_dict['states'].append(last_decision_state)

                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    # last_decision_obs = b_obs
                    
                    for id in env.UAV_ids:
                        
                        # debug
                        UAV = env.find_uav_with_id(id)
                        side_word = 'red' if UAV.side == 'r' else 'blue'
                        num_word = id if UAV.side == 'r' else id-200

                        # print("我是", side_word, num_word)
                        
                        # 2. Agent 产生一个动作
                        if 1: # 如果类型是规则智能体而非神经网络
                            fly_action, fire_action, radar_action = basic_rules(rule_inputs_[id], mask_inputs_[id], rules_num=5)
                        else:
                            # 如果类型是神经网络
                            pass
                        
                        fly_action_all[id] = fly_action.copy()
                        fire_action_all[id] = fire_action.copy()
                        radar_action_all[id] = radar_action

                        # 暂不考虑同时攻击多个目标
                        if fire_action['do']:
                            target_index = fire_action['at']
                            env.fire_120(id, target_index, tabu=0)
                            
                        # for i, fire in enumerate(fire_action['do']):
                        #     if fire:
                        #         target_index = fire_action['at'][i]
                        #         env.fire_120(id, target_index, tabu=0)
                        
                        # 必须确认雷达开关状态
                        env.radar_action(id, radar_action)

                        # debug
                        # if side_word == 'red' and \
                        #     (
                        #     # 47 < env.t < 64 or\
                        #         3*60 < env.t < 3*60+12
                        #     ):
                        #     print('id', UAV.id)
                        #     print('enm_infos', rule_inputs_[id]['enm_infos'][fly_action['at']])
                        #     print('weapon_infos', rule_inputs_[id]['weapon_infos'])
                        #     print('RWR_infos', rule_inputs_[id]['RWR_infos'])
                        #     print('机动动作是：', fly_action)
                        #     print('开火动作是:', fire_action)
                        #     print()

                    decide_steps_after_update += 1

                    # b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    # current_action = b_action_label

                        # # debug
                        # if env.t > 40:
                        #     print("r_state_check", r_state_check["warning"])
                        #     print("r_action_label", r_action_label)
                        #     print()
                        #     print("b_state_check", b_state_check["warning"])
                        #     print("b_action_label", b_action_label)
                        #     print()
                
                # 循环调用底层控制器
                flight_cmds = {}
                for id in env.UAV_ids:
                    fly_action = fly_action_all[id]
                    fire_action = fire_action_all[id]
                    radar_action = radar_action_all[id]

                    maneuver_cmds = fly_action['do']
                    target_index = fly_action['at']
                    flight_cmds[id] = env.maneuver_for_1_enm(id, target_index, maneuver_cmds, rule_inputs_[id])
                    # # debug
                    # if id == 1:
                    #     print("target_index", target_index)
                    #     print("maneuver_cmds", maneuver_cmds)
                    #     print("rule_inputs_", rule_inputs_[id]['enm_infos'][target_index])
                    #     print("flight_cmds", flight_cmds[id])

                env.step(flight_cmds.values()) # Environment updates every dt_maneuver
                done, _, _, _ = env.combat_terminate_and_reward('r', None, action_cycle_multiplier)

                # b_reward = b_reward_event + b_reward_constraint + b_reward_shaping

                # # Accumulate rewards between agent decisions
                # episode_return += b_reward * env.dt_maneuver

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            episode_end_time = time.time()  # 记录结束时间

            env.clear_render(t_bias=t_bias)
            t_bias += env.t


        training_end_time = time.time()  # 记录结束时间
        


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送

