import numpy as np
from math import *
import torch
import argparse
from numpy.linalg import norm
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)



from Envs.battle6dof1v1_missile0919 import launch_missile_if_possible
from Envs.Tasks.ChooseStrategyEnv2_2 import * # 三元组奖励
from Math_calculates.sub_of_angles import *
import re

use_tacview = 1  # 是否可视化
action_cycle_multiplier = 30

def basic_rules(state_check, rules_num, last_action=0):
    '''
    rules_num = 0: 保持和目标相同高度只打进攻(action_number = 0,1,3)，0平飞追踪 1爬升追踪 3下降追踪，攻击区内发射导弹，上一枚导弹发射后如果还在中制导，不发射新导弹
    rules_num = 1: 保持和目标相同高度进攻(0,1,3), 发射完导弹立马crank(5,6), 受到威胁立刻回转至5000m高度以下(11水平回转, 12俯冲回转), 威胁结束后回归进攻
    rules_num = 2: 保持和目标相同高度打首轮进攻(0,1,3), 在距离40km以外先爬升60°(2), 一个决策回合后射击, 否则直接射击，设计后立刻crank(6)，
        收到威胁立刻splitS(8), 威胁解除后回转进攻(0,1,3)
    '''

    delta_theta = state_check["target_information"][2] # 目标相对俯仰角
    distance = state_check["target_information"][3] # 距离
    delta_alt = distance*sin(delta_theta)  # 目标相对高度
    d_hor, leftright = state_check["border"]
    speed = state_check["ego_main"][0]
    alt = state_check["ego_main"][1] # 我机高度
    cos_delta_psi = state_check["target_information"][0]
    sin_delta_psi = state_check["target_information"][1]
    delta_psi = atan2(sin_delta_psi, cos_delta_psi)
    delta_psi_threat = atan2(state_check["threat"][1], state_check["threat"][0])
    RWR = state_check["warning"] # 受到威胁标志
    on_guiding = state_check["missile_in_mid_term"] # 中制导状态标志
    t_fired = state_check["weapon"] # 导弹发射后计时，<12s不允许发射新导弹
    ATA = state_check["target_information"][4]
    AA_hor = state_check["target_information"][6]

    # 1. 计算初始的开火意图
    fire_missile = False
    if distance < 60e3 and ATA < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
        if t_fired >= 12 and not on_guiding and not (distance>12e3 and abs(AA_hor) < 30*pi/180):
            fire_missile = True

    # # 2. 根据目标相对高度选择基础进攻机动
    
    if abs(delta_theta) < pi/6:
        base_offensive_action = 0  # 平飞追踪
    elif delta_theta >= pi/6:
        base_offensive_action = 1  # 爬升追踪
    else: # delta_theta < -pi/6
        base_offensive_action = 3  # 下降追踪

    action_number = base_offensive_action # 默认执行基础进攻

    # 3. 根据规则编号(rules_num)决定最终机动和开火决策
    if rules_num == 0:
        # 规则0: 纯进攻
        action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    elif rules_num in [1, 3]:
        # 规则1: 带防御机动
        if RWR: # 受到威胁
            # 优先俯冲回转至5000m以下
            if alt > 5000:
                action_number = 12 # 俯冲回转
            else:
                action_number = 11 # 水平回转
            # fire_missile = False # 防御时不发射
        elif on_guiding: # 如果本回合决定发射导弹
            action_number = 6 if delta_psi < 0 else 5 # random.choice([5,6]) # 立刻crank
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    elif rules_num in [2, 4]:
        # 规则2: Loft爬升射击序列
        if RWR: # 受到威胁
            action_number = 8 # 立刻 split-S
            fire_missile = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离，或上一回合是爬升
            action_number = 6 if delta_psi < 0 else 5 # random.choice([5,6]) # 立刻crank
        elif fire_missile and distance > 40e3: # 满足开火条件且在远距离
            if last_action != 2: # 如果上一动作为非爬升
                action_number = 2 # 则本回合执行爬升
                fire_missile = False
            else:
                fire_missile = True
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    # if fire_missile_affirmative:
    #     launch_missile_immediately(env, side)
    
    if rules_num in [3, 4]:
        # 不准出界
        if d_hor < 8e3:
            action_number = base_offensive_action

    return action_number, fire_missile_affirmative


if __name__=='__main__':
    
    print("\n根目录为：", project_root, "\n")
    
    # 在这里调用规则(编号)下的策略
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=10*60,  # 8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=55e3,  # 70e3 还是太大了
                        help="")
    args = parser.parse_args()

    env = ChooseStrategyEnv(args, tacview_show=use_tacview)
    # test
    env.dt_move = 0.05

    env.shielded = 1

    r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def creat_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 45e3
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
        
        # 采集不同轨迹的动作
        for i_episode in range(5):

            last_r_action_label = 0
            last_b_action_label = 0

            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],}

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            r_action_label=0
            b_action_label=0
            last_decision_state = None
            current_action = None
            b_reward = None

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 回合结束判断
                # print(env.running)
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                # 在这里将观测信息压入记忆
                # env.RUAV.obs_memory = r_check_obs.copy()
                # env.BUAV.obs_memory = b_check_obs.copy()

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    # if steps_of_this_eps > 0:
                    #     transition_dict['states'].append(last_decision_state)

                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作

                    # 红方根据规则活动
                    r_state_check = env.unscale_state(r_check_obs)
                    r_action_label, r_fire = basic_rules(r_state_check, i_episode, last_action=last_r_action_label)
                    last_r_action_label = r_action_label
                    if r_fire:
                        launch_missile_immediately(env, 'r')

                    # 蓝方根据规则活动
                    b_state_check = env.unscale_state(b_check_obs)
                    b_action_label, b_fire = basic_rules(b_state_check, 2, last_action=last_b_action_label)
                    last_b_action_label = b_action_label
                    if b_fire:
                        launch_missile_immediately(env, 'b')

                    decide_steps_after_update += 1
                    
                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    current_action = b_action_label

                r_action = env.maneuver14LR(env.RUAV, r_action_label)
                b_action = env.maneuver14LR(env.BUAV, b_action_label)

                env.step(r_action, b_action) # Environment updates every dt_maneuver
                done, b_reward_event, b_reward_constraint, b_reward_shaping = env.combat_terminate_and_reward('b', b_action_label, b_fire)
                b_reward = b_reward_event + b_reward_constraint + b_reward_shaping

                # Accumulate rewards between agent decisions
                episode_return += b_reward * env.dt_maneuver

                next_b_check_obs = env.base_obs('b')
                next_b_obs = flatten_obs(next_b_check_obs, env.key_order)


                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            
            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            r_action_list = np.array(r_action_list)
            # b_action_list is no longer appended every dt_maneuver, need to rethink if you need this for logging


        training_end_time = time.time()  # 记录结束时间
        


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送

        b_action_arrays = np.array(b_action_list)

        import matplotlib.pyplot as plt

        if b_action_arrays.size == 0:
            print("b_action_arrays is empty, nothing to plot.")
        else:
            x = b_action_arrays[:, 0].astype(float)
            y = b_action_arrays[:, 1].astype(float)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, y, marker='o', linestyle='-')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('b_action_label')
            ax.set_title('b_action over time')

            # 自定义 x 轴刻度：每 10s 一个刻度；若刻度能被60整除，额外在刻度下方显示整除后的结果（分钟数），
            # 否则显示该刻度除以60后的余数（秒）
            step = 10
            xmin, xmax = x.min(), x.max()
            ticks = np.arange(np.floor(xmin / step) * step, np.ceil(xmax / step) * step + 1, step)
            labels = []
            for t in ticks:
                ti = int(round(t))
                if ti % 60 == 0:
                    labels.append(f"{ti}\n{ti//60}")
                else:
                    labels.append(str(ti % 60))
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

            ax.grid(True)
            plt.tight_layout()
            plt.show()
            print()
