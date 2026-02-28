import argparse
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from Envs.Tasks.ChooseStrategyEnv_load_agents import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
import matplotlib
import json
import glob
import copy
import socket
import threading
from send2tacview import *
from Algorithms.Rules import decision_rule
from Math_calculates.CartesianOnEarth import *
from Math_calculates.sub_of_angles import *
from torch.distributions import Normal
import numpy as np
import torch as th
import re   # <-- 新增
from torch import nn
import torch.nn.functional as F
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Envs.UAVmodel6d import UAVModel
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Visualize.tacview_visualize import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOdiscrete import *
from Algorithms.PPOdiscrete import PolicyNetDiscrete, take_action_from_policy_discrete # 显式导入
# from tqdm import tqdm #  停用tqdm
from LaunchZone.calc_DLZ import *
from Math_calculates.one_hot import *

use_tacview = 0  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')

# def shoot_action_shield(at, distance, alpha, AA_hor, launch_interval):
#     at0 = at
#     # if distance > 60e3:
#     #     interval_refer = 30
#     # elif distance>40e3:
#     #     interval_refer = 20
#     # elif distance>20e3:
#     #     interval_refer = 15
#     # else:
#     #     interval_refer = 8

#     if distance>20e3:
#         interval_refer = 16
#     else:
#         interval_refer = 8
    
#     if distance > 80e3 or alpha > 60*pi/180:
#         at = 0
#     # if distance < 10e3 and alpha < pi/12 and abs(AA_hor) > pi*3/4 and launch_interval>30:
#     #     at = 1
#     if launch_interval < interval_refer:
#         at = 0

#     if abs(AA_hor) < pi*1/3 and distance>12e3: ## 禁止超视距完全尾追发射 新增
#         at=0

#     same = int(bool(at0) == bool(at))
#     xor  = int(bool(at0) != bool(at))  

#     return at, xor


start_time = time.time()
launch_time_count = 0


# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=300,  # 8 * 60, # todo 300s 太少了，改成8分钟!!!!!!!!!!!!!!!
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60, 
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# 超参数
dt_maneuver = 0.2  # 0.2 2
action_cycle_multiplier = 30 # 30
actor_lr = 1e-4  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# max_episodes = 1000 # 1000
max_steps = 120e4  # 120e4 65e4
hidden_dim = [128, 128, 128]  # 128
gamma = 0.95  # 0.9
lmbda = 0.95  # 0.9
epochs = 10  # 10
eps = 0.2
pre_train_rate = 0  # 0.25 # 0.25
k_entropy = 0.01  # 熵系数
mission_name = 'CombatPFSP'


env = ChooseStrategyEnv(args, tacview_show=use_tacview)

r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

state_dim = 35  # len(b_obs_spaces)
action_dim = 4  # 5 #######################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 整一下高度-攻击区离散表（使用有限的选择）
# 水平距离20km~80km, 我机高度 env.min_alt_safe 到 env.max_alt_safe 按 2e3 间隔划分
# 目标高度 = 我机高度-2e3, 0, 2e3, 双方速度均为1.2Ma，最后构建一个[我机高度, 目标高度, 不可逃逸区边界，最大边界]的查询表



# data_dir = os.path.join(project_root, "data")
# save_path = os.path.join(data_dir, "Crankinitial_states.npy")
# initial_states = np.load(save_path)
# print("读取的数据\n", initial_states)


def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

def creat_initial_state():
    # 飞机出生状态指定
    # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
    blue_height = np.random.uniform(4000, 12000)
    red_height = blue_height + np.random.uniform(-2000, 2500)
    red_psi = np.random.choice([-1, 1]) * pi/2 # random.uniform(-pi, pi)
    blue_psi = sub_of_radian(red_psi, -pi)
    # blue_beta = red_psi
    red_N = 0
    red_E = -np.sign(red_psi) * 40e3
    blue_N = red_N
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                        'psi': red_psi
                        }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- ELO Rating System ---
K_FACTOR = 32
INITIAL_ELO = 1200

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

def update_elo(player_elo, opponent_elo, score):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    new_player_elo = player_elo + K_FACTOR * (score - expected)
    return new_player_elo

def get_opponent_probabilities(elo_ratings, target_elo=None, sigma=200.0):
    """返回与 elo_ratings.keys() 顺序对应的概率数组。
    - target_elo: 要优先靠近的 ELO（传入 main_agent_elo）。
    - sigma: 高斯核标准差，越小越只选接近 target_elo 的对手。
    """
    keys = list(elo_ratings.keys())
    if len(keys) == 0:
        return np.array([])

    elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

    if target_elo is None:
        target_elo = np.mean(elos)

    # 以高斯核度量相似度（基于差的平方）
    diffs = elos - float(target_elo)
    # 数值稳定性：将 exponent 的常数项减去 max
    scores = np.exp(-0.5 * (diffs / float(sigma))**2)
    probs = scores / (scores.sum() + 1e-12)
    return probs

# 新增：将完整路径或名字缩短为 actor_rein<数字> 形式的 key（优先匹配 actor_*）
def shorten_actor_key(path_or_name):
    base = os.path.basename(path_or_name)
    name, _ = os.path.splitext(base)
    m = re.search(r'(actor_[A-Za-z]*\d+)', name)
    if m:
        return m.group(1)
    return name
# --- End ELO System ---

def append_b_experience(td, state, action, reward, next_state, done):
    """
    统一把一次蓝方经验追加到 transition_dict。
    只负责按给定值追加，不处理死亡冻结逻辑（主流程中通过 dead_dict 控制何时计算 next_state/reward）。
    """
    td['states'].append(state)
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    return td


if __name__=="__main__":
    
    # Define the action cycle multiplier
    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2) # 2,2

    # 为FSP实例化一个红方策略网络
    red_actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    # log_dir = "./logs"
    from datetime import datetime
    # log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # log_dir = os.path.join("./logs", f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logs_dir = os.path.join(project_root, "logs/combat")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Math_calculates.ScaleLearningRate import scale_learning_rate
    # 根据参数数量缩放学习率
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)
    agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    from Visualize.tensorboard_visualize import TensorBoardLogger

    # --- ELO 初始化 ---
    main_agent_elo = INITIAL_ELO
    # try to load existing elo json if present, otherwise start empty dict
    elo_json_path = os.path.join(log_dir, "elo_ratings.json")
    if os.path.isfile(elo_json_path):
        try:
            with open(elo_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 将任意 key 归一化为短 key（actor_reinXXX）
            elo_ratings = {shorten_actor_key(k): float(v) for k, v in raw.items()}
        except Exception as e:
            print(f"Warning: failed to load elo json '{elo_json_path}': {e}")
            elo_ratings = {}
    else:
        elo_ratings = {}
    # --- ELO 初始化结束 ---

    return_list = []

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    total_steps = 0
    i_episode = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0
    decide_steps_after_update = 0
    try:
        
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],}
        
        # 强化学习训练
        while total_steps < int(max_steps*(1-pre_train_rate)):
            
            i_episode += 1
            test_run = 0

            # --- FSP核心：为红方加载一个历史策略 ---
            # 查找所有已保存的 actor 模型
            actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
            if not actor_files:
                # 如果没有历史模型，红方使用固定策略（例如总是进攻）
                red_actor_loaded = False
                opponent_path = None
                opponent_key = None
            else:
                # 为每个短 key 选择一个路径（使用最新修改时间覆盖）
                path_map = {}
                for f in actor_files:
                    key = shorten_actor_key(f)
                    # 选择最新的文件作为该 key 的路径
                    if key not in path_map or os.path.getmtime(f) > os.path.getmtime(path_map[key]):
                        path_map[key] = f
                    if key not in elo_ratings:
                        elo_ratings[key] = INITIAL_ELO
                
                # 根据ELO分数计算概率并选择对手（使用短 key）
                opponent_keys = list(elo_ratings.keys())
                probabilities = get_opponent_probabilities(elo_ratings, target_elo=main_agent_elo, sigma=200.0)
                opponent_key = np.random.choice(opponent_keys, p=probabilities)
                opponent_path = path_map.get(opponent_key, None)

                try:
                    if opponent_path is not None:
                        red_actor.load_state_dict(torch.load(opponent_path, map_location=device))
                        red_actor.eval() # 设置为评估模式
                        red_actor_loaded = True
                        if i_episode % 20 == 0: # 每20回合打印一次对手信息
                            opponent_elo = elo_ratings.get(opponent_key, "N/A")
                            print(f"Episode {i_episode}: Red opponent is {opponent_key} (ELO: {opponent_elo:.0f})")
                    else:
                        raise FileNotFoundError("No actor file found for selected opponent key.")
                except Exception as e:
                    print(f"Warning: Failed to load opponent model {opponent_path}. Error: {e}")
                    red_actor_loaded = False

            episode_return = 0
            

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            
            last_decision_state = None
            current_action = None
            b_reward = None

            # 新增：每回合的死亡查询表（0 表示存活，1 表示已记录死亡瞬间）
            dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            # Initialize variables to track the last actions and crank flags
            last_r_action_label = None
            last_b_action_label = None
            crank_after_attack_red = 0
            crank_after_attack_blue = 0

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
                r_check_obs = env.base_obs('r')
                b_check_obs = env.base_obs('b')
                b_obs_n = flatten_obs(b_check_obs, env.key_order)
                # 在这里将观测信息压入记忆
                # env.RUAV.obs_memory = r_check_obs.copy()
                # env.BUAV.obs_memory = b_check_obs.copy()
                b_obs = np.squeeze(b_obs_n)
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 action_cycle_multiplier 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    if steps_of_this_eps > 0:
                        transition_dict = append_b_experience(transition_dict, last_decision_state, current_action, b_reward, b_obs, False)

                    # 开始新的动作周期：记录起始状态并决策
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作
                    if not test_run:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=True)
                    else:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=False)
                    decide_steps_after_update += 1
                    b_action_options = [
                        "attack",
                        "escape",
                        "left",
                        "right",
                    ]

                    current_action = b_action_label

                    # --- 红方决策 ---
                    if not red_actor_loaded:
                        r_action_label = 0  # 如果没有加载模型，则执行默认动作
                    else:
                        r_obs_n = flatten_obs(r_check_obs, env.key_order)
                        r_obs = np.squeeze(r_obs_n)
                        r_action_label = take_action_from_policy_discrete(red_actor, r_obs, device, explore=False)

                ### 发射导弹，这部分不受10step约束
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                # 发射导弹判决
                if distance <= 40e3 and distance >= 5e3: # and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
                    launch_time_count = 0
                    if r_action_label==0:
                        # launch_missile_if_possible(env, side='r')
                        launch_missile_with_basic_rules(env, side='r') # fixme 距离超过40km的时候根本就不会发射导弹，所以学不会crank
                    if b_action_label==0:
                        # launch_missile_if_possible(env, side='b')
                        launch_missile_with_basic_rules(env, side='b')
                
                # --- Update crank flags based on action transitions ---
                if last_r_action_label is not None and last_r_action_label == 0 and r_action_label in [2, 3]:
                    crank_after_attack_red = 1
                else:
                    crank_after_attack_red = 0

                if last_b_action_label is not None and last_b_action_label == 0 and b_action_label in [2, 3]:
                    crank_after_attack_blue = 1
                else:
                    crank_after_attack_blue = 0

                # Update last action labels
                last_r_action_label = r_action_label
                last_b_action_label = b_action_label

                # --- 发射导弹逻辑 ---
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                if distance > 40e3 and crank_after_attack_red:
                    launch_missile_immediately(env, side='r')
                if distance > 40e3 and crank_after_attack_blue:
                    launch_missile_immediately(env, side='b')

                env.step(r_action_label, b_action_label) # Environment updates every dt_maneuver
                done, b_reward, b_event_reward = env.combat_terminate_and_reward('b', b_action_label)
                done = done
                
                episode_return += b_reward * env.dt_maneuver

                # 仅在蓝方刚被击毁（第一次检测到死亡瞬间）时，计算一次 reward 和 next_b_obs 并标记 dead_dict
                if dead_dict['b'] == 0:
                    next_b_check_obs = env.base_obs('b')
                    next_b_obs = np.squeeze(flatten_obs(next_b_check_obs, env.key_order))
                    if env.BUAV.dead:
                        dead_dict['b'] = 1


                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            # --- 回合结束处理 ---
            # 存储最后一个不完整动作周期的经验（若 last_decision_state 存在）
            if last_decision_state is not None:
                # # 若在回合结束前未曾在死亡瞬间计算 next_b_obs（例如超时终止或其他非击毁终止），做一次后备计算
                # if 'next_b_obs' not in locals():
                #     next_b_check_obs = env.base_obs('b')
                #     next_b_obs = flatten_obs(next_b_check_obs, env.key_order)
                transition_dict = append_b_experience(transition_dict, last_decision_state, current_action, b_reward, next_b_obs, True)

            # --- ELO 更新 ---
            if red_actor_loaded and opponent_key in elo_ratings:
                opponent_elo = elo_ratings[opponent_key]
                if env.win: # 蓝方(主智能体)胜利
                    score_for_main = 1.0
                    score_for_opponent = 0.0
                elif env.lose: # 蓝方失败
                    score_for_main = 0.0
                    score_for_opponent = 1.0
                else: # 平局
                    score_for_main = 0.5
                    score_for_opponent = 0.5
                
                new_main_elo = update_elo(main_agent_elo, opponent_elo, score_for_main)
                new_opponent_elo = update_elo(opponent_elo, main_agent_elo, score_for_opponent)
                
                main_agent_elo = new_main_elo
                elo_ratings[opponent_key] = new_opponent_elo
            # --- ELO 更新结束 ---

            if 1: # len(transition_dict['next_states']) >= transition_dict_threshold: # decide_steps_after_update >= transition_dict_threshold
                '''agent.update'''
                agent.update(transition_dict, adv_normed=False)
                decide_steps_after_update = 0
                # Clear transition_dict after update
                
                actor_grad_norm = agent.actor_grad
                actor_pre_clip_grad = agent.pre_clip_actor_grad
                critic_grad_norm = agent.critic_grad
                critic_pre_clip_grad = agent.pre_clip_critic_grad

                # 梯度监控
                logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
                logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
                logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
                logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
                # 损失函数监控
                logger.add("train/7 actor_loss", agent.actor_loss, total_steps)
                logger.add("train/8 critic_loss", agent.critic_loss, total_steps)
                # 强化学习actor特殊项监控
                logger.add("train/9 entropy", agent.entropy_mean, total_steps)
                logger.add("train/10 ratio", agent.ratio_mean, total_steps) 
                logger.add("train/11 episode/step", i_episode, total_steps)    

                transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

            
            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            return_list.append(episode_return)

            # tensorboard 训练进度显示
            if test_run:
                # logger.add("train/1 episode_return", episode_return, total_steps)
                # logger.add("train/2 not lose", 1-env.lose, total_steps)
                pass # 不专门区分训练和测试回合
            else:
                total_steps += steps_of_this_eps + 1
                logger.add("train/1 episode_return", episode_return, total_steps)
                logger.add("train/2 win", env.win, total_steps)
                logger.add("train/2 lose", env.lose, total_steps)
                logger.add("train/2 draw", env.draw, total_steps)
                

            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            

            # --- 保存模型
            os.makedirs(log_dir, exist_ok=True)
            # critic overwrite
            critic_path = os.path.join(log_dir, "critic.pt")
            th.save(agent.critic.state_dict(), critic_path)
            # actor RL snapshot
            if i_episode % 10 == 0:
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

                # save elo ratings to json (atomic write)
                try:
                    tmp_path = elo_json_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        # elo_ratings 已为短 key，直接保存即可
                        json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, elo_json_path)
                except Exception as e:
                    print(f"Warning: failed to save elo json '{elo_json_path}': {e}")
            
            # 训练进度显示
            if (i_episode) >= 10:
                print(f"episode {i_episode}, 进度: {total_steps/max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")
            else:
                print(f"episode {i_episode}, total_steps {total_steps}")


        training_end_time = time.time()  # 记录结束时间
        elapsed = training_end_time - training_start_time
        from datetime import timedelta
        td = timedelta(seconds=elapsed)
        d = td.days
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        print(f"总训练时长: {d}天 {h}小时 {m}分钟 {s}秒")


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        # 保存最终 elo 到 json
        try:
            tmp_path = elo_json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, elo_json_path)
        except Exception as e:
            print(f"Warning: failed to save final elo json '{elo_json_path}': {e}")

        # 打印最终的ELO排名
        if elo_ratings:
            print("\n--- Final ELO Rankings ---")
            sorted_elos = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
            for key, elo in sorted_elos:
                print(f"ELO: {elo:.0f} - {key}")
            print(f"Main Agent Final ELO: {main_agent_elo:.0f}")
            print("------------------------\n")
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")

