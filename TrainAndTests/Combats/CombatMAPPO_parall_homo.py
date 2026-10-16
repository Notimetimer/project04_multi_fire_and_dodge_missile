'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

import typing
import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time  # 确保引入 time 模块
import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import traceback # [新增]
import random
import collections

from _context import *
import BasicRules_Homo
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
from Algorithms.ActorWithPointingNet import ActorWrapper
from Algorithms.CriticWrapper import critic_wrapper
from Algorithms.MAPPOHybrid import MAPPOHybrid
from Math_calculates.sub_of_angles import sub_of_radian, sub_of_degree
from Visualize.tensorboard_visualize import TensorBoardLogger
from VsBaseline_MA import test_worker

    
dt_move = 0.04

# 测试对战初始弹药（可调）
test_red_init_ammo = 6
test_blue_init_ammo = 6

# 周期性测试所对抗的规则对手编号 (仅测试 0, 1, 2, 3)
TEST_RULE_IDS = [0, 1, 2, 3]


def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

def summarize(il_dict):
    if il_dict is None: return
    print("\nData Summary:")
    for k in ('states', 'actions', 'returns'):
        v = il_dict.get(k)
        if isinstance(v, dict):
            for sk, sv in v.items():
                print(f"  {k}['{sk}']: shape={sv.shape}")
        else:
            print(f"  {k}: shape={getattr(v, 'shape', 'Unknown')}")



# calculate_expected_score 概率计算

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None,
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta',
                               compete_old_rate=0.5, deltaFSP_epsilon=0.5,):
    """
    优化后的对手采样逻辑：
    1. 优先判定是否进入“规则复习”分支。
    2. 若未进入，则根据 SP_type 执行具体的采样策略。
    """
    # 【核心修改】在函数内部合并出一个临时的全集字典用于查询分数
    # 这样 keys 里的任何元素都能在这里找到对应的 ELO
    
    if hall_of_fame is not None:
        candidate_pool = hall_of_fame.copy()
        candidate_pool.update(elite_elo_ratings)
    else:
        candidate_pool = elite_elo_ratings
    keys = list(candidate_pool.keys())
    
    if not keys: return np.array([]), []

    #  第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护)
    # 只要 compete_old_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < compete_old_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    #  第二层判断：进入核心采样逻辑
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = min(np.max(elos), float(target_elo) + 300)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
            # # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            # actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
        diffs = elos - actual_target
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
        return probs, keys

    # 2. 处理 FSP (全样本均匀分布)
    elif SP_type == 'FSP':
        probs = np.ones(len(keys)) / len(keys)
        return probs, keys

    # 3. 处理 deltaFSP (新旧池切分)
    elif SP_type == 'deltaFSP':
        n = len(keys)
        new_count = max(1, int(np.ceil(n * 0.2)))
        new_keys = keys[-new_count:]
        old_keys = keys[:-new_count]
        
        # 这里的 deltaFSP_epsilon 建议直接作为参数传入或使用全局变量
        if np.random.rand() < float(deltaFSP_epsilon) or not old_keys:
            target_keys = new_keys
        else:
            target_keys = old_keys
            
        probs = np.ones(len(target_keys)) / len(target_keys)
        return probs, target_keys

    # 4. 处理 SP (最新历史版本)
    elif SP_type == 'SP':
        # rein_keys = [k for k in keys if k.startswith('actor_rein') and '_step_' not in k]
        # 严格匹配 actor_rein + 数字
        rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            # try: return int(k.replace('actor_rein', ''))
            # except: return -1
            try: return int(re.search(r'actor_rein(\d+)', k).group(1))
            except: return -1
            
        best_key = max(rein_keys, key=extract_number)
        return np.array([1.0]), [best_key]

    # 5. 兜底逻辑: Rule 均匀采样 (None)
    else:
        if not rule_keys: return np.array([]), []
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                8000.0 * (1-int(randomized))
    if np.random.uniform(0,1) < 0.2 and int(randomized): # 以很低概率在更大的范围随机初始高度
        blue_height = np.random.uniform(4000.0, 12000.0)
    red_height = blue_height
    # 初始航向随机化
    red_psi = sub_of_radian(np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    blue_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    init_North = np.random.uniform(-20e3, 20e3) * int(randomized)
    red_N = init_North
    red_E = - 55e3 # 45e3
    if np.random.uniform(0,1) < 0.2: # 以低概率随机初始距离
        red_E = np.random.uniform(40e3,55e3)
    blue_N = init_North
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process(rank, pipe, args, device_worker, dt_maneuver, seed, dt_move=0.07, pomdp=1, vertices=None):
    """采样 worker：红方学生共享一个 ActorWrapper（同构），蓝方为规则或历史 nn（对手）。
    经验只按学生 id 分组保存在内存，堆叠后通过 pipe 传回 master，不写磁盘。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.ActorWithPointingNet import ActorWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims, load_init_states, append_experience

    env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    env.dt_move = dt_move
    env.dt_maneuver = dt_maneuver

    # 网络维度统一从 json 读
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    # settings 未携带时的初始数量/挂载兜底值（与 init_states.json 一致）
    init_cfg = load_init_states()
    # 红方学生共享一个 ActorWrapper（同构），蓝方历史 nn 对手按蓝机 id 存放
    actor_dict = {}
    student_actor = None

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RUN_EPISODE':
                student_actor_weights = packet[0]   # 同构单份 state_dict，全队共享
                opp_name, opp_type, opp_data, opp_temperature = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])

                # 加载红方学生共享 actor 权重（学生 id 与环境红机 id 一致，全队共用一个实例）
                if student_actor is None:
                    student_actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                        device=device_worker,
                    )
                student_actor.load_state_dict(student_actor_weights)
                student_actor.eval()

                # 蓝方历史 nn 对手权重（master 加载时已把学生 id 重映射为蓝机 id 打包）
                blue_nn_weights = opp_data if (opp_type == 'nn' and opp_data) else {}
                for id, sd in blue_nn_weights.items():
                    if id not in actor_dict:
                        actor_dict[id] = ActorWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker,
                        )
                    actor_dict[id].load_state_dict(sd)
                    actor_dict[id].eval()

                rb, bb = settings['red_birth'], settings['blue_birth']
                red_shuffle_seed = settings.get('red_shuffle_seed')
                blue_shuffle_seed = settings.get('blue_shuffle_seed')
                env.reset(red_birth_state=rb, blue_birth_state=bb,
                          red_init_num=red_init_num, blue_init_num=blue_init_num,
                          red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo,
                          seed=(red_shuffle_seed, blue_shuffle_seed))

                # 经验只按学生（红方）id 收集（蓝方为规则或历史 nn，不参与训练）
                student_ids = [id for id in env.UAV_ids if env.find_uav_with_id(id).side == 'r']
                LocalTrans = {id: {'obs': [], 'states': [], 'actions': [],
                                   'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                              for id in student_ids}
                Last_Obs = {id: None for id in student_ids}
                Last_States = {id: None for id in student_ids}
                Last_Actions = {id: None for id in student_ids}
                Rewards = {id: 0.0 for id in student_ids}

                maneuver_cmds_all = {}
                target_index_all = {}
                done = False
                ep_return = 0.0
                ep_return_by_id = {id: 0.0 for id in student_ids} # 个体回报（tensorboard 按 id 记录）

                for count in range(round(args.max_episode_len / dt_maneuver)):
                    if not env.running or done:
                        break

                    agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
                    Curr_States = {}
                    for id in env.UAV_ids:
                        agent_input, mask_input, rule_input = env.get_obs(id, pomdp=pomdp)
                        agent_inputs_[id] = agent_input
                        mask_inputs_[id] = mask_input
                        rule_inputs_[id] = rule_input
                        Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)

                    # 决策点
                    if count % action_cycle_multiplier == 0:
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            is_student = UAV.side == 'r'
                            # 保存上一步经验（仅学生，决策步对齐，统一走 append_experience）
                            if is_student and Last_Obs[id] is not None and count > 0:
                                LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id],
                                                                   Last_Actions[id], Rewards[id], Curr_States[id],
                                                                   done=0, active_mask=not UAV.dead)

                            if is_student or id in blue_nn_weights:
                                # 神经网络决策（红方学生走共享实例 / 蓝方历史 nn 按 id 取）
                                act_model = student_actor if is_student else actor_dict[id]
                                actor_rule = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                                with torch.no_grad():
                                    actor_out = act_model.get_action(
                                        agent_inputs_[id], mask_inputs_[id], actor_rule)
                                vert = int(actor_out['vert_idx'].item())
                                hor = int(actor_out['hor_idx'].item())
                                fire = int(actor_out['fire'].item())
                                enm_idx = int(actor_out['enm_idx'].item())
                                maneuver_cmds_all[id] = [vert, hor]
                                target_index_all[id] = enm_idx
                                fire_target = enm_idx
                                fire_flag = bool(fire)
                                # 记录动作供经验池使用（仅学生，log_prob 为行为策略真实值，update 直接读取）
                                if is_student:
                                    Last_Actions[id] = {
                                        "vert": vert, "hor": hor, "fire": fire,
                                        "enm_idx": enm_idx,
                                        "ally_idx": int(actor_out['ally_idx'].item()),
                                        "log_prob_enm": float(actor_out['log_prob_enm'].item()),
                                        "log_prob_ally": float(actor_out['log_prob_ally'].item()),
                                        "log_prob_act": float(actor_out['log_prob_act'].item()),
                                    }
                            else:
                                # 蓝方规则对手
                                fly_action, fire_action, _ = basic_rules(
                                    rule_inputs_[id], mask_inputs_[id], rules_num=opp_data)
                                maneuver_cmds_all[id] = fly_action['do']
                                target_index_all[id] = int(fly_action['at'])
                                fire_target = int(fire_action['at'])
                                fire_flag = bool(fire_action['do'])

                            # 决策瞬间的 obs/states 只按学生记录
                            if is_student:
                                Last_Obs[id] = {
                                    'agent_input': agent_inputs_[id],
                                    'mask_input': mask_inputs_[id],
                                    'rule_input': take_some_of_rule_inputs(rule_inputs_[id]),
                                }
                                Last_States[id] = Curr_States[id]

                            if fire_flag:
                                try:
                                    env.fire_120(id, fire_target, tabu=0)
                                except ValueError as e:
                                    pass
                            env.radar_action(id, 1)

                    # 执行机动
                    flight_cmds = {}
                    for id in env.UAV_ids:
                        try:
                            flight_cmds[id] = env.maneuver_for_1_enm(
                                id, target_index_all[id], maneuver_cmds_all[id], rule_inputs_[id])
                        except ValueError:
                            flight_cmds[id] = [0.0, 0.0, 0.0]
                    env.step(flight_cmds.values())

                    # 奖励结算（combat_terminate_and_reward 每机都要调，回报只按学生累计）
                    ep_return = 0.0
                    Curr_States = {}
                    for id in env.UAV_ids:
                        done, reward_dict = env.combat_terminate_and_reward(
                            id=id, action_cycle_multiplier=action_cycle_multiplier)
                        if id in Rewards:
                            Rewards[id] = sum(reward_dict.values())
                            ep_return += Rewards[id]
                            ep_return_by_id[id] += Rewards[id]

                    for id in env.UAV_ids:
                        Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)

                # 最后一帧经验（done=1，仅学生，统一走 append_experience）
                for id in student_ids:
                    UAV = env.find_uav_with_id(id)
                    LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id],
                                                        Last_Actions[id], Rewards[id],
                                                        Curr_States.get(id, Last_States[id]),
                                                        done=1, active_mask=not UAV.dead)

                # 堆叠各 id 经验（dict-of-arrays，只有学生 id）
                stacked_trans = {}
                for id, lt in LocalTrans.items():
                    stacked_trans[id] = {k: stack_field(v) for k, v in lt.items()}

                # 胜负判定：直接读环境端导出的 r_outcome（1 胜 / -1 负 / 0 平，学生为红方）
                outcome = env.r_outcome if env.r_outcome is not None else 0
                win = outcome == 1
                lose = outcome == -1
                draw = outcome == 0
                all_dead = all(uav.dead for uav in env.UAVs)

                pipe.send({
                    'trans': stacked_trans,
                    'metrics': {
                        'steps': count,
                        'return': ep_return,
                        'return_by_id': ep_return_by_id, # 个体回报
                        'event_return': 0,
                        'win': win,
                        'lose': lose,
                        'draw': draw,
                        'BVR_perish_together': all_dead and env.t <= args.max_episode_len - 10,
                    },
                    'opp_name': opp_name,
                })
            elif cmd == 'EXIT':
                break
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            pipe.send({'error': tb})
            break


def run_MLP_simulation(
    collape_recover={
        "collapsed": False,
        "best_actor_name": None,
        "actor_frozen_batchs": 5,
    },
    num_workers=10, # 并行进程数，根据CPU核数调整，建议 10-20
    mission_name='无名',
    actor_lr=1e-4,
    critic_lr=5e-4,
    max_steps=4 * 165e4,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=0.001,
    action_cycle_multiplier=10,
    dt_move=0.07,
    dt_maneuver=0.2,
    max_episode_duration=15*60,
    R_cage=62.00e3,
    red_init_num=None, blue_init_num=None,
    red_init_ammo=None, blue_init_ammo=None,
    # 自博弈相关
    init_elo_ratings={"Rule_0": 1200, "Rule_1": 1200, "Rule_2": 1200},
    self_play_type='PFSP',
    hist_agent_as_opponent=1,
    sigma_elo=400,
    WARM_UP_STEPS=500e3,
    ADMISSION_THRESHOLD=0.5,
    MAX_HISTORY_SIZE=50,
    deltaFSP_epsilon=0.8,
    compete_old_rate=0.2,
    K_FACTOR=16,
    randomized_birth=1,
    save_interval=1,
    transition_dict_threshold=2048,
    mini_batch_size_mixed=256,
    num_runs=3,
    device=torch.device("cpu"),
    vertices=None,
    resume_dir=None,
):
    """多智能体 MAPPO 训练主函数。
    同构：全队共享一套 ActorWrapper/critic_wrapper/MAPPOHybrid，经验按学生 id 收集、更新前合并。
    保存命名：actor_rein{batch_idx}.pt / critic.pt。"""
    from _context import load_obs_dims, load_init_states

    # 初始数量/挂载：显式传参优先，否则统一从 init_states.json 读取
    init_cfg = load_init_states()
    red_init_num = init_cfg['red_init_num'] if red_init_num is None else red_init_num
    blue_init_num = init_cfg['blue_init_num'] if blue_init_num is None else blue_init_num
    red_init_ammo = init_cfg['red_init_ammo'] if red_init_ammo is None else red_init_ammo
    blue_init_ammo = init_cfg['blue_init_ammo'] if blue_init_ammo is None else blue_init_ammo

    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. 参数与环境配置
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration)
    parser.add_argument("--R-cage", type=float, default=R_cage)
    args = parser.parse_args()

    print(f"Master training device: {device}")

    # 3. 网络维度从 json 读取，用 dummy env 获取 UAV id 列表
    dummy_env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    dummy_env.reset(red_init_num=red_init_num, blue_init_num=blue_init_num,
                    red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo)
    uav_ids = list(dummy_env.UAV_ids)
    # 学生统一为红方：学生 id 与环境红机 id 一致（1, 2），蓝方为规则或历史 nn 对手
    red_student_ids = sorted([i for i in uav_ids if dummy_env.find_uav_with_id(i).side == 'r'])
    del dummy_env

    fixed_dim, enm_shape, ally_shape = load_obs_dims()

    # 4. 全队共享一套 actor/critic/MAPPOHybrid（完全同构）
    actor = ActorWrapper(
        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
    critic = critic_wrapper(
        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
    agent = MAPPOHybrid(
        actor=actor, critic=critic,
        actor_lr=actor_lr, critic_lr=critic_lr,
        lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma,
        device=device, k_entropy=k_entropy,
    )

    # 日志记录
    logs_dir = os.path.join(project_root, "logs/combat")
    if resume_dir is not None and os.path.exists(resume_dir):
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

    # 中断续训：加载最新 actor_rein{latest_batch}.pt 和 critic.pt（同构单份）
    if resume_dir is not None and os.path.exists(resume_dir):
        actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
        if actor_files:
            def extract_batch(f):
                m = re.search(r'actor_rein(\d+)', os.path.basename(f))
                return int(m.group(1)) if m else -1
            latest_batch = max(extract_batch(f) for f in actor_files)
            ap = os.path.join(log_dir, f"actor_rein{latest_batch}.pt")
            if os.path.exists(ap):
                actor.load_state_dict(torch.load(ap, map_location=device))
                print(f"Loaded shared actor from: {ap}")
            cp = os.path.join(log_dir, "critic.pt")
            if os.path.exists(cp):
                critic.load_state_dict(torch.load(cp, map_location=device))
                print(f"Loaded shared critic from: {cp}")
            batch_idx = latest_batch
        else:
            batch_idx = 0

        opt_path = os.path.join(log_dir, "optimizers_state.pt")
        if os.path.exists(opt_path):
            try:
                opt_states = torch.load(opt_path, map_location=device)
                agent.actor_optimizer.load_state_dict(opt_states['actor_optimizer'])
                agent.critic_optimizer.load_state_dict(opt_states['critic_optimizer'])
                print("Loaded optimizer states.")
            except Exception as e:
                print(f"Failed to load optimizers: {e}")
    else:
        batch_idx = 0

    # logger
    logger = TensorBoardLogger(
        log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 存储在线训练前的网络参数（同构单份）
    torch.save(actor.state_dict(), os.path.join(log_dir, "actor_rein0.pt"))
    torch.save(critic.state_dict(), os.path.join(log_dir, "critic.pt"))

    # 并行环境初始化
    mp.set_start_method('spawn', force=True)
    test_pool = mp.Pool(processes=3, maxtasksperchild=10)

    # 启动并行训练 Worker
    workers = []
    pipes = []
    worker_device = torch.device('cpu')
    args.max_episode_len = max_episode_duration
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process,
                       kwargs={
                           'rank': i,
                           'pipe': child_conn,
                           'args': args,
                           'device_worker': worker_device,
                           'dt_maneuver': dt_maneuver,
                           'seed': seed,
                           'dt_move': dt_move,
                           'pomdp': 1,
                           'vertices': vertices,
                       })
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = {} # {copy.deepcopy(elo_ratings)}
    hall_of_fame = {}
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")
    # 新增：对手及精英开火与导弹期参数统计文件路径
    Elite_Fire_Stats_path = os.path.join(log_dir, "Elite_Fire_Stats.json")

    GameTimes = {}
    # 初始化开火与导弹期参数统计字典（不区分普通和精英）
    Elite_Fire_Stats = {}

    # 尝试加载历史 # 中断续训
    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f: elo_ratings = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f: hall_of_fame = json.load(f)
    if os.path.exists(Elite_Fire_Stats_path):
        with open(Elite_Fire_Stats_path, 'r', encoding='utf-8') as f: Elite_Fire_Stats = json.load(f)

    main_agent_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)

    # 团队命名：actor_rein{batch} 代表一个团队（含所有 id）
    int_team_name = "actor_rein0"

    # 不论如何，记录在线训练前的网络参数
    if not elite_elo_ratings:
        for k in init_elo_ratings.keys():
            elo_ratings[k] = main_agent_elo
        if hist_agent_as_opponent:
            elo_ratings[int_team_name] = main_agent_elo
        elite_elo_ratings = copy.deepcopy(elo_ratings)
        for k in elo_ratings.keys():
            GameTimes[k] = 0
            Elite_Fire_Stats[k] = [0.0, 0.0, 0.0, 0.0, 0.0]
        Elite_Fire_Stats["__CURRENT_MAIN__"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 训练循环变量
    total_steps = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
    else:
        actor_freeze_until = -1

    current_max_steps = int(max_steps)

    # 学生 id 独立的全局 Buffer
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [],
                             'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    transition_dicts = {uid: copy.deepcopy(empty_transition_dict) for uid in red_student_ids}

    # 基于胜率的在线 EMA
    ema_score = 0.5
    ema_step = 0
    target_p1 = 0.65

    # 主循环
    while True:
        while total_steps < current_max_steps:
            # 获取当前学生队共享策略权重 (CPU，单份同构)
            current_actor_weights = {k: v.cpu() for k, v in actor.state_dict().items()}

            # [测试] 每 batch 采样前：并行测试当前学生队 vs 规则对手
            if num_runs > 0:
                print(f"\n>>> [Per-Batch Test] Testing student team vs rules (batch {batch_idx})...")
                # 学生即红方且不改 id：测试环境红机 id 与学生 id 一致，直接下发整队
                test_team_weights = current_actor_weights
                test_tasks = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        test_worker,
                        kwds={
                            'model_state_dict': test_team_weights,
                            'rules_num': r_idx,
                            'env_args': args,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'vertices': vertices,
                            'dt_move': dt_move,
                            'dt_maneuver': dt_maneuver,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'red_init_num': red_init_num,
                            'blue_init_num': blue_init_num,
                            'red_init_ammo': test_red_init_ammo,
                            'blue_init_ammo': test_blue_init_ammo,
                        }
                    )
                    test_tasks.append(obj)

                # 阻塞等待全部测试完成后再进入采样
                test_results = [t.get() for t in test_tasks]
                outcomes = {rule_num: score for rule_num, score, wins, loses, draws, p_t_ in test_results}
                outcomes_perish = {rule_num: p_t_ for rule_num, score, wins, loses, draws, p_t_ in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps/action_cycle_multiplier)
                    print(f"  [Test Result] Rule_{r_num}: {score}")

                # 所有对手的平均Score和超视距双杀率
                avg_score = np.mean(list(outcomes.values()))
                avg_perish_together = np.mean(list(outcomes_perish.values()))
                logger.add("test/avg_score", avg_score, total_steps/action_cycle_multiplier)
                logger.add("test/BVR perish together", avg_perish_together, total_steps/action_cycle_multiplier)

            # 分发采样任务给 Worker
            worker_metrics_buffer = []

            # 对手池
            if not init_elo_ratings:
                sorted_all_keys = [k for k in sorted(elo_ratings.keys(),
                                    key=lambda x: elo_ratings[x] if not x.startswith("__") else -1e9,
                                    reverse=True) if not k.startswith("__")]
                effective_pool = {k: elo_ratings[k] for k in sorted_all_keys[:MAX_HISTORY_SIZE]}
            else:
                effective_pool = elite_elo_ratings

            selected_opponents = []
            for rank in range(num_workers):
                probs, opponent_keys = get_opponent_probabilities(
                    effective_pool, hall_of_fame,
                    target_elo=main_agent_elo, SP_type=self_play_type,
                    sigma=sigma_elo, compete_old_rate=compete_old_rate,
                    deltaFSP_epsilon=deltaFSP_epsilon,
                )
                sel = np.random.choice(opponent_keys, p=probs)
                selected_opponents.append(sel)

            for rank in range(num_workers):
                selected_opponent_name = selected_opponents[rank]
                if selected_opponent_name in GameTimes:
                    GameTimes[selected_opponent_name] += 1
                else:
                    GameTimes[selected_opponent_name] = 1

                # 准备对手数据
                opp_type = 'rule'
                opp_data = 0
                if "Rule" in selected_opponent_name:
                    try:
                        opp_data = int(selected_opponent_name.split('_')[1])
                    except:
                        opp_data = 0
                else:
                    opp_type = 'nn'
                    # 历史团队：读同构单份权重，按蓝机 id 打包（同一份 sd 多 key 引用，pickle 自动去重）
                    m = re.match(r'^actor_rein(\d+)$', selected_opponent_name)
                    if m:
                        old_batch = int(m.group(1))
                        ap = os.path.join(log_dir, f"actor_rein{old_batch}.pt")
                        if os.path.exists(ap):
                            shared_opp_sd = torch.load(ap, map_location='cpu')
                            # 蓝机 id = 学生 id + 200（同编队位次：1→201, 2→202）
                            opp_data = {uid + 200: shared_opp_sd for uid in red_student_ids}
                        else:
                            opp_type = 'rule'
                            opp_data = 0
                    else:
                        opp_type = 'rule'
                        opp_data = 0

                opp_temperature = np.random.uniform(0.99, 1.0)
                opp_info = (selected_opponent_name, opp_type, opp_data, opp_temperature)

                rb, bb = create_initial_state_worker(randomized_birth)
                settings = {
                    'action_cycle_multiplier': action_cycle_multiplier,
                    'red_birth': rb,
                    'blue_birth': bb,
                    'red_init_num': red_init_num,
                    'blue_init_num': blue_init_num,
                    'red_init_ammo': red_init_ammo,
                    'blue_init_ammo': blue_init_ammo,
                    # 红蓝各自独立的位置打乱种子（每 episode 重新生成）
                    'red_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
                    'blue_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
                }

                pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # 等待所有 Worker 完成 (Barrier)
            batch_results = []
            for rank in range(num_workers):
                try:
                    res = pipes[rank].recv()
                except EOFError:
                    print(f"[Error] Worker {rank} crashed silently.")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed.")
                if isinstance(res, dict) and 'error' in res:
                    print(f"--- Master received error from Worker {rank}, aborting. ---")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed with error:\n{res['error']}")
                batch_results.append(res)

            # 数据聚合
            batch_total_steps = 0
            batch_wins = 0
            batch_loss_cnt = 0
            batch_draw_cnt = 0
            batch_bvr_perish_together_cnt = 0
            batch_total_return = 0
            batch_return_by_id = {uid: 0 for uid in red_student_ids} # 个体回报累加器

            for res in batch_results:
                l_tr_by_id = res['trans']  # {id: transition_dict}
                metrics = res['metrics']
                opp_name = res['opp_name']

                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")

                batch_total_steps += metrics['steps']
                batch_total_return += metrics['return']
                # 个体回报累加
                for uid, r_id in metrics.get('return_by_id', {}).items():
                    if uid in batch_return_by_id:
                        batch_return_by_id[uid] += r_id
                if metrics.get('BVR_perish_together', False):
                    batch_bvr_perish_together_cnt += 1

                if metrics['win']: batch_wins += 1
                elif metrics['lose']: batch_loss_cnt += 1
                else: batch_draw_cnt += 1

                # 按 id 聚合经验（worker 已堆叠为 dict-of-arrays，递归合并；dict 字段禁用 extend）
                for uid, uid_trans in l_tr_by_id.items():
                    if uid not in transition_dicts:
                        continue
                    for k in transition_dicts[uid]:
                        transition_dicts[uid][k] = merge_transition_tree(transition_dicts[uid][k], uid_trans[k])

                # ELO 更新
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                if opp_name in elo_ratings:
                    prev_main_elo = main_agent_elo
                    adv_elo = elo_ratings[opp_name]
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                    elo_ratings["__CURRENT_MAIN__"] = main_agent_elo
                else:
                    new_adv_elo = main_agent_elo
                elo_ratings[opp_name] = new_adv_elo
                if opp_name in elite_elo_ratings:
                    elite_elo_ratings[opp_name] = new_adv_elo
                # 同步更新 Hall of Fame 中已有的对手Elo分值
                if opp_name in hall_of_fame:
                    hall_of_fame[opp_name] = new_adv_elo
            
        
            # 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps += batch_total_steps
            batch_idx += 1

            # 记录批次聚合指标
            batch_score = (batch_wins + batch_draw_cnt * 0.5) / num_workers
            ema_step += 1
            alpha_ema = 0.1
            if ema_step == 1:
                ema_score = batch_score
            else:
                ema_score = (1 - alpha_ema) * ema_score + alpha_ema * batch_score

            filtered_score = ema_score
            logger.add("train_plus/batch_score", batch_score, total_steps/action_cycle_multiplier)
            logger.add("train_plus/filtered_score", filtered_score, total_steps/action_cycle_multiplier)
            logger.add("train_plus/target_p1", target_p1, total_steps/action_cycle_multiplier)

            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps/action_cycle_multiplier)
            # 个体回报（每 id 一条）
            for uid, r_id in batch_return_by_id.items():
                logger.add(f"train/1 avg_episode_return_id{uid}", r_id / num_workers, total_steps/action_cycle_multiplier)
            logger.add("train/2 win", batch_wins / num_workers, total_steps/action_cycle_multiplier)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps/action_cycle_multiplier)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps/action_cycle_multiplier)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, total_steps/action_cycle_multiplier)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, total_steps/action_cycle_multiplier)
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps/action_cycle_multiplier)
            logger.add("agent/ batch_step", batch_idx, total_steps/action_cycle_multiplier)

            # 更新，保存与维护 (Checkpoint & Pool)
            if batch_idx % save_interval == 0 and \
                len(transition_dicts[red_student_ids[0]]['dones']) >= transition_dict_threshold:

                # ELO 相对位置
                all_keys = list(elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                target_pool_keys = rule_keys + latest_rein_keys
                avg_pool_elo = np.mean([elo_ratings[k] for k in target_pool_keys])
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, total_steps/action_cycle_multiplier)

                # 同构更新：合并各学生 id 的经验为一个 buffer，共享网络单次 PPO 更新
                merged_td = copy.deepcopy(empty_transition_dict)
                for uid in red_student_ids:
                    td = transition_dicts[uid]
                    if len(td['dones']) == 0:
                        continue
                    for k in merged_td:
                        merged_td[k] = merge_transition_tree(merged_td[k], td[k])
                if len(merged_td['dones']) > 0:
                    m = agent.update(merged_td, adv_normed=1, mini_batch_size=mini_batch_size_mixed)
                    logger.add("train/7 actor_loss", m.get('actor_loss', 0.0), total_steps/action_cycle_multiplier)
                    logger.add("train/8 critic_loss", m.get('critic_loss', 0.0), total_steps/action_cycle_multiplier)
                    logger.add("train/9 entropy", m.get('entropy', 0.0), total_steps/action_cycle_multiplier)
                    logger.add("train/10 approx_kl", m.get('kl', 0.0), total_steps/action_cycle_multiplier)
                    logger.add("train/11 clip_frac", m.get('clip_frac', 0.0), total_steps/action_cycle_multiplier)
                    # 记录未禁火 mask 内的开火概率极值（全队共享策略，单条曲线）
                    logger.add("train/12 max_fire_prob", m.get('fire_prob_max', float('nan')), total_steps/action_cycle_multiplier)
                    logger.add("train/13 min_fire_prob", m.get('fire_prob_min', float('nan')), total_steps/action_cycle_multiplier)

                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # A. 保存模型（同构单份，无 _uid 后缀）
                actor_team_key = f"actor_rein{batch_idx}"
                if batch_idx % 10 == 0:
                    torch.save(actor.state_dict(), os.path.join(log_dir, f"{actor_team_key}.pt"))
                    torch.save(critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                    print(f"Saved Checkpoint team: {actor_team_key}")
                    # 只注册已落盘的版本，保证自博弈采样对手时总能加载到对应权重
                    elo_ratings[actor_team_key] = main_agent_elo

                # 清空各学生 id Buffer
                for uid in red_student_ids:
                    transition_dicts[uid] = copy.deepcopy(empty_transition_dict)

                # B. 精英池维护
                if total_steps >= WARM_UP_STEPS:
                    elo_ratings[opp_name] = main_agent_elo

                # 全量历史记录
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx

                # 定期重刷精英池
                if total_steps > 0 and total_steps % 1e6 == 0:
                    print(f"\n>>> [Elite Pool Refresh] Refreshing elite_elo_ratings at {total_steps} steps...")
                    non_rule_elo = {k: v for k, v in elo_ratings.items() if not k.startswith("Rule") and not k.startswith("__")}
                    current_non_rule_count = len([k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")])
                    refresh_size = min(current_non_rule_count, MAX_HISTORY_SIZE)
                    sorted_non_rule = sorted(non_rule_elo.items(), key=lambda x: x[1], reverse=True)
                    top_non_rule = dict(sorted_non_rule[:refresh_size])
                    rule_elo = {k: v for k, v in elo_ratings.items() if k.startswith("Rule")}
                    elite_elo_ratings = {**rule_elo, **top_non_rule}
                    print(f"  [Elite Pool Refresh] Refreshed: {len(rule_elo)} Rules + {len(top_non_rule)} top agents (size={refresh_size})")

                # 保存全量日志
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

                # 精英池快照
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)

                # 名人堂
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                # 行为统计
                with open(Elite_Fire_Stats_path, "w", encoding="utf-8") as f:
                    json.dump(Elite_Fire_Stats, f, ensure_ascii=False, indent=2)

                #  日志记录 (Logging) - 保持不变，展示的是精英池状态
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps/action_cycle_multiplier)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    
                    # 升序：Elo 低（弱对手）在前。threshold=0 取最弱 Rule Elo（最低），=1 取最强 Rule Elo（最高）
                    rule_elos = sorted([v for k, v in elite_elo_ratings.items() if k.startswith("Rule")], reverse=False) # 升序排列
                    if not rule_elos:
                        rule_elo_thres = main_agent_elo
                    elif ADMISSION_THRESHOLD < 0:
                        rule_elo_thres = main_agent_elo  # 无门槛，差值仅供参考
                    else:
                        t = np.clip(ADMISSION_THRESHOLD, 0, 1)
                        idx = np.clip(int(t * (len(rule_elos) - 1) + 0.5), 0, len(rule_elos) - 1)
                        rule_elo_thres = rule_elos[idx]  # =0 取最低 Rule Elo，=0.5 取中位数，=1 取最高 Rule Elo

                    # 记录与 Rule 阈值的差值 (维持旧指标名)
                    elo_diff_to_thres = main_agent_elo - rule_elo_thres
                    logger.add("Elo_Centered/EloDiffToMean", elo_diff_to_thres, total_steps/action_cycle_multiplier)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps/action_cycle_multiplier)

                    curr_rank = 0.5 if elo_spread == 0 else (main_agent_elo - min_elo) / elo_spread

                    # Elo分数没有稳定不许记录
                    if total_steps >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rank_normed %", curr_rank * 100, total_steps/action_cycle_multiplier)

                    if main_agent_elo >= rule_elo_thres:
                        # 满员清理
                        history_keys = [k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")]
                        while len(history_keys) >= MAX_HISTORY_SIZE:
                            # 每次找到当前池子中最弱的一个
                            weakest_history_key = min(history_keys, key=lambda k: elite_elo_ratings[k])
                            old_elo = elite_elo_ratings[weakest_history_key]
                            # 从 ELO 字典和局部列表中同步删除
                            del elite_elo_ratings[weakest_history_key]
                            history_keys.remove(weakest_history_key)
                            print(f"[Pool Cleanup] Kicked weakest: {weakest_history_key} (Elo: {old_elo:.0f}), Current Pool: {len(history_keys)}")
                        
                        #  正式入池
                        if hist_agent_as_opponent:
                            elite_elo_ratings[actor_team_key] = main_agent_elo
                            print(f"Accepted {actor_team_key} into Elite Pool.")

                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps/action_cycle_multiplier)

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, total_steps/action_cycle_multiplier)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps/action_cycle_multiplier)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, total_steps/action_cycle_multiplier)

                # 例行保存优化器状态（同构单份）
                opt_save = {
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }
                torch.save(opt_save, os.path.join(log_dir, "optimizers_state.pt"))

                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

        #  [新增] 达到 max_steps 后的交互逻辑
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        # 【新增】明确告知用户 Worker 当前状态
        print("All simulation workers are now idling safely. System is paused.") 
        
        inp = input(f"Enter new max_steps (current {total_steps}), or press Enter to exit: ")
        
        if not inp.strip():
            print("No input provided. Exiting training.")
            break # 退出外层 while True 循环
        try:
            new_max = int(inp)
            if new_max > total_steps:
                current_max_steps = new_max
                print(f"Continuing training until {current_max_steps} steps.")
            else:
                print("Input steps less than current. Exiting...")
                break # 退出外层 while True 循环
        except ValueError:
            print("Invalid input (not a number). Exiting...")
            break

    # Cleanup
    print("Closing workers...")
    for pipe in pipes:
        try: # 【新增】防止管道已断开导致的报错
            pipe.send(('EXIT', None))
        except:
            pass
            
    for p in workers:
        p.join(timeout=5) # 【修改】给子进程 5 秒优雅退出的时间
        if p.is_alive():
            p.terminate() # 如果没死，强制结束

    # 【修改】调整顺序：先尝试关闭测试池
    try:
        test_pool.close()
        test_pool.join()
    except:
        pass
    
    logger.close()
    print("Training Finished.")