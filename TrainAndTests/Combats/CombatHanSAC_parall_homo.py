"""CombatHanSAC_parall_homo：Han et al. (2026) JPO 路线的 2V2 同构离策略训练入口。

并行采样/对手池/Elo 等基础设施沿用 CombatMAPPO_parall_homo.py 的约定（不改既有文件）：
- 同步 barrier 的多 worker 采样（spawn），红方学生共享一份 HanWrapper；
- 蓝方 = 规则对手 或 精英池历史 checkpoint（PFSP_balanced，Elo 高斯核）；
- 开火命中记录由环境回填、worker 汇总回传。

与 MAPPO 版本的核心差异（论文 JPO 路线）：
- 离策略：worker 每批回传的 (obs, a_flat, r, next_obs, done) 直接写入 D_BFM 回放池（1e5），
  master 每批从池中采样做若干次 SAC 更新（双 Q + 目标网络 τ=0.05 + 三温度自动调节）；
- 开火监督 Lf 不再单独冻结更新，而是并入 LNet 一次反传（式 35）；
- 测试（test_worker_han）按论文设定：联合分布 argmax + 开火命中概率阈值 0.8。

有意简化/差异（相对 MAPPO 版基础设施）：
- 不支持 enm_chosen_visable（队友目标黑板，默认本就关闭）；
- collape_recover 参数保留入口兼容，但不再实现冻结恢复逻辑；
- Elite_Fire_Stats 统计不保留（原实现只写不算）。
"""

import typing
import os
import sys
import csv
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time
import datetime
import torch.multiprocessing as mp
import traceback
import random
import collections

from _context import *
import BasicRules_Homo
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
from Algorithms.HanWrapper import HanWrapper, N_TIM, N_TDM
from Algorithms.HANSACHybrid import HANSACHybrid
from Math_calculates.sub_of_angles import sub_of_radian, sub_of_degree
from Visualize.tensorboard_visualize import TensorBoardLogger
from SelfPlayCalculation import calculate_expected_score, update_elo, get_opponent_probabilities

dt_move = 0.04

# 测试对战初始弹药：统一从 init_states.json 读取
_init_cfg = load_init_states()
red_init_ammo = _init_cfg['red_init_ammo']
blue_init_ammo = _init_cfg['blue_init_ammo']

# 周期性测试所对抗的规则对手编号
TEST_RULE_IDS = [0, 1, 2, 3]


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方（复制原逻辑，保持初始化分布一致）
def create_initial_state_worker(randomized=0):
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                8000.0 * (1-int(randomized))
    if np.random.uniform(0, 1) < 0.2 and int(randomized):
        blue_height = np.random.uniform(4000.0, 12000.0)
    red_height = blue_height

    init_North = np.random.uniform(-20e3, 20e3) * int(randomized)
    red_N = init_North
    red_E = - 55e3
    red_psi = sub_of_radian(np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    if np.random.uniform(0, 1) < 0.2:
        red_E = np.random.uniform(-40e3, -55e3)
    blue_N = init_North
    blue_E = - red_E
    blue_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    if np.random.uniform(0, 1) < 0.5:  # 随机换边
        red_E = - red_E
        blue_E = - blue_E
        red_psi = - red_psi
        blue_psi = - blue_psi

    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process_han(rank, pipe, args, device_worker, dt_maneuver, seed, dt_move=0.07, pomdp=1, vertices=None):
    """采样 worker：红方学生共享一个 HanWrapper（同构），蓝方为规则或历史 nn。
    离策略：每个决策点产出 (obs, a_flat, r, next_obs, done) 转移，堆叠后经 pipe 回传 master。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.HanWrapper import HanWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims, load_init_states

    env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    env.dt_move = dt_move
    env.dt_maneuver = dt_maneuver

    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    init_cfg = load_init_states()
    actor_dict = {}
    student_actor = None

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RUN_EPISODE':
                student_actor_weights = packet[0]
                opp_name, opp_type, opp_data = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])

                if student_actor is None:
                    student_actor = HanWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                        device=device_worker)
                student_actor.load_state_dict(student_actor_weights)
                student_actor.eval()

                blue_nn_weights = opp_data if (opp_type == 'nn' and opp_data) else {}
                for id, sd in blue_nn_weights.items():
                    if id not in actor_dict:
                        actor_dict[id] = HanWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker)
                    actor_dict[id].load_state_dict(sd)
                    actor_dict[id].eval()

                rb, bb = settings['red_birth'], settings['blue_birth']
                env.reset(red_birth_state=rb, blue_birth_state=bb,
                          red_init_num=red_init_num, blue_init_num=blue_init_num,
                          red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo,
                          seed=(settings.get('red_shuffle_seed'), settings.get('blue_shuffle_seed')))

                student_ids = [id for id in env.UAV_ids if env.find_uav_with_id(id).side == 'r']
                # SAC 转移：obs/next_obs 均为 (agent_input, mask_input, rule_input) 三元组
                LocalTrans = {id: {'obs': [], 'actions': [], 'rewards': [],
                                   'next_obs': [], 'dones': [], 'active_masks': []}
                              for id in student_ids}
                Last_Obs = {id: None for id in student_ids}
                Last_Actions = {id: None for id in student_ids}
                Rewards = {id: 0.0 for id in student_ids}

                def _obs_triple(oid):
                    return {'agent_input': agent_inputs_[oid],
                            'mask_input': mask_inputs_[oid],
                            'rule_input': take_some_of_rule_inputs(rule_inputs_[oid])}

                maneuver_cmds_all = {}
                target_index_all = {}
                done = False
                ep_return = 0.0
                ep_return_by_id = {id: 0.0 for id in student_ids}
                count = 0

                for count in range(round(args.max_episode_len / dt_maneuver)):
                    if not env.running or done:
                        break

                    agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
                    for id in env.UAV_ids:
                        agent_input, mask_input, rule_input = env.get_obs(id, pomdp=pomdp)
                        agent_inputs_[id] = agent_input
                        mask_inputs_[id] = mask_input
                        rule_inputs_[id] = rule_input

                    # 决策点
                    if count % action_cycle_multiplier == 0:
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            is_student = UAV.side == 'r'

                            # 上一决策点的转移在此闭环：r 为上一周期结算回报，next_obs 为当前观测
                            if is_student and Last_Obs[id] is not None and count > 0:
                                LocalTrans[id]['obs'].append(Last_Obs[id])
                                LocalTrans[id]['actions'].append(Last_Actions[id])
                                LocalTrans[id]['rewards'].append(Rewards[id])
                                LocalTrans[id]['next_obs'].append(_obs_triple(id))
                                LocalTrans[id]['dones'].append(0)
                                LocalTrans[id]['active_masks'].append(not UAV.dead)

                            if is_student or id in blue_nn_weights:
                                act_model = student_actor if is_student else actor_dict[id]
                                rule_input4actor = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                                with torch.no_grad():
                                    actor_out = act_model.get_action(
                                        agent_inputs_[id], mask_inputs_[id], rule_input4actor)
                                vert = int(actor_out['vert_idx'].item())
                                hor = int(actor_out['hor_idx'].item())
                                fire = int(actor_out['fire'].item())
                                enm_idx = int(actor_out['enm_idx'].item())
                                fire_target = int(actor_out['fire_target'].item())
                                maneuver_cmds_all[id] = [vert, hor]
                                target_index_all[id] = enm_idx
                                fire_flag = bool(fire)
                                if is_student:
                                    Last_Actions[id] = {
                                        'flat_idx': int(actor_out['flat_idx'].item()),
                                        'enm_idx': enm_idx, 'vert': vert, 'hor': hor,
                                        'fire': fire, 'fire_target': fire_target,
                                    }
                            else:
                                fly_action, fire_action, _ = basic_rules(
                                    rule_inputs_[id], mask_inputs_[id], rules_num=opp_data)
                                maneuver_cmds_all[id] = fly_action['do']
                                target_index_all[id] = int(fly_action['at'])
                                fire_target = int(fire_action['at'])
                                fire_flag = bool(fire_action['do'])

                            if is_student:
                                Last_Obs[id] = _obs_triple(id)

                            if fire_flag:
                                try:
                                    env.fire_120(id, fire_target, tabu=0)
                                except ValueError:
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

                    # 奖励结算
                    ep_return = 0.0
                    for id in env.UAV_ids:
                        done, reward_dict = env.combat_terminate_and_reward(
                            id=id, action_cycle_multiplier=action_cycle_multiplier)
                        if id in Rewards:
                            Rewards[id] = sum(reward_dict.values())
                            ep_return += Rewards[id]
                            ep_return_by_id[id] += Rewards[id]

                # 最后一帧转移（done=1）
                for id in student_ids:
                    UAV = env.find_uav_with_id(id)
                    LocalTrans[id]['obs'].append(Last_Obs[id])
                    LocalTrans[id]['actions'].append(Last_Actions[id])
                    LocalTrans[id]['rewards'].append(Rewards[id])
                    LocalTrans[id]['next_obs'].append(Last_Obs[id])  # 终止帧 next_obs 不参与 bootstrap
                    LocalTrans[id]['dones'].append(1)
                    LocalTrans[id]['active_masks'].append(not UAV.dead)

                stacked_trans = {}
                for id, lt in LocalTrans.items():
                    stacked_trans[id] = {k: stack_field(v) for k, v in lt.items()}

                outcome = env.r_outcome if env.r_outcome is not None else 0
                fire_hit_records = []
                for uav in env.UAVs:
                    fire_hit_records += uav.fire_hit_records

                pipe.send({
                    'trans': stacked_trans,
                    'metrics': {
                        'steps': count,
                        'return': ep_return,
                        'return_by_id': ep_return_by_id,
                        'win': outcome == 1,
                        'lose': outcome == -1,
                        'draw': outcome == 0,
                        'BVR_perish_together': all(uav.dead for uav in env.UAVs)
                                               and env.t <= args.max_episode_len - 10,
                    },
                    'opp_name': opp_name,
                    'fire_hit_records': fire_hit_records,
                })
            elif cmd == 'EXIT':
                break
        except Exception:
            pipe.send({'error': traceback.format_exc()})
            break


def test_worker_han(model_state_dict, rules_num, env_args, device_name='cpu', num_runs=3,
                    vertices=None, dt_move=0.07, dt_maneuver=0.2, action_cycle_multiplier=10,
                    red_init_num=2, blue_init_num=2, red_init_ammo=4, blue_init_ammo=4,
                    temperature=None, enm_chosen_visable=0):
    """周期性规则测试：论文设定 —— 联合分布 argmax + 开火命中概率阈值 0.8（deterministic）。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.HanWrapper import HanWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims

    env = ChooseStrategyEnv(env_args, tacview_show=0, vertices=vertices)
    env.dt_move = dt_move
    env.dt_maneuver = dt_maneuver
    fixed_dim, enm_shape, ally_shape = load_obs_dims()

    actor = HanWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                       device=torch.device(device_name))
    actor.load_state_dict(model_state_dict)
    actor.eval()

    wins = loses = draws = perish_togethers = 0
    kill_count = die_count = 0

    for _ in range(num_runs):
        rb, bb = create_initial_state_worker(1)
        env.reset(red_birth_state=rb, blue_birth_state=bb,
                  red_init_num=red_init_num, blue_init_num=blue_init_num,
                  red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo)
        done = False
        maneuver_cmds_all, target_index_all = {}, {}
        count = 0
        for count in range(round(env_args.max_episode_len / dt_maneuver)):
            if not env.running:
                break
            agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
            for id in env.UAV_ids:
                agent_inputs_[id], mask_inputs_[id], rule_inputs_[id] = env.get_obs(id, pomdp=1)

            if count % action_cycle_multiplier == 0:
                for id in env.UAV_ids:
                    UAV = env.find_uav_with_id(id)
                    if UAV.side == 'r':
                        rule_input4actor = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                        with torch.no_grad():
                            actor_out = actor.get_action(
                                agent_inputs_[id], mask_inputs_[id], rule_input4actor,
                                deterministic=True)
                        maneuver_cmds_all[id] = [int(actor_out['vert_idx'].item()),
                                                 int(actor_out['hor_idx'].item())]
                        target_index_all[id] = int(actor_out['enm_idx'].item())
                        fire_flag = bool(int(actor_out['fire'].item()))
                        fire_target = int(actor_out['fire_target'].item())
                    else:
                        fly_action, fire_action, _ = basic_rules(
                            rule_inputs_[id], mask_inputs_[id], rules_num=rules_num)
                        maneuver_cmds_all[id] = fly_action['do']
                        target_index_all[id] = int(fly_action['at'])
                        fire_flag = bool(fire_action['do'])
                        fire_target = int(fire_action['at'])
                    if fire_flag:
                        try:
                            env.fire_120(id, fire_target, tabu=0)
                        except ValueError:
                            pass
                    env.radar_action(id, 1)

            flight_cmds = {}
            for id in env.UAV_ids:
                try:
                    flight_cmds[id] = env.maneuver_for_1_enm(
                        id, target_index_all[id], maneuver_cmds_all[id], rule_inputs_[id])
                except ValueError:
                    flight_cmds[id] = [0.0, 0.0, 0.0]
            env.step(flight_cmds.values())
            for id in env.UAV_ids:
                done, _ = env.combat_terminate_and_reward(
                    id=id, action_cycle_multiplier=action_cycle_multiplier)
            if done:
                break

        outcome = env.r_outcome if env.r_outcome is not None else 0
        wins += int(outcome == 1)
        loses += int(outcome == -1)
        draws += int(outcome == 0)
        perish_togethers += int(all(uav.dead for uav in env.UAVs)
                                and env.t <= env_args.max_episode_len - 10)
        kill_count += sum(1 for uav in env.UAVs if uav.side == 'b' and uav.dead)
        die_count += sum(1 for uav in env.UAVs if uav.side == 'r' and uav.dead)

    result = (wins + 0.5 * draws) / num_runs
    return {'rules_num': rules_num, 'result': result,
            'wins': wins, 'loses': loses, 'draws': draws,
            'perish_togethers': perish_togethers,
            'kill_count': kill_count, 'die_count': die_count}


def run_HanSAC_simulation(
    collape_recover={"collapsed": False, "best_actor_name": None, "actor_frozen_batchs": 5},
    num_workers=10,
    mission_name='HanSAC无名',
    lr=1e-4,                    # 论文 lr=1e-4
    max_steps=4 * 165e4,
    gamma=0.9,                  # 论文 γ=0.9
    tau=0.05,                   # 论文式(39) τ=0.05
    batch_size=128,             # 论文 batch=128
    buffer_size=int(1e5),       # 论文 |D_BFM|=1e5
    updates_per_batch=256,      # 每批采样后的梯度更新次数（论文为每步交互后更新；此处按批折中）
    action_cycle_multiplier=10,
    dt_move=0.07,
    dt_maneuver=0.2,
    max_episode_duration=15*60,
    R_cage=62.00e3,
    red_init_num=None, blue_init_num=None,
    red_init_ammo=None, blue_init_ammo=None,
    # 自博弈相关（沿用 MAPPO 版基础设施）
    init_elo_ratings={"Rule_0": 1200, "Rule_1": 1200, "Rule_2": 1200},
    self_play_type='PFSP_balanced',
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
    num_runs=3,
    device=torch.device("cpu"),
    vertices=None,
    resume_dir=None,
    fire_buffer_size=1000,      # 论文 |D_f|=1e3
    fire_batch_size=128,        # 论文开火监督批 128
    kI=0.8, kD=0.8,             # 论文期望熵系数
    **_unused,
):
    """HanSAC 主训练函数：离策略 SAC + 开火命中监督（式 35 联合反传）+ 自博弈。"""
    from _context import load_obs_dims, load_init_states

    init_cfg = load_init_states()
    red_init_num = init_cfg['red_init_num'] if red_init_num is None else red_init_num
    blue_init_num = init_cfg['blue_init_num'] if blue_init_num is None else blue_init_num
    red_init_ammo = init_cfg['red_init_ammo'] if red_init_ammo is None else red_init_ammo
    blue_init_ammo = init_cfg['blue_init_ammo'] if blue_init_ammo is None else blue_init_ammo

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser("UAV swarm confrontation (HanSAC)")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration)
    parser.add_argument("--R-cage", type=float, default=R_cage)
    args = parser.parse_args([])

    print(f"Master training device: {device}")

    dummy_env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    dummy_env.reset(red_init_num=red_init_num, blue_init_num=blue_init_num,
                    red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo)
    red_student_ids = sorted([i for i in dummy_env.UAV_ids
                              if dummy_env.find_uav_with_id(i).side == 'r'])
    del dummy_env

    fixed_dim, enm_shape, ally_shape = load_obs_dims()

    # 策略-价值一体网络 + SAC 智能体（含目标网络与三个温度）
    ac = HanWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
    agent = HANSACHybrid(
        ac, lr=lr, gamma=gamma, tau=tau, batch_size=batch_size,
        buffer_size=buffer_size, fire_buffer_size=fire_buffer_size,
        fire_batch_size=fire_batch_size, kI=kI, kD=kD, device=device)

    # 日志目录
    logs_dir = os.path.join(project_root, "logs/combat")
    if resume_dir is not None and not os.path.exists(resume_dir):
        raise FileNotFoundError(f"resume_dir 不存在，已中止以防静默新训：{resume_dir}")
    if resume_dir is not None:
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" +
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

    # 断点续训：加载最近一次完整智能体状态
    batch_idx = 0
    if resume_dir is not None:
        agent_path = os.path.join(log_dir, "han_agent.pt")
        if os.path.exists(agent_path):
            agent.load_state_dict(torch.load(agent_path, map_location=device))
            actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
            if actor_files:
                batch_idx = max(int(re.search(r'actor_rein(\d+)', os.path.basename(f)).group(1))
                                for f in actor_files if re.search(r'actor_rein(\d+)', os.path.basename(f)))
            print(f"Loaded agent state from {agent_path}, batch_idx={batch_idx}")

    logger = TensorBoardLogger(
        log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    torch.save(ac.state_dict(), os.path.join(log_dir, "actor_rein0.pt"))
    torch.save(agent.state_dict(), os.path.join(log_dir, "han_agent.pt"))

    # 并行 worker（spawn）
    mp.set_start_method('spawn', force=True)
    test_pool = mp.Pool(processes=3, maxtasksperchild=10)
    workers, pipes = [], []
    worker_device = torch.device('cpu')
    args.max_episode_len = max_episode_duration
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process_han,
                       kwargs={'rank': i, 'pipe': child_conn, 'args': args,
                               'device_worker': worker_device, 'dt_maneuver': dt_maneuver,
                               'seed': seed, 'dt_move': dt_move, 'pomdp': 1, 'vertices': vertices})
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化（沿用 MAPPO 版口径）
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = {}
    hall_of_fame = {}
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")
    GameTimes = {}

    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f:
            elite_elo_ratings = json.load(f)
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f:
            hall_of_fame = json.load(f)

    main_agent_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)
    int_team_name = "actor_rein0"
    if not elite_elo_ratings:
        for k in init_elo_ratings.keys():
            elo_ratings[k] = main_agent_elo
        if hist_agent_as_opponent:
            elo_ratings[int_team_name] = main_agent_elo
        elite_elo_ratings = copy.deepcopy(elo_ratings)
        for k in elo_ratings.keys():
            GameTimes[k] = 0

    total_steps = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    current_max_steps = int(max_steps)

    ema_score, ema_step = 0.5, 0
    target_p1 = 0.65

    while True:
        while total_steps < current_max_steps:
            current_ac_weights = {k: v.cpu() for k, v in ac.state_dict().items()}

            # [测试] 每 batch 采样前：并行测试当前学生队 vs 规则对手（论文测试设定：argmax+阈值）
            if num_runs > 0:
                print(f"\n>>> [Per-Batch Test] Testing student team vs rules (batch {batch_idx})...")
                test_tasks = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        test_worker_han,
                        kwds={'model_state_dict': current_ac_weights, 'rules_num': r_idx,
                              'env_args': args, 'device_name': 'cpu', 'num_runs': num_runs,
                              'vertices': vertices, 'dt_move': dt_move, 'dt_maneuver': dt_maneuver,
                              'action_cycle_multiplier': action_cycle_multiplier,
                              'red_init_num': red_init_num, 'blue_init_num': blue_init_num,
                              'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo})
                    test_tasks.append(obj)
                test_results = [t.get() for t in test_tasks]
                outcomes = {r['rules_num']: r['result'] for r in test_results}
                outcomes_perish = {r['rules_num']: r['perish_togethers'] / num_runs
                                   for r in test_results}
                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score,
                               total_steps / action_cycle_multiplier)
                    print(f"  [Test Result] Rule_{r_num}: {score}")
                avg_score = float(np.mean(list(outcomes.values())))
                logger.add("test/avg_score", avg_score, total_steps / action_cycle_multiplier)
                logger.add("test/BVR perish together",
                           float(np.mean(list(outcomes_perish.values()))),
                           total_steps / action_cycle_multiplier)
                total_kills = sum(r['kill_count'] for r in test_results)
                total_dies = sum(r['die_count'] for r in test_results)
                logger.add("test/log_10_KD",
                           float(np.log10(np.clip(total_kills / (total_dies + 1e-6), 1/20, 20))),
                           total_steps / action_cycle_multiplier)

                test_csv_path = os.path.join(log_dir, "test_vs_rules.csv")
                _need_header = not os.path.exists(test_csv_path)
                with open(test_csv_path, 'a', newline='', encoding='utf-8-sig') as _f:
                    _w = csv.writer(_f)
                    if _need_header:
                        _w.writerow(["batch_idx", "steps", "rule_num", "wins", "loses",
                                     "draws", "score", "perish_together", "num_runs"])
                    for r in test_results:
                        _w.writerow([batch_idx, int(total_steps / action_cycle_multiplier),
                                     r['rules_num'], r['wins'], r['loses'], r['draws'],
                                     float(r['result']), float(r['perish_togethers'] / num_runs),
                                     num_runs])

            # 对手采样（PFSP/Elo，沿用原口径）
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
                    effective_pool, hall_of_fame, target_elo=main_agent_elo,
                    SP_type=self_play_type, sigma=sigma_elo, compete_old_rate=compete_old_rate,
                    deltaFSP_epsilon=deltaFSP_epsilon)
                selected_opponents.append(np.random.choice(opponent_keys, p=probs))

            for rank in range(num_workers):
                selected_opponent_name = selected_opponents[rank]
                GameTimes[selected_opponent_name] = GameTimes.get(selected_opponent_name, 0) + 1

                opp_type, opp_data = 'rule', 0
                if "Rule" in selected_opponent_name:
                    try:
                        opp_data = int(selected_opponent_name.split('_')[1])
                    except Exception:
                        opp_data = 0
                else:
                    opp_type = 'nn'
                    m = re.match(r'^actor_rein(\d+)$', selected_opponent_name)
                    if m:
                        ap = os.path.join(log_dir, f"actor_rein{int(m.group(1))}.pt")
                        if os.path.exists(ap):
                            shared_opp_sd = torch.load(ap, map_location='cpu')
                            opp_data = {uid + 200: shared_opp_sd for uid in red_student_ids}
                        else:
                            opp_type, opp_data = 'rule', 0
                    else:
                        opp_type, opp_data = 'rule', 0

                opp_info = (selected_opponent_name, opp_type, opp_data)
                rb, bb = create_initial_state_worker(randomized_birth)
                settings = {
                    'action_cycle_multiplier': action_cycle_multiplier,
                    'red_birth': rb, 'blue_birth': bb,
                    'red_init_num': red_init_num, 'blue_init_num': blue_init_num,
                    'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo,
                    'red_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
                    'blue_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
                }
                pipes[rank].send(('RUN_EPISODE', (current_ac_weights, opp_info, settings)))

            # Barrier 收取
            batch_results = []
            for rank in range(num_workers):
                try:
                    res = pipes[rank].recv()
                except EOFError:
                    for p in workers:
                        p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed silently.")
                if isinstance(res, dict) and 'error' in res:
                    for p in workers:
                        p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed with error:\n{res['error']}")
                batch_results.append(res)

            # 写入回放池 + 开火监督缓冲；统计战况
            batch_total_steps = 0
            batch_wins = batch_loss_cnt = batch_draw_cnt = 0
            batch_bvr_perish_together_cnt = 0
            batch_total_return = 0.0
            batch_return_by_id = {uid: 0 for uid in red_student_ids}
            new_transitions = 0
            worker_metrics_buffer = []

            for res in batch_results:
                metrics = res['metrics']
                opp_name = res['opp_name']
                worker_metrics_buffer.append(
                    f"{opp_name}: {'Win' if metrics['win'] else ('Lose' if metrics['lose'] else 'Draw')}")

                batch_total_steps += metrics['steps']
                batch_total_return += metrics['return']
                for uid, r_id in metrics.get('return_by_id', {}).items():
                    if uid in batch_return_by_id:
                        batch_return_by_id[uid] += r_id
                batch_bvr_perish_together_cnt += int(metrics.get('BVR_perish_together', False))
                batch_wins += int(metrics['win'])
                batch_loss_cnt += int(metrics['lose'])
                batch_draw_cnt += int(metrics['draw'])

                for uid, uid_trans in res['trans'].items():
                    n_i = len(uid_trans['dones']) if uid_trans['dones'] is not None else 0
                    if n_i > 0:
                        agent.push_stacked(uid_trans)
                        new_transitions += n_i

                fire_records = [r for r in res.get('fire_hit_records', [])
                                if r.get('hit_record') is not None]
                agent.push_fire_records(fire_records)

                # ELO 更新（同原口径）
                actual_score = 1.0 if metrics['win'] else (0.0 if metrics['lose'] else 0.5)
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
                if opp_name in hall_of_fame:
                    hall_of_fame[opp_name] = new_adv_elo

            print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            total_steps += batch_total_steps
            batch_idx += 1

            batch_score = (batch_wins + batch_draw_cnt * 0.5) / num_workers
            ema_step += 1
            ema_score = batch_score if ema_step == 1 else 0.9 * ema_score + 0.1 * batch_score
            step_axis = total_steps / action_cycle_multiplier
            logger.add("train_plus/batch_score", batch_score, step_axis)
            logger.add("train_plus/filtered_score", ema_score, step_axis)
            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, step_axis)
            for uid, r_id in batch_return_by_id.items():
                logger.add(f"train/1 avg_episode_return_id{uid}", r_id / num_workers, step_axis)
            logger.add("train/2 win", batch_wins / num_workers, step_axis)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, step_axis)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, step_axis)
            logger.add("agent/buffer_size", len(agent.buffer), step_axis)

            # SAC 更新（回放池够一批即开始）
            if batch_idx % save_interval == 0 and len(agent.buffer) >= agent.batch_size:
                n_up = int(updates_per_batch) if updates_per_batch is not None \
                    else max(1, new_transitions)
                m = agent.update(n_updates=n_up).metrics
                if m:
                    logger.add("train/7 loss_q", m.get('loss_q', 0.0), step_axis)
                    logger.add("train/8 loss_pi", m.get('loss_pi', 0.0), step_axis)
                    logger.add("train/9 H_J", m.get('H_J', 0.0), step_axis)
                    logger.add("train/9 H_I", m.get('H_I', 0.0), step_axis)
                    logger.add("train/9 H_D", m.get('H_D', 0.0), step_axis)
                    logger.add("train/10 alpha_J", m.get('alpha_J', 0.0), step_axis)
                    logger.add("train/10 alpha_I", m.get('alpha_I', 0.0), step_axis)
                    logger.add("train/10 alpha_D", m.get('alpha_D', 0.0), step_axis)
                    logger.add("train/supervised_fire_loss", m.get('loss_fire', 0.0), step_axis)
                    logger.add("train_plus/fire_buffer_size", m.get('fire_buffer', 0.0), step_axis)

                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}, "
                      f"ELO {main_agent_elo:.0f}, Buffer {len(agent.buffer)}")

                # 定期落盘：actor_rein{batch}.pt 供自博弈池加载，han_agent.pt 供续训
                actor_team_key = f"actor_rein{batch_idx}"
                if batch_idx % 10 == 0:
                    torch.save(ac.state_dict(), os.path.join(log_dir, f"{actor_team_key}.pt"))
                    torch.save(agent.state_dict(), os.path.join(log_dir, "han_agent.pt"))
                    elo_ratings[actor_team_key] = main_agent_elo
                    print(f"Saved Checkpoint team: {actor_team_key}")

                # 精英池维护（沿用原口径）
                if total_steps >= WARM_UP_STEPS:
                    elo_ratings[opp_name] = main_agent_elo
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx

                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    rule_elos = sorted([v for k, v in elite_elo_ratings.items()
                                        if k.startswith("Rule")], reverse=False)
                    if not rule_elos or ADMISSION_THRESHOLD < 0:
                        rule_elo_thres = main_agent_elo
                    else:
                        t = np.clip(ADMISSION_THRESHOLD, 0, 1)
                        idx = np.clip(int(t * (len(rule_elos) - 1) + 0.5), 0, len(rule_elos) - 1)
                        rule_elo_thres = rule_elos[idx]
                    if main_agent_elo >= rule_elo_thres:
                        history_keys = [k for k in elite_elo_ratings.keys()
                                        if not k.startswith("Rule") and not k.startswith("__")]
                        while len(history_keys) >= MAX_HISTORY_SIZE:
                            weakest = min(history_keys, key=lambda k: elite_elo_ratings[k])
                            del elite_elo_ratings[weakest]
                            history_keys.remove(weakest)
                        if hist_agent_as_opponent:
                            elite_elo_ratings[actor_team_key] = main_agent_elo
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, step_axis)
                    logger.add("Elo/History_Pool_Size",
                               len([k for k in valid_elos if not k.startswith("Rule")]), step_axis)

                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
                with open(elite_json_path, "w", encoding='utf-8') as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        print("All simulation workers are now idling safely. System is paused.")
        inp = input(f"Enter new max_steps (current {total_steps}), or press Enter to exit: ")
        if not inp.strip():
            print("No input provided. Exiting training.")
            break
        try:
            new_max = int(inp)
            if new_max > total_steps:
                current_max_steps = new_max
                print(f"Continuing training until {current_max_steps} steps.")
            else:
                print("Input steps less than current. Exiting...")
                break
        except ValueError:
            print("Invalid input (not a number). Exiting...")
            break

    print("Closing workers...")
    for pipe in pipes:
        try:
            pipe.send(('EXIT', None))
        except Exception:
            pass
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    try:
        test_pool.close()
        test_pool.join()
    except Exception:
        pass
    logger.close()
    print("Training Finished.")


if __name__ == '__main__':
    # 与《二对二训练同构训练MAPPO.py》同口径的默认配置；SAC 超参按论文（γ=0.9, τ=0.05, lr=1e-4, batch=128）
    dt_decide = 2
    dt_maneuver = 0.2
    action_cycle_multiplier = int(round(dt_decide / dt_maneuver))
    MAX_DEC_STEPS = 1e6

    run_HanSAC_simulation(
        num_workers=15,
        mission_name='HanSAC_homo_JPO',
        lr=1e-4,
        max_steps=MAX_DEC_STEPS * action_cycle_multiplier,
        gamma=0.9,
        tau=0.05,
        batch_size=128,
        buffer_size=int(1e5),
        updates_per_batch=256,
        action_cycle_multiplier=action_cycle_multiplier,
        dt_move=0.07,
        max_episode_duration=15 * 60,
        R_cage=62.00e3,
        dt_maneuver=dt_maneuver,
        init_elo_ratings={'Rule_0': 1200, 'Rule_1': 1200, 'Rule_2': 1200, 'Rule_3': 1200},
        self_play_type='PFSP_balanced',
        hist_agent_as_opponent=1,
        WARM_UP_STEPS=0e3,
        ADMISSION_THRESHOLD=-1,
        MAX_HISTORY_SIZE=50,
        compete_old_rate=0.2,
        K_FACTOR=16,
        randomized_birth=1,
        save_interval=1,
        num_runs=3,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        vertices=None,
        resume_dir=None,
        fire_buffer_size=1000,
        fire_batch_size=128,
        kI=0.8, kD=0.8,
    )
