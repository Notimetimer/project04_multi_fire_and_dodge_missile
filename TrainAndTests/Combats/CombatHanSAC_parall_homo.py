"""CombatHanSAC_parall_homo：Han et al. (2026) JPO 路线的 2V2 同构离策略训练入口。

并行采样/对手池/Elo 等基础设施沿用 CombatMAPPO_parall_homo.py 的约定（不改既有文件）：
- 流式多 worker 采样（spawn）：worker 按回合接收 assignment（权重按回合粒度同步），
  回合内每个决策点把闭合转移即时流式回传 master，回合结束立即回传末端转移；
- 蓝方 = 规则对手 或 精英池历史 checkpoint（PFSP_balanced，Elo 高斯核）；
- 开火命中记录由环境回填、worker 随回合结束回传。

与 MAPPO 版本的核心差异（论文 JPO 路线）：
- 离策略：worker 流式回传的转移在 master 侧按 worker lane 堆叠隔离，攒够
  update_batch_points 个决策点后临时降维写入 D_BFM 回放池（1e5），并按
  points_per_update（论文口径：每次交互 1 次更新）触发 SAC 更新
  （双 Q + 目标网络 τ=0.05 + 三温度自动调节）；转移用 terminated/truncated
  区分环境终结与时限截断，末端转移不足 min_terminal_samples 时不更新（末端门控）；
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
from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
from Algorithms.HanWrapper import HanWrapper
from Algorithms.HANSACHybrid import HANSACHybrid
from Math_calculates.sub_of_angles import sub_of_radian, sub_of_degree
from Visualize.tensorboard_visualize import TensorBoardLogger
from SelfPlayCalculation import calculate_expected_score, update_elo, get_opponent_probabilities

dt_move = 0.04

# 测试对战初始弹药：统一从 init_states.json 读取，避免硬编码与训练配置脱节
_init_cfg = load_init_states()
red_init_ammo = _init_cfg['red_init_ammo']
blue_init_ammo = _init_cfg['blue_init_ammo']

# 周期性测试所对抗的规则对手编号 (仅测试 0, 1, 2, 3)
TEST_RULE_IDS = [0, 1, 2, 3]


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方（复制原逻辑，保持初始化分布一致）
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                8000.0 * (1-int(randomized))
    if np.random.uniform(0,1) < 0.2 and int(randomized): # 以很低概率在更大的范围随机初始高度
        blue_height = np.random.uniform(4000.0, 12000.0)
    red_height = blue_height
    
    init_North = np.random.uniform(-20e3, 20e3) * int(randomized)
    # 红方初始化
    red_N = init_North
    red_E = - 55e3 # 45e3
    red_psi = sub_of_radian(np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    if np.random.uniform(0,1) < 0.2: # 以低概率随机初始距离
        red_E = np.random.uniform(-40e3, -55e3)
    # 蓝方对称初始化
    blue_N = init_North
    blue_E = - red_E
    blue_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-np.pi*1/4, np.pi*1/4))
    if np.random.uniform(0,1) < 0.5: # 随机换边
        red_E = - red_E
        blue_E = - blue_E
        red_psi = - red_psi
        blue_psi = - blue_psi

    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process_han(rank, pipe, args, device_worker, dt_maneuver, seed, dt_move=0.07, pomdp=1, vertices=None):
    """采样 worker：红方学生共享一个 HanWrapper（同构），蓝方为规则或历史 nn。
    离策略：每个决策点产出 (obs, a_flat, r, next_obs, terminated/truncated) 转移并即时
    经 pipe 流式回传 master；回合结束发末端转移（区分 terminated/truncated）与回合统计。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.HanWrapper import HanWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims, load_init_states

    # 单进程单线程：worker 每决策点只做 batch=1 的 actor 前向，多线程在 20 逻辑核上会与
    # 其余 worker 争抢并产生线程自旋开销（实测 14 线程比 1 线程更慢）
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

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
                opp_name, opp_type, opp_data = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])

                # 加载红方学生共享 actor 权重（学生 id 与环境红机 id 一致，全队共用一个实例）
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
                # 流式：不在 worker 侧攒局，闭合一条即发一条（master 侧按 worker lane 堆叠隔离）
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
                        closed_trans = {}
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            is_student = UAV.side == 'r'

                            # 上一决策点的转移在此闭环：r 为上一周期结算回报，next_obs 为当前观测
                            if is_student and Last_Obs[id] is not None and count > 0:
                                closed_trans[id] = {
                                    'obs': Last_Obs[id], 'actions': Last_Actions[id],
                                    'rewards': Rewards[id], 'next_obs': _obs_triple(id),
                                    'terminateds': 0, 'truncateds': 0,
                                    'active_masks': not UAV.dead}

                            if is_student or id in blue_nn_weights:
                                # 神经网络决策（红方学生走共享实例 / 蓝方历史 nn 按 id 取）
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
                                # 记录动作供经验池使用（仅学生，log_prob 为行为策略真实值，update 直接读取）
                                if is_student:
                                    Last_Actions[id] = {
                                        'flat_idx': int(actor_out['flat_idx'].item()),
                                        'enm_idx': enm_idx, 'vert': vert, 'hor': hor,
                                        'fire': fire, 'fire_target': fire_target,
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
                                Last_Obs[id] = _obs_triple(id)

                            if fire_flag:
                                try:
                                    env.fire_120(id, fire_target, tabu=0)
                                except ValueError:
                                    pass
                            env.radar_action(id, 1)

                        # 流式回传：本决策点闭合的转移立即发给 master（rank 标记供 lane 隔离）
                        if closed_trans:
                            pipe.send({'type': 'trans', 'rank': rank,
                                       'n_points': 1, 'payload': closed_trans})

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

                # 最后一帧转移：terminated=环境终结（不 bootstrap），truncated=时限截断（照常 bootstrap）
                terminated = bool(done) or (not env.running)
                final_next = {}
                for id in student_ids:
                    if terminated:
                        final_next[id] = Last_Obs[id]  # 终止帧 next_obs 不参与 bootstrap
                    else:
                        # 截断帧需要真实末端观测用于 bootstrap
                        ai_, mi_, ri_ = env.get_obs(id, pomdp=pomdp)
                        final_next[id] = {'agent_input': ai_, 'mask_input': mi_,
                                          'rule_input': take_some_of_rule_inputs(ri_)}
                final_trans = {}
                for id in student_ids:
                    UAV = env.find_uav_with_id(id)
                    final_trans[id] = {
                        'obs': Last_Obs[id], 'actions': Last_Actions[id],
                        'rewards': Rewards[id], 'next_obs': final_next[id],
                        'terminateds': int(terminated), 'truncateds': int(not terminated),
                        'active_masks': not UAV.dead}
                pipe.send({'type': 'trans', 'rank': rank, 'n_points': 0,
                           'payload': final_trans})  # 末端转移即时回传，不计决策点

                outcome = env.r_outcome if env.r_outcome is not None else 0
                fire_hit_records = []
                for uav in env.UAVs:
                    fire_hit_records += uav.fire_hit_records

                pipe.send({
                    'type': 'episode_end', 'rank': rank,
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
                    temperature=None, enm_chosen_visable=0, use_tacview=0):
    """周期性规则测试：论文设定 —— 联合分布 argmax + 开火命中概率阈值 0.8（deterministic）。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.HanWrapper import HanWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims

    # 测试池进程同样只做 batch=1 前向，限制为单线程以免与训练 worker 抢核
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    env = ChooseStrategyEnv(env_args, tacview_show=use_tacview, vertices=vertices)
    env.dt_move = dt_move
    env.dt_maneuver = dt_maneuver
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    # 部署规模由传入机数决定：敌机槽位=蓝方机数，队友槽位=红方机数-1（不含自己）
    enm_shape = (int(blue_init_num), enm_shape[1])
    ally_shape = (max(int(red_init_num) - 1, 0), ally_shape[1])

    actor = HanWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                       device=torch.device(device_name))
    actor.load_state_dict(model_state_dict)
    actor.eval()

    wins = loses = draws = perish_togethers = 0
    kill_count = die_count = 0
    # 多场共用一条 acmi 时间轴：每场结束后累加 env.t，避免轨迹在 t=0 处重叠
    t_bias = 0

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
            # 可视化：唯一写 acmi 的入口就是 render（tacview_show=0 时内部直接返回）
            env.render(t_bias=t_bias)
            if done:
                break

        # 单场轨迹收尾：清空残余缓存并把时间轴累加，多场写入同一份 acmi 时首尾相接
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

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
    max_steps_of_maneuver=4 * 165e4,
    gamma=0.9,                  # 论文 γ=0.9
    tau=0.05,                   # 论文式(39) τ=0.05
    batch_size=128,             # 论文 batch=128
    buffer_size=int(1e5),       # 论文 |D_BFM|=1e5
    min_terminal_samples=32,    # 末端门控：池中 terminated+truncated 帧累计达该数才开始更新
    update_batch_points=16,     # 每攒够多少决策点触发一次 lane flush + 梯度更新
    points_per_update=1,        # 论文口径：每次交互（决策点）对应 1 次梯度更新
    action_cycle_multiplier=1,
    dt_move=0.07,
    dt_maneuver=0.2,
    max_episode_duration=15*60,
    R_cage=62.00e3,
    red_init_num=None, blue_init_num=None,
    red_init_ammo=None, blue_init_ammo=None,
    # 自博弈相关
    init_elo_ratings={"Rule_0": 1200, "Rule_1": 1200, "Rule_2": 1200},
    self_play_type='PFSP_balanced',
    hist_agent_as_opponent=1,
    sigma_elo=400,
    WARM_UP_STEPS=500e3,
    MAX_HISTORY_SIZE=50,
    deltaFSP_epsilon=0.8,
    compete_old_rate=0.2,
    K_FACTOR=16,
    randomized_birth=1,
    test_interval_dp=3333,       # 测试窗口触发间隔：累计 2000 决策点触发一次完整测试+日志+保存
    num_runs=3,
    device=torch.device("cpu"),
    vertices=None,
    resume_dir=None,
    pause_on_finish=1,          # 达标后是否进入续训提示符（后台/调试运行设 0 直接退出）
    fire_buffer_size=1000,      # 论文 |D_f|=1e3
    fire_batch_size=128,        # 论文开火监督批 128
    kI=0.8, kD=0.8,             # 论文期望熵系数
    **_unused,
):
    """HanSAC 主训练函数：离策略 SAC + 开火命中监督（式 35 联合反传）+ 自博弈。"""
    from _context import load_obs_dims, load_init_states
    from multiprocessing.connection import wait as conn_wait

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

    parser = argparse.ArgumentParser("UAV swarm confrontation (HanSAC)")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration)
    parser.add_argument("--R-cage", type=float, default=R_cage)
    args = parser.parse_args([])

    print(f"Master training device: {device}")

    # master 只做 batch=128 的小网络更新，实测单线程最快（2/4/14 线程分别约 15.9/22.8/19.0 ms，
    # 单线程 11.3 ms）；同时把核留给 15 个采样 worker 与 3 个测试进程
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # 3. 网络维度从 json 读取，用 dummy env 获取 UAV id 列表
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
        fire_batch_size=fire_batch_size, min_terminal_samples=min_terminal_samples,
        kI=kI, kD=kD, device=device)

    # 日志目录
    logs_dir = os.path.join(project_root, "logs/combat")
    if resume_dir is not None and not os.path.exists(resume_dir):
        raise FileNotFoundError(f"resume_dir 不存在，已中止以防静默新训：{resume_dir}")
    if resume_dir is not None and os.path.exists(resume_dir):
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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

    # 存储在线训练前的网络参数（同构单份）
    # actor_rein0 只在首次创建时落盘；续训时若已存在则保留原始初始策略，避免被 latest 权重覆盖
    if not os.path.exists(os.path.join(log_dir, "actor_rein0.pt")):
        torch.save(ac.state_dict(), os.path.join(log_dir, "actor_rein0.pt"))
    torch.save(agent.state_dict(), os.path.join(log_dir, "han_agent.pt"))

    # 开火记录滚动缓冲：跨 batch 累积（FIFO 逐条覆盖）；D_f 由 HANSACHybrid 内部维护，
    # 脚本层只在续训时与磁盘 pkl 交互（格式与 MAPPO 版一致，观测配置相同时可互通）
    # 若中断续训，恢复已保存的开火监督 buffer
    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
    if os.path.exists(fire_buffer_path):
        try:
            with open(fire_buffer_path, "rb") as f:
                fb_data = pickle.load(f)
            if isinstance(fb_data, dict):
                loaded_records = fb_data.get("records", [])
                loaded_maxlen = fb_data.get("maxlen", int(fire_buffer_size))
            elif isinstance(fb_data, collections.deque):
                loaded_records = list(fb_data)
                loaded_maxlen = fb_data.maxlen
            else:
                loaded_records = list(fb_data)
                loaded_maxlen = int(fire_buffer_size)
            agent.fire_buffer.clear()
            agent.fire_buffer.extend(loaded_records[-loaded_maxlen:])
            print(f"Loaded fire records buffer from {fire_buffer_path}. Size: {len(agent.fire_buffer)}")
        except Exception as e:
            print(f"Failed to load fire records buffer: {e}")

    # 并行环境初始化
    mp.set_start_method('spawn', force=True)
    test_pool = mp.Pool(processes=len(TEST_RULE_IDS), maxtasksperchild=10)

    # 启动并行训练 Worker
    workers = []
    pipes = []
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

    # ELO 初始化
    # available_opps：可用对手注册表 {name: version}，规则版本号统一为 0，NN 版本号=batch_idx；
    #              仅记录名称→版本号，不再承载 Elo 分（分数只存在 elite_elo_ratings）。
    #              其中 __CURRENT_MAIN__ / __LAST_UPDATE_STEP__ / __LAST_UPDATE_BATCH__ 为训练状态元数据。
    available_opps = {k: 0 for k in init_elo_ratings.keys()}  # 规则版本号统一为 0
    # elite_elo_ratings：限定容量的 Elo 表 {name: elo_score}，满员在 NN 内部按 Elo 踢出最弱（规则永不踢出）
    elite_elo_ratings = {}

    full_json_path = os.path.join(log_dir, "available_opps.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")

    GameTimes = {}

    # 尝试加载历史 # 中断续训
    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f: available_opps = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)

    # 续训清洗：剔除本地没有对应权重文件的 NN 快照，防止采样到加载失败的幽灵对手
    def _opp_ckpt_exists(name):
        if name.startswith("Rule"):
            return True
        return os.path.exists(os.path.join(log_dir, f"{name}.pt"))
    available_opps = {k: v for k, v in available_opps.items()
                   if k.startswith("__") or _opp_ckpt_exists(k)}
    # elite 池只保留真实对手（规则/NN 快照）；__CURRENT_MAIN__ 等元数据键只存于 available_opps，
    # elite json 里残留的 __ 键必须剔除，否则会被当成对手采样
    elite_elo_ratings = {k: v for k, v in elite_elo_ratings.items()
                         if not k.startswith("__") and _opp_ckpt_exists(k)}

    main_agent_elo = available_opps.get("__CURRENT_MAIN__", 1200)

    # 团队命名：actor_rein{batch} 代表一个团队（含所有 id）
    int_team_name = "actor_rein0"

    # 不论如何，记录在线训练前的网络参数
    if not elite_elo_ratings:
        # 规则初始 Elo 取自 init_elo_ratings（仅用于 elite 池，不再写入 available_opps）
        elite_elo_ratings = copy.deepcopy(init_elo_ratings)
        # 可用注册表：规则版本号 0
        available_opps = {k: 0 for k in init_elo_ratings.keys()}
        if hist_agent_as_opponent:
            available_opps[int_team_name] = 0  # actor_rein0 版本号 0
            elite_elo_ratings[int_team_name] = main_agent_elo
        for k in elite_elo_ratings.keys():
            GameTimes[k] = 0

    # 首次落盘 ELO 元数据（续训依赖，不受测试窗口门槛影响）
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(available_opps, f, ensure_ascii=False, indent=2)
    with open(elite_json_path, "w", encoding="utf-8") as f:
        json.dump(copy.deepcopy(elite_elo_ratings), f, ensure_ascii=False, indent=2)

    # 训练循环变量
    total_steps_of_maneuver = available_opps.get("__LAST_UPDATE_STEP__", 0)
    current_max_steps_of_maneuver = int(max_steps_of_maneuver)

    ema_score, ema_step = 0.5, 0
    target_p1 = 0.65

    # 主循环
    while True:
        # ============ 流式主循环（方案 A） ============
        # worker 按回合接收 assignment（权重按回合粒度同步），回合内每个决策点把闭合转移
        # 即时流式回传；master 事件驱动收取，转移在各 worker 的 lane 内按时间维堆叠（worker
        # 间数据隔离），攒够 update_batch_points 个决策点后堆叠→push_stacked 临时降维入池，
        # 并按 points_per_update（论文口径 1 决策点 : 1 次更新）触发 SAC 更新。
        decision_points_total = int(total_steps_of_maneuver / action_cycle_multiplier)
        next_test_dp = decision_points_total + test_interval_dp
        pending_points = 0
        # 吞吐观测窗口：dp/s（采样）、updates/s（更新密度）、ms/update 与 master 更新占用率，
        # 用于区分“采样受限”与“更新受限”，不参与训练逻辑
        perf_win_t = time.perf_counter()
        perf_win_dp = 0
        perf_win_updates = 0
        perf_win_update_time = 0.0
        perf_win_flushes = 0
        local_trans = {r: {} for r in range(num_workers)}  # rank -> uid -> {field: [t=0..]}
        episodes_window = []      # 本测试窗口内的 (opp_name, metrics)
        last_update_metrics = {}
        gate_notices = 0
        stopping = False
        idle_ranks = set()

        def make_assignment(rank):
            """PFSP 选对手 + 出生状态 + 最新权重，构成一局 assignment（回合粒度权重同步）。"""
            # 对手池：采样一律从限定容量的 elite_elo_ratings（真实 Elo 表）取
            effective_pool = elite_elo_ratings

            probs, opponent_keys = get_opponent_probabilities(
                effective_pool, target_elo=main_agent_elo,
                SP_type=self_play_type, sigma=sigma_elo, compete_old_rate=compete_old_rate,
                deltaFSP_epsilon=deltaFSP_epsilon)
            selected_opponent_name = np.random.choice(opponent_keys, p=probs)
            GameTimes[selected_opponent_name] = GameTimes.get(selected_opponent_name, 0) + 1

            opp_type, opp_data = 'rule', 0
            if "Rule" in selected_opponent_name:
                try:
                    opp_data = int(selected_opponent_name.split('_')[1])
                except Exception:
                    opp_data = 0
            else:
                opp_type = 'nn'
                # 历史团队：读同构单份权重，按蓝机 id 打包（同一份 sd 多 key 引用，pickle 自动去重）
                m = re.match(r'^actor_rein(\d+)$', selected_opponent_name)
                if m:
                    old_batch = int(m.group(1))
                    ap = os.path.join(log_dir, f"actor_rein{old_batch}.pt")
                    if not os.path.exists(ap):
                        # 权重文件缺失：直接抛错，禁止静默降级到 Rule_0
                        raise FileNotFoundError(
                            f"对手 {selected_opponent_name} 的权重文件不存在: {ap}")
                    shared_opp_sd = torch.load(ap, map_location='cpu')
                    # 蓝机 id = 学生 id + 200（同编队位次：1→201, 2→202）
                    opp_data = {uid + 200: shared_opp_sd for uid in red_student_ids}
                else:
                    raise ValueError(f"无法解析的 NN 对手名称: {selected_opponent_name}")

            opp_info = (selected_opponent_name, opp_type, opp_data)

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
            current_ac_weights = {k: v.cpu() for k, v in ac.state_dict().items()}
            return ('RUN_EPISODE', (current_ac_weights, opp_info, settings))

        for r in range(num_workers):
            pipes[r].send(make_assignment(r))

        while True:
            # 目标达成且所有 worker 已空闲（回合自然结束、未再派新局）→ 退出到提示符
            if stopping and len(idle_ranks) == num_workers:
                break
            ready = conn_wait(pipes)
            for conn in ready:
                r = pipes.index(conn)
                try:
                    msg = conn.recv()
                except EOFError:
                    for p in workers:
                        p.terminate()
                    raise RuntimeError(f"Worker {r} crashed silently.")
                if isinstance(msg, dict) and 'error' in msg:
                    for p in workers:
                        p.terminate()
                    raise RuntimeError(f"Worker {r} crashed with error:\n{msg['error']}")

                if msg.get('type') == 'trans':
                    if stopping:
                        continue
                    lane = local_trans[r]
                    for uid, tr in msg['payload'].items():
                        slot = lane.setdefault(uid, {k: [] for k in tr})
                        for k, v in tr.items():
                            slot[k].append(v)
                    pending_points += msg['n_points']
                    perf_win_dp += msg['n_points']
                    decision_points_total += msg['n_points']
                    total_steps_of_maneuver = decision_points_total * action_cycle_multiplier

                elif msg.get('type') == 'episode_end':
                    metrics = msg['metrics']
                    opp_name = msg['opp_name']
                    episodes_window.append((opp_name, metrics))
                    fire_records = [x for x in msg.get('fire_hit_records', [])
                                    if x.get('hit_record') is not None]
                    agent.push_fire_records(fire_records)

                    # ELO 更新：仅当对手已入 elite 池时才更新双方 Elo；未入池对手不参与 Elo 记账
                    actual_score = 1.0 if metrics['win'] else (0.0 if metrics['lose'] else 0.5)
                    if opp_name in elite_elo_ratings:
                        prev_main_elo = main_agent_elo
                        adv_elo = elite_elo_ratings[opp_name]
                        main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                        new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                        available_opps["__CURRENT_MAIN__"] = main_agent_elo
                        elite_elo_ratings[opp_name] = new_adv_elo

                    if decision_points_total * action_cycle_multiplier >= current_max_steps_of_maneuver:
                        stopping = True
                    if stopping:
                        idle_ranks.add(r)   # 不再派新局，worker 阻塞等待即为安全空闲
                    else:
                        pipes[r].send(make_assignment(r))

            # ---- 步数触发：lanes 堆叠 → push_stacked 临时降维入池 → 按比例更新 ----
            if pending_points >= update_batch_points:
                for r, lane in local_trans.items():
                    for uid, fields in lane.items():
                        if fields['terminateds']:
                            agent.push_stacked({k: stack_field(v) for k, v in fields.items()})
                local_trans = {r: {} for r in range(num_workers)}
                n_up = max(1, pending_points // max(1, int(points_per_update)))
                _t_upd = time.perf_counter()
                last_update_metrics = agent.update(n_updates=n_up).metrics  # 末端门控未开时返回 {}
                perf_win_update_time += time.perf_counter() - _t_upd
                perf_win_updates += n_up if last_update_metrics else 0
                pending_points = 0
                perf_win_flushes += 1
                if perf_win_flushes % 20 == 0:
                    _el = max(1e-9, time.perf_counter() - perf_win_t)
                    print(f"  [Perf] {perf_win_dp / _el:.1f} dp/s | "
                          f"updates {perf_win_updates / _el:.1f}/s | "
                          f"{1000 * perf_win_update_time / max(1, perf_win_updates):.1f} ms/update | "
                          f"master update duty {100 * perf_win_update_time / _el:.0f}% | "
                          f"replay {len(agent.buffer)}")
                    perf_win_t = time.perf_counter()
                    perf_win_dp = 0
                    perf_win_updates = 0
                    perf_win_update_time = 0.0
                if not last_update_metrics:
                    gate_notices += 1
                    if gate_notices % 25 == 1:
                        print(f"  [Gate] 末端门控未开：buffer {len(agent.buffer)}/{agent.batch_size}, "
                              f"ended {agent.buffer.n_ended}/{agent.min_terminal_samples}")

            # ---- 测试窗口：每累计 test_interval_dp（默认 2000）决策点做一轮规则测试 + 日志 + checkpoint ----
            if decision_points_total >= next_test_dp:
                batch_idx += 1
                step_axis = float(decision_points_total)

                # [测试] 并行测试当前学生队 vs 规则对手（论文测试设定：argmax+阈值）
                if num_runs > 0:
                    print(f"\n>>> [Per-Batch Test] Testing student team vs rules (batch {batch_idx})...")
                    current_ac_weights = {k: v.cpu() for k, v in ac.state_dict().items()}
                    test_tasks = []
                    for r_idx in TEST_RULE_IDS:
                        obj = test_pool.apply_async(
                            test_worker_han,
                            kwds={'model_state_dict': current_ac_weights, 'rules_num': r_idx,
                                  'env_args': args, 'device_name': 'cpu', 'num_runs': num_runs,
                                  'vertices': vertices, 'dt_move': dt_move, 'dt_maneuver': 0.5, # dt_maneuver, # 测试的时候需要精细，训练的时候可以粗略
                                  'action_cycle_multiplier': action_cycle_multiplier,
                                  'red_init_num': red_init_num, 'blue_init_num': blue_init_num,
                                  'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo})
                        test_tasks.append(obj)
                    test_results = [t.get() for t in test_tasks]
                    outcomes = {r['rules_num']: r['result'] for r in test_results}
                    outcomes_perish = {r['rules_num']: r['perish_togethers'] / num_runs
                                       for r in test_results}
                    for r_num, score in outcomes.items():
                        logger.add(f"test/agent_vs_rule{r_num}", score, step_axis)
                        print(f"  [Test Result] Rule_{r_num}: {score}")
                    avg_score = float(np.mean(list(outcomes.values())))
                    logger.add("test/avg_score", avg_score, step_axis)
                    logger.add("test/BVR perish together",
                               float(np.mean(list(outcomes_perish.values()))), step_axis)
                    total_kills = sum(r['kill_count'] for r in test_results)
                    total_dies = sum(r['die_count'] for r in test_results)
                    logger.add("test/log_10_KD",
                               float(np.log10(np.clip(total_kills / (total_dies + 1e-6), 1/20, 20))),
                               step_axis)

                    test_csv_path = os.path.join(log_dir, "test_vs_rules.csv")
                    _need_header = not os.path.exists(test_csv_path)
                    with open(test_csv_path, 'a', newline='', encoding='utf-8-sig') as _f:
                        _w = csv.writer(_f)
                        if _need_header:
                            _w.writerow(["batch_idx", "steps", "rule_num", "wins", "loses",
                                         "draws", "score", "perish_together", "num_runs"])
                        for r in test_results:
                            _w.writerow([batch_idx, int(step_axis), r['rules_num'], r['wins'],
                                         r['loses'], r['draws'], float(r['result']),
                                         float(r['perish_togethers'] / num_runs), num_runs])

                # 窗口内回合统计（替代原批次统计）
                n_win_eps = len(episodes_window)
                batch_wins = sum(int(m['win']) for _, m in episodes_window)
                batch_loss_cnt = sum(int(m['lose']) for _, m in episodes_window)
                batch_draw_cnt = sum(int(m['draw']) for _, m in episodes_window)
                batch_total_return = sum(m['return'] for _, m in episodes_window)
                batch_return_by_id = {uid: 0.0 for uid in red_student_ids}
                for _, m in episodes_window:
                    for uid, r_id in m.get('return_by_id', {}).items():
                        if uid in batch_return_by_id:
                            batch_return_by_id[uid] += r_id

                print(f"  [Batch {batch_idx}] Results: " + ', '.join(
                    f"{o}: {'Win' if m['win'] else ('Lose' if m['lose'] else 'Draw')}"
                    for o, m in episodes_window))

                batch_score = (batch_wins + batch_draw_cnt * 0.5) / n_win_eps
                ema_step += 1
                ema_score = batch_score if ema_step == 1 else 0.9 * ema_score + 0.1 * batch_score
                logger.add("train_plus/batch_score", batch_score, step_axis)
                logger.add("train_plus/filtered_score", ema_score, step_axis)
                logger.add("train/1 avg_episode_return", batch_total_return / n_win_eps, step_axis)
                for uid, r_id in batch_return_by_id.items():
                    logger.add(f"train/1 avg_episode_return_id{uid}", r_id / n_win_eps, step_axis)
                logger.add("train/2 win", batch_wins / n_win_eps, step_axis)
                logger.add("train/2 lose", batch_loss_cnt / n_win_eps, step_axis)
                logger.add("train/2 draw", batch_draw_cnt / n_win_eps, step_axis)
                logger.add("agent/buffer_size", len(agent.buffer), step_axis)
                logger.add("agent/buffer_ended", agent.buffer.n_ended, step_axis)

                if last_update_metrics:
                    m = last_update_metrics
                    logger.add("train/7 loss_q", m.get('loss_q', 0.0), step_axis)
                    logger.add("train/8 loss_pi", m.get('loss_pi', 0.0), step_axis)
                    logger.add("train/9 H_J", m.get('H_J', 0.0), step_axis)
                    logger.add("train/9 H_D", m.get('H_D', 0.0), step_axis)
                    logger.add("train/9 H_I", m.get('H_I', 0.0), step_axis)
                    logger.add("train/9 B_J", m.get('B_J', 0.0), step_axis)
                    logger.add("train/10 alpha_J", m.get('alpha_J', 0.0), step_axis)
                    logger.add("train/10 alpha_D", m.get('alpha_D', 0.0), step_axis)
                    logger.add("train/10 alpha_I", m.get('alpha_I', 0.0), step_axis)
                    logger.add("train/supervised_fire_loss", m.get('loss_fire', 0.0), step_axis)
                    logger.add("train_plus/fire_buffer_size", m.get('fire_buffer', 0.0), step_axis)

                print(f"Step {total_steps_of_maneuver}: Batch WinRate {batch_wins}/{n_win_eps}, "
                      f"ELO {main_agent_elo:.0f}, Buffer {len(agent.buffer)}")

                # ELO 相对位置（基于 elite 池中的真实 Elo，available_opps 仅存版本号不可再用）
                all_keys = list(elite_elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                target_pool_keys = rule_keys + latest_rein_keys
                avg_pool_elo = np.mean([elite_elo_ratings[k] for k in target_pool_keys]) \
                    if target_pool_keys else main_agent_elo
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, step_axis)

                # 定期落盘：actor_rein{batch}.pt 供自博弈池加载，han_agent.pt 供续训
                actor_team_key = f"actor_rein{batch_idx}"
                if batch_idx % 10 == 0:
                    torch.save(ac.state_dict(), os.path.join(log_dir, f"{actor_team_key}.pt"))
                    torch.save(agent.state_dict(), os.path.join(log_dir, "han_agent.pt"))
                    print(f"Saved Checkpoint team: {actor_team_key}")
                    # 每次落盘新的 actor_rein 时记录：横轴当前步数，纵轴 batch_idx
                    logger.add("checkpoint/actor_rein_idx", batch_idx, step_axis)
                    # 只注册已落盘的版本到可用注册表（版本号=batch_idx），保证采样时总能加载到权重
                    available_opps[actor_team_key] = batch_idx

                # 保存开火监督数据 buffer
                fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
                with open(fire_buffer_path, "wb") as f:
                    pickle.dump({"records": list(agent.fire_buffer), "maxlen": agent.fire_buffer.maxlen}, f)
                print(f"Saved fire records buffer to {fire_buffer_path}. Size: {len(agent.fire_buffer)}")

                # 全量历史记录（训练状态元数据只存 available_opps）
                available_opps["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                available_opps["__LAST_UPDATE_BATCH__"] = batch_idx

                # 保存全量日志
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(available_opps, f, ensure_ascii=False, indent=2)

                # 精英池快照（不再往 save_elite 塞 __CURRENT_MAIN__，元数据只存 available_opps）
                save_elite = copy.deepcopy(elite_elo_ratings)
                with open(elite_json_path, "w", encoding='utf-8') as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)

                # 日志记录：展示精英池状态
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))

                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, step_axis)

                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))

                    # 升序：Elo 低（弱对手）在前
                    rule_elos = sorted([v for k, v in elite_elo_ratings.items()
                                        if k.startswith("Rule")], reverse=False)

                    # ELO 极差（max - min），用于判断 PFSP sigma 是否合适
                    elo_spread = float(max_elo - min_elo)
                    logger.add("Elo/Spread", elo_spread, step_axis)

                    # 口径：把 main 合入规则池一起归一化（不含历史 RL 对手），
                    # 0%=最弱 Rule（或 main 自己垫底），100%=最强 Rule（或 main 自己顶位）
                    ref_values = [main_agent_elo] + list(rule_elos)
                    ref_min = float(np.min(ref_values))
                    ref_max = float(np.max(ref_values))
                    ref_spread = ref_max - ref_min
                    curr_rate = 0.0 if ref_spread == 0 else (main_agent_elo - ref_min) / ref_spread

                    # Elo分数没有稳定不许记录
                    if total_steps_of_maneuver >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rate_normed %", curr_rate * 100, step_axis)

                    # 池容量占用（规则 + NN 总数）
                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    pool_total = len([k for k in elite_elo_ratings.keys() if not k.startswith("__")])
                    logger.add("Elo/History_Pool_Size", hist_count, step_axis)
                    logger.add("Elo/Pool_Total_Size", pool_total, step_axis)

                # 精英池维护（容量只对 NN 生效，规则对手永不踢出；不再定期全量重刷）
                if hist_agent_as_opponent and os.path.exists(os.path.join(log_dir, f"{actor_team_key}.pt")):
                    new_entry_elo = main_agent_elo
                    nn_keys = [k for k in elite_elo_ratings.keys()
                               if not k.startswith("__") and not k.startswith("Rule")]
                    if len(nn_keys) >= MAX_HISTORY_SIZE:
                        # 满员：在 NN 内部按 Elo 踢出最弱；规则永不被踢出
                        weakest_key = min(nn_keys, key=lambda k: elite_elo_ratings[k])
                        weakest_elo = elite_elo_ratings[weakest_key]
                        if new_entry_elo > weakest_elo:
                            del elite_elo_ratings[weakest_key]
                            elite_elo_ratings[actor_team_key] = new_entry_elo
                            print(f"[Pool Cleanup] Kicked {weakest_key} (Elo: {weakest_elo:.0f}), "
                                  f"accepted {actor_team_key} (Elo: {new_entry_elo:.0f}).")
                        else:
                            print(f"[Pool Skip] {actor_team_key} (Elo: {new_entry_elo:.0f}) not stronger "
                                  f"than weakest NN {weakest_key} (Elo: {weakest_elo:.0f}), skipped.")
                    else:
                        elite_elo_ratings[actor_team_key] = new_entry_elo
                        print(f"Accepted {actor_team_key} into Elite Pool.")

                    # 最新个体的绝对分与相对池均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, step_axis)
                    # Rule 对手的绝对分与主智能体对其的差值
                    rule_keys_ = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys_):
                        rule_elo = float(valid_elos[rk])
                        logger.add(f"Elo_Raw/{rk}", rule_elo, step_axis)
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, step_axis)

                episodes_window = []

                # 推进测试窗口阈值
                next_test_dp = decision_points_total + test_interval_dp

        print(f"\n--- Target steps reached: {total_steps_of_maneuver} / {current_max_steps_of_maneuver} ---")
        print("All simulation workers are now idling safely.")
        if not pause_on_finish:
            print("pause_on_finish=0, exiting training.")
            break
        try:
            inp = input(f"Enter new max_steps_of_maneuver (current {total_steps_of_maneuver}), or press Enter to exit: ")
        except (EOFError, OSError):   # 无 stdin（后台/管道运行）时自动退出而非挂起
            print("No stdin available. Exiting training.")
            break
        if not inp.strip():
            print("No input provided. Exiting training.")
            break # 退出外层 while True 循环
        try:
            new_max = int(inp)
            if new_max > total_steps_of_maneuver:
                current_max_steps_of_maneuver = new_max
                print(f"Continuing training until {current_max_steps_of_maneuver} steps.")
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
        except Exception:
            pass
            
    for p in workers:
        p.join(timeout=5) # 【修改】给子进程 5 秒优雅退出的时间
        if p.is_alive():
            p.terminate() # 如果没死，强制结束

    # 【修改】调整顺序：先尝试关闭测试池
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

    # --debug：小配置冒烟（2 workers、约 120 决策步），验证 worker spawn / 流式转移 /
    # 末端门控 / SAC 更新 / 测试窗口 / checkpoint 全链路；不带参数走正式配置
    debug = '--debug' in sys.argv
    if debug:
        run_kwargs = dict(
            num_workers=2,
            mission_name='HanSAC_homo_JPO_debug',
            lr=1e-4,
            max_steps_of_maneuver=120 * action_cycle_multiplier,
            gamma=0.9,
            tau=0.05,
            batch_size=32,
            buffer_size=int(1e5),
            min_terminal_samples=4,      # 调试：4 条末端帧即开门控，几局内就能看到首次更新
            update_batch_points=16,
            points_per_update=1,
            action_cycle_multiplier=action_cycle_multiplier,
            dt_move=0.07,
            max_episode_duration=15 * 60,
            R_cage=62.00e3,
            dt_maneuver=dt_maneuver,
            init_elo_ratings={'Rule_0': 1200, 'Rule_1': 1200, 'Rule_2': 1200, 'Rule_3': 1200},
            self_play_type='PFSP_balanced',
            hist_agent_as_opponent=1,
            WARM_UP_STEPS=0,
            MAX_HISTORY_SIZE=50,
            compete_old_rate=0.2,
            K_FACTOR=16,
            randomized_birth=0,          # 调试：固定出生状态，便于复现问题
            test_interval_dp=3333,       # 测试窗口触发间隔：累计 2000 决策点触发一次完整测试+日志+保存
            num_runs=1,
            device=torch.device("cpu"),  # 调试：强制 CPU，避免 CUDA 初始化干扰小规模验证
            vertices=None,
            resume_dir=None,
            pause_on_finish=0,           # 调试：跑完直接退出，不进续训提示符
            fire_buffer_size=1000,
            fire_batch_size=128,
            kI=0.8, kD=0.8,
        )
    else:
        run_kwargs = dict(
            num_workers=15,
            mission_name='HanSAC_homo_JPO',
            lr=1e-4,
            max_steps_of_maneuver=MAX_DEC_STEPS * action_cycle_multiplier,
            gamma=0.9,
            tau=0.05,
            batch_size=128,
            buffer_size=int(1e5),
            min_terminal_samples=32,
            update_batch_points=16,
            points_per_update=1,
            action_cycle_multiplier=action_cycle_multiplier,
            dt_move=0.07,
            max_episode_duration=15 * 60,
            R_cage=62.00e3,
            dt_maneuver=dt_maneuver,
            init_elo_ratings={'Rule_0': 1200, 'Rule_1': 1200, 'Rule_2': 1200, 'Rule_3': 1200},
            self_play_type='PFSP_balanced',
            hist_agent_as_opponent=1,
            WARM_UP_STEPS=0e3,
            MAX_HISTORY_SIZE=50,
            compete_old_rate=0.2,
            K_FACTOR=16,
            randomized_birth=1,
            test_interval_dp=3333,       # 测试窗口触发间隔：累计 2000 决策点触发一次完整测试+日志+保存
            num_runs=3,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            vertices=None,
            resume_dir=None,
            fire_buffer_size=1000,
            fire_batch_size=128,
            kI=0.8, kD=0.8,
        )

    run_HanSAC_simulation(**run_kwargs)
