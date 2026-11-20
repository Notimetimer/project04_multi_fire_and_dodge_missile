"""CombatCOMA_parall_homo：COMA（集中式 critic + 反事实优势）的 2V2 同构训练入口。

来源：CombatHanSAC_parall_homo.py（复制改造，原件不动；对比算法规范 §"共享优先，原件不动"）。
采样/对手池/Elo/测试/日志等基础设施与 HanSAC 版逐位一致，只替换学习节拍与更新：
- on-policy：worker 仍按决策点流式回传 per-uid 观测/奖励/存活；
  master 把同一 worker 同一决策点的 N 条记录重组为一条"联合决策"，
  回合结束把该回合的决策链 push 给 COMAHybrid（TD(λ) 只在链内回传）；
- 攒够 rollout_min_decisions 个决策点后调用一次 update()（update_epochs × 1 轮），
  随后清空 rollout（不做跨 rollout 复用），SAC 的温度/双 Q/τ 软更新全部移除；
- 目标 critic 按梯度步计数硬拷贝（target_update_interval）；
- 开火仍由独立 fire_head 的 BCE 监督（D_f），策略损失里不含开火梯度。

相对 HanSAC 的声明偏差见 run_config.json 的 declared_deviations。
"""

import typing
import os
import sys
import csv
import subprocess
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
from Algorithms.COMAWrapper import COMAWrapper
from Algorithms.COMAHybrid import COMAHybrid
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
    """SB3 式采样 worker：只执行环境步进，红方学生由 master 集中推理。

    协议（对齐 stable-baselines3 SubprocVecEnv 的同步采样）：
    - RESET：接收 (opp_info, settings)，加载蓝方历史权重并重置环境，回传红方初始观测（READY）；
    - STEP：接收 master 批量推理出的红方动作，执行 action_cycle_multiplier 帧仿真，
      回传 next_obs / reward / done / active，回合结束时附带 episode_end 统计；
    - 蓝方（规则或历史 nn）仍在 worker 本地推理，避免逐帧下发蓝方动作。
    """
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
    # 蓝方历史 nn 对手按蓝机 id 存放；红方学生不再在 worker 侧持有
    actor_dict = {}
    episode = None

    def _read_obs_all():
        agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
        for uid in env.UAV_ids:
            agent_input, mask_input, rule_input = env.get_obs(uid, pomdp=pomdp)
            agent_inputs_[uid] = agent_input
            mask_inputs_[uid] = mask_input
            rule_inputs_[uid] = rule_input
        return agent_inputs_, mask_inputs_, rule_inputs_

    def _obs_triple(oid, agent_inputs_, mask_inputs_, rule_inputs_):
        return {'agent_input': agent_inputs_[oid],
                'mask_input': mask_inputs_[oid],
                'rule_input': take_some_of_rule_inputs(rule_inputs_[oid])}

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RESET':
                opp_info, settings = packet
                opp_name, opp_type, opp_data = opp_info
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])

                # 蓝方历史 nn 权重仍随 RESET 下发（红方学生权重不再需要）
                blue_nn_weights = opp_data if (opp_type == 'nn' and opp_data) else {}
                for uid, sd in blue_nn_weights.items():
                    if uid not in actor_dict:
                        actor_dict[uid] = HanWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker)
                    actor_dict[uid].load_state_dict(sd, strict=False)   # COMA 快照只含 actor 侧
                    actor_dict[uid].eval()

                rb, bb = settings['red_birth'], settings['blue_birth']
                env.reset(red_birth_state=rb, blue_birth_state=bb,
                          red_init_num=red_init_num, blue_init_num=blue_init_num,
                          red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo,
                          seed=(settings.get('red_shuffle_seed'), settings.get('blue_shuffle_seed')))

                red_ids = sorted([id for id in env.UAV_ids if env.find_uav_with_id(id).side == 'r'])
                episode = {
                    'opp_name': opp_name,
                    'opp_data': opp_data,
                    'settings': settings,
                    'blue_nn_weights': blue_nn_weights,
                    'red_ids': red_ids,
                    'maneuver_cmds_all': {},
                    'target_index_all': {},
                    'done': False,
                    'ep_return': 0.0,
                    'return_by_id': {id: 0.0 for id in red_ids},
                    'decisions': 0,
                    'frames': 0,
                    'max_decisions': max(1, int(round(args.max_episode_len / dt_maneuver)
                                             // settings['action_cycle_multiplier'])),
                }
                ai_, mi_, ri_ = _read_obs_all()
                pipe.send({'type': 'ready', 'rank': rank,
                           'obs': {id: _obs_triple(id, ai_, mi_, ri_) for id in red_ids}})
            elif cmd == 'STEP':
                ep = episode
                actions = packet['actions']
                action_cycle_multiplier = ep['settings']['action_cycle_multiplier']
                last_reward = {id: 0.0 for id in ep['red_ids']}
                done = ep['done']
                for frame in range(action_cycle_multiplier):
                    if not env.running or done:
                        break

                    agent_inputs_, mask_inputs_, rule_inputs_ = _read_obs_all()

                    # 决策帧：应用 master 下发的红方动作；非决策帧沿用上一决策点的机动指令
                    if frame == 0:
                        ep['decisions'] += 1
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            is_student = UAV.side == 'r'

                            if is_student:
                                a = actions[id]
                                ep['maneuver_cmds_all'][id] = [int(a['vert']), int(a['hor'])]
                                ep['target_index_all'][id] = int(a['enm_idx'])
                                fire_flag = bool(int(a['fire']))
                                fire_target = int(a['fire_target'])
                            elif id in ep['blue_nn_weights']:
                                # 蓝方历史 nn 对手仍在 worker 本地推理
                                act_model = actor_dict[id]
                                rule_input4actor = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                                with torch.no_grad():
                                    actor_out = act_model.get_action(
                                        agent_inputs_[id], mask_inputs_[id], rule_input4actor)
                                ep['maneuver_cmds_all'][id] = [int(actor_out['vert_idx'].item()),
                                                               int(actor_out['hor_idx'].item())]
                                ep['target_index_all'][id] = int(actor_out['enm_idx'].item())
                                fire_flag = bool(int(actor_out['fire'].item()))
                                fire_target = int(actor_out['fire_target'].item())
                            else:
                                # 蓝方规则对手
                                fly_action, fire_action, _ = basic_rules(
                                    rule_inputs_[id], mask_inputs_[id], rules_num=ep['opp_data'])
                                ep['maneuver_cmds_all'][id] = fly_action['do']
                                ep['target_index_all'][id] = int(fly_action['at'])
                                fire_target = int(fire_action['at'])
                                fire_flag = bool(fire_action['do'])

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
                                id, ep['target_index_all'].get(id, 0),
                                ep['maneuver_cmds_all'].get(id, [1, 0]), rule_inputs_[id])
                        except ValueError:
                            flight_cmds[id] = [0.0, 0.0, 0.0]
                    env.step(flight_cmds.values())

                    # 奖励结算
                    ep_return = 0.0
                    for id in env.UAV_ids:
                        done, reward_dict = env.combat_terminate_and_reward(
                            id=id, action_cycle_multiplier=action_cycle_multiplier)
                        if id in ep['return_by_id']:
                            last_reward[id] = sum(reward_dict.values())
                            ep_return += last_reward[id]
                            ep['return_by_id'][id] += last_reward[id]
                    ep['ep_return'] = ep_return
                    ep['frames'] += 1

                ep['done'] = bool(done)
                terminated = bool(done) or (not env.running)
                truncated = (not terminated) and ep['decisions'] >= ep['max_decisions']
                episode_done = bool(terminated or truncated)

                # 终止帧的 next_obs 不参与 bootstrap（与流式版一致，直接置空由 master 回填）
                next_obs = None
                if not terminated:
                    next_obs = {}
                    for id in ep['red_ids']:
                        ai_, mi_, ri_ = env.get_obs(id, pomdp=pomdp)
                        next_obs[id] = {'agent_input': ai_, 'mask_input': mi_,
                                        'rule_input': take_some_of_rule_inputs(ri_)}

                msg = {
                    'type': 'step_result', 'rank': rank,
                    'rewards': dict(last_reward),
                    'active': {id: not env.find_uav_with_id(id).dead for id in ep['red_ids']},
                    'next_obs': next_obs,
                    'done': episode_done,
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                }
                if episode_done:
                    outcome = env.r_outcome if env.r_outcome is not None else 0
                    # D_f 收全部飞机（红方 + 蓝方规则/历史 NN）的开火记录，与 PPO 口径一致
                    # [A方案] 自 2026-11-18 起只收红方：蓝方开火记录不进 D_f（上面口径已弃用）。
                    fire_hit_records = []
                    for id in env.UAV_ids:
                        if env.find_uav_with_id(id).side != 'r':
                            continue
                        fire_hit_records += env.find_uav_with_id(id).fire_hit_records
                    msg['episode_end'] = {
                        'metrics': {
                            'steps': ep['frames'],
                            'return': ep['ep_return'],
                            'return_by_id': dict(ep['return_by_id']),
                            'win': outcome == 1,
                            'lose': outcome == -1,
                            'draw': outcome == 0,
                            'BVR_perish_together': all(uav.dead for uav in env.UAVs)
                                                   and env.t <= args.max_episode_len - 10,
                        },
                        'opp_name': ep['opp_name'],
                        'fire_hit_records': fire_hit_records,
                    }
                    episode = None
                pipe.send(msg)
            elif cmd == 'EXIT':
                break
        except Exception:
            pipe.send({'error': traceback.format_exc()})
            break


def test_worker_han(model_state_dict, rules_num, env_args, device_name='cpu', num_runs=3,
                    vertices=None, dt_move=0.07, dt_maneuver=0.2, action_cycle_multiplier=10,
                    red_init_num=2, blue_init_num=2, red_init_ammo=4, blue_init_ammo=4,
                    temperature=None, enm_chosen_visable=0, use_tacview=0):
    """周期性规则测试：与 PPO 测试口径对齐 —— 机动用 temperature=0.3 的 Categorical 采样，
    开火用 Bernoulli 概率采样（不用 argmax，也不设开火阈值）。"""
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
    actor.load_state_dict(model_state_dict, strict=False)   # COMA 快照只含 actor 侧
    actor.eval()

    wins = loses = draws = perish_togethers = 0
    kill_count = die_count = 0
    # 多场共用一条 acmi 时间轴：每场结束后累加 env.t，避免轨迹在 t=0 处重叠
    t_bias = 0
    # 红方开火概率曲线采集：id -> [(全局时间 s, 开火概率)]；每场的时间区间另存一份供画图分隔
    fire_prob_series = {}
    episode_spans = []

    for _ in range(num_runs):
        rb, bb = create_initial_state_worker(1)
        env.reset(red_birth_state=rb, blue_birth_state=bb,
                  red_init_num=red_init_num, blue_init_num=blue_init_num,
                  red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo)
        done = False
        maneuver_cmds_all, target_index_all = {}, {}
        count = 0
        # 本场红机 id 固定为 1..red_init_num；先建好曲线槽位，死掉的飞机后续不再追加点
        for id in env.UAV_ids:
            if env.find_uav_with_id(id).side == 'r':
                fire_prob_series.setdefault(int(id), [])
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
                            # 测试部署口径：temperature=0.3 的 Categorical 采样（非 argmax），
                            # 开火由 get_action 内部按 Bernoulli 采样（不设阈值）
                            actor_out = actor.get_action(
                                agent_inputs_[id], mask_inputs_[id], rule_input4actor,
                                temperature=(0.3 if temperature is None else float(temperature)),
                                deterministic=False)
                        maneuver_cmds_all[id] = [int(actor_out['vert_idx'].item()),
                                                 int(actor_out['hor_idx'].item())]
                        target_index_all[id] = int(actor_out['enm_idx'].item())
                        fire_flag = bool(int(actor_out['fire'].item()))
                        fire_target = int(actor_out['fire_target'].item())
                        # 记录开火概率（= 所选 TDM 敌机的 sigmoid 命中概率，已做禁火掩码回退）
                        fire_prob_series[int(id)].append(
                            (t_bias + float(env.t), float(actor_out['fire_prob_primary'])))
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
        episode_spans.append((float(t_bias), float(t_bias + env.t)))
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
            'kill_count': kill_count, 'die_count': die_count,
            # 曲线数据：每个红机的 (时间 s, 开火概率) 序列 + 每场的全局时间区间
            'fire_prob_series': fire_prob_series, 'episode_spans': episode_spans}


def _actor_only_state_dict(ac):
    """只取 actor 侧权重：测试 worker 与自博弈对手都用 HanWrapper 加载。"""
    return {k: v for k, v in ac.state_dict().items() if not k.startswith('critic.')}


def run_COMA_simulation(
    num_workers=10,
    mission_name='COMA无名',
    lr=3e-4,                    # COMA 论文 lr=1e-4；on-policy 更新次数多，取 3e-4 与 PPO 口径接近
    max_steps_of_maneuver=4 * 165e4,
    gamma=0.9,                  # 论文 γ=0.9
    lam=0.8,                    # TD(λ) 的 λ
    entropy_coef=0.001,         # 固定熵系数（COMA 原版无熵正则，见 declared_deviations）
    update_epochs=3,            # 每个 rollout 复用的 epoch 数
    batch_size=512,             # minibatch 大小（critic/actor 共用）
    rollout_min_decisions=1024, # 攒够多少"联合决策"才触发一次 update
    target_update_interval=200, # 目标 critic 硬拷贝间隔（按梯度步）
    max_grad_norm=10.0,         # critic/actor 各自梯度裁剪阈值
    fire_batch_size=128,        # 开火 BCE 批大小
    fire_min_samples=100,       # D_f 少于该样本时不更新 fire_head
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
    test_interval_dp=3333,       # 测试窗口触发间隔：累计该决策点触发一次完整测试+日志+保存
    num_runs=3,
    device=torch.device("cpu"),
    vertices=None,
    resume_dir=None,
    pause_on_finish=1,          # 达标后是否进入续训提示符（后台/调试运行设 0 直接退出）
    fire_buffer_size=1000,      # 论文 |D_f|=1e3
    seed=42,
    **_unused,
):
    """COMA 主训练函数：on-policy 反事实策略梯度 + 集中式 critic + 开火 BCE 监督 + 自博弈。"""
    from _context import load_obs_dims, load_init_states
    from multiprocessing.connection import wait as conn_wait

    # 初始数量/挂载：显式传参优先，否则统一从 init_states.json 读取
    init_cfg = load_init_states()
    red_init_num = init_cfg['red_init_num'] if red_init_num is None else red_init_num
    blue_init_num = init_cfg['blue_init_num'] if blue_init_num is None else blue_init_num
    red_init_ammo = init_cfg['red_init_ammo'] if red_init_ammo is None else red_init_ammo
    blue_init_ammo = init_cfg['blue_init_ammo'] if blue_init_ammo is None else blue_init_ammo

    # 1. 设置随机数种子 (Master)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser("UAV swarm confrontation (COMA)")
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

    # actor 复用 HanWrapper 的异构注意力骨干，critic 换成集中式单 Q；
    # learner 为 on-policy COMA（TD(λ) 目标 + 反事实基线 + 开火 BCE）
    ac = COMAWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
    agent = COMAHybrid(
        ac, lr=lr, gamma=gamma, lam=lam, entropy_coef=entropy_coef,
        update_epochs=update_epochs, minibatch_size=batch_size,
        rollout_min_decisions=rollout_min_decisions,
        target_update_interval=target_update_interval, max_grad_norm=max_grad_norm,
        fire_buffer_size=fire_buffer_size, fire_batch_size=fire_batch_size,
        fire_min_samples=fire_min_samples, device=device)

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

    # 断点续训：加载最近一次完整智能体状态（含 critic / target / 三个 optimizer）
    batch_idx = 0
    if resume_dir is not None:
        agent_path = os.path.join(log_dir, "coma_agent.pt")
        if os.path.exists(agent_path):
            agent.load_state_dict(torch.load(agent_path, map_location=device))
            actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
            if actor_files:
                batch_idx = max(int(re.search(r'actor_rein(\d+)', os.path.basename(f)).group(1))
                                for f in actor_files if re.search(r'actor_rein(\d+)', os.path.basename(f)))
            print(f"Loaded agent state from {agent_path}, batch_idx={batch_idx}")

    logger = TensorBoardLogger(
        log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 训练配置落盘（对比算法规范：run_config.json 需能反查代码版本/超参/种子/
    # declared_deviations；续训目录同样刷新一份，便于按最新参数复现）
    run_config = {
        'algo': 'COMA',
        'env': 'homo_2v2',
        'mission_name': mission_name,
        'seed': seed,
        'code': {
            'entry': 'TrainAndTests/Combats/多机空战同构训练COMA.py',
            'runner': 'TrainAndTests/Combats/CombatCOMA_parall_homo.py',
            'actor_shell': 'Algorithms/COMAWrapper.py',
            'learner': 'Algorithms/COMAHybrid.py',
            'base_actor': 'Algorithms/HanWrapper.py',
        },
        'hyperparams': {
            'lr': lr, 'gamma': gamma, 'lam': lam, 'entropy_coef': entropy_coef,
            'update_epochs': update_epochs, 'minibatch_size': batch_size,
            'rollout_min_decisions': rollout_min_decisions,
            'target_update_interval': target_update_interval, 'max_grad_norm': max_grad_norm,
            'fire_buffer_size': fire_buffer_size, 'fire_batch_size': fire_batch_size,
            'fire_min_samples': fire_min_samples,
            'num_workers': num_workers, 'num_runs': num_runs,
            'test_interval_dp': test_interval_dp, 'max_steps_of_maneuver': max_steps_of_maneuver,
            'action_cycle_multiplier': action_cycle_multiplier,
            'dt_move': dt_move, 'dt_maneuver': dt_maneuver,
            'max_episode_duration': max_episode_duration, 'R_cage': R_cage,
            'red_init_num': red_init_num, 'blue_init_num': blue_init_num,
            'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo,
            'self_play_type': self_play_type, 'hist_agent_as_opponent': hist_agent_as_opponent,
            'WARM_UP_STEPS': WARM_UP_STEPS, 'MAX_HISTORY_SIZE': MAX_HISTORY_SIZE,
            'compete_old_rate': compete_old_rate, 'K_FACTOR': K_FACTOR,
            'randomized_birth': randomized_birth,
        },
        'declared_deviations': [
            'critic 的状态是各 agent 本地观测集合的编码（EntityBackbone），非上帝视角真值状态',
            '用队伍级交叉注意力（自身 key 屏蔽）取代 COMA/PyMARL 的全连接 MLP critic',
            '动作空间为 2A（机动×开火）；开火仍由独立 fire_head 的 BCE 监督，反事实基线里开火概率 detach',
            '策略损失加固定熵系数（默认 0.001），COMA 原版无熵正则',
            '无 GRU：actor 与 critic 均为单步前馈（状态信息由集中 critic 补足）',
            '时限截断片段末端用"末步联合动作 + 当前策略"的软价值 bootstrap（近似）',
            'critic 目标使用独立 target 网络（按梯度步硬拷贝），COMA/PyMARL 原版直接用在线 critic 算 TD 目标',
            'rollout 以"回合"为链单位攒批，在飞回合的决策要等本回合结束后才进入更新（存在少量滞后）',
            '评测沿用主方法（HanSAC）的 checkpoint 选择与测试协议，测试规模 2v2/4v4/6v6',
        ],
    }
    try:
        run_config['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root, text=True).strip()
        run_config['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=project_root, text=True).strip())
    except Exception as e:
        run_config['git_commit'] = None
        run_config['git_error'] = str(e)
    with open(os.path.join(log_dir, 'run_config.json'), 'w', encoding='utf-8') as _f:
        json.dump(run_config, _f, ensure_ascii=False, indent=2)

    # 存储在线训练前的网络参数（同构单份，只含 actor 侧 key，便于 HanWrapper 直接加载）
    # actor_rein0 只在首次创建时落盘；续训时若已存在则保留原始初始策略，避免被 latest 权重覆盖
    if not os.path.exists(os.path.join(log_dir, "actor_rein0.pt")):
        torch.save(_actor_only_state_dict(ac), os.path.join(log_dir, "actor_rein0.pt"))
    torch.save(agent.state_dict(), os.path.join(log_dir, "coma_agent.pt"))

    # 开火记录滚动缓冲：跨 batch 累积（FIFO 逐条覆盖）；D_f 由 COMAHybrid 内部维护，
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
    # 自博弈关闭（hist_agent_as_opponent=0）：对手池只保留预设规则对手；
    # 即使续训目录里残留历史 NN 快照，也一律剔除，保证池内不出现自博弈对手
    if not hist_agent_as_opponent:
        elite_elo_ratings = {k: v for k, v in elite_elo_ratings.items()
                             if k.startswith("Rule")}

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
        # ============ SB3 式同步步级主循环 ============
        # master 集中推理红方动作 → 向所有活跃 worker 发 STEP → 收齐全部回包（步级栅栏）
        # → 按 worker 重组联合决策；回合结束 push_chain，攒够 rollout_min_decisions 后 update。
        decision_points_total = int(total_steps_of_maneuver / action_cycle_multiplier)
        next_test_dp = decision_points_total + test_interval_dp
        # 吞吐观测窗口：dp/s（采样）、updates/s（更新密度）、ms/update 与 master 更新占用率，
        # 用于区分“采样受限”与“更新受限”，不参与训练逻辑
        perf_win_t = time.perf_counter()
        perf_win_dp = 0
        perf_win_updates = 0
        perf_win_update_time = 0.0
        perf_win_flushes = 0
        ep_chains = {r: [] for r in range(num_workers)}   # rank -> 本回合已采的联合决策链
        lane_obs = {}              # rank -> uid -> 当前观测三元组（master 侧持有，供集中推理）
        lane_uids = {}             # rank -> 本回合红机 id 列表（固定 N，含已死机，保证链内可堆叠）
        lane_alive = {}            # rank -> uid -> 决策时刻是否存活
        active_ranks = set()       # 当前有在跑回合、可继续 step 的 worker
        episodes_window = []      # 本测试窗口内的 (opp_name, metrics)
        last_update_metrics = {}
        stopping = False
        idle_ranks = set()

        def _stack_uid_obs(obs_by_uid, uids):
            """把逐 uid 的观测三元组按 uids 顺序堆成 (N,...)，供 COMA 联合决策使用。"""
            return {
                'agent_input': {
                    k: np.stack([np.asarray(obs_by_uid[u]['agent_input'][k]) for u in uids])
                    for k in obs_by_uid[uids[0]]['agent_input']
                },
                'mask_input': {
                    k: np.stack([np.asarray(obs_by_uid[u]['mask_input'][k]) for u in uids])
                    for k in obs_by_uid[uids[0]]['mask_input']
                },
                'rule_input': stack_field([obs_by_uid[u]['rule_input'] for u in uids]),
            }

        def _register_ready(ready_obs):
            """READY 后登记本回合的 N 与存活状态（所有红机一律按满编处理）。"""
            for r_, obs_r in ready_obs.items():
                lane_uids[r_] = sorted(obs_r.keys())
                lane_alive[r_] = {u: True for u in lane_uids[r_]}

        def make_episode_setup(rank):
            """PFSP 选对手 + 出生状态，构成一局的 RESET 参数（红方权重由 master 集中持有，不再下发）。"""
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
            return opp_info, settings

        def _check_msg(msg, r):
            if isinstance(msg, dict) and 'error' in msg:
                for p in workers:
                    p.terminate()
                raise RuntimeError(f"Worker {r} crashed with error:\n{msg['error']}")

        def _collect_ready(ranks):
            """收齐指定 worker 的 READY（初始观测），用于 RESET 后的屏障对齐。"""
            pending_r = set(ranks)
            obs = {}
            while pending_r:
                for conn in conn_wait([pipes[r] for r in pending_r]):
                    r = pipes.index(conn)
                    try:
                        msg = conn.recv()
                    except EOFError:
                        for p in workers:
                            p.terminate()
                        raise RuntimeError(f"Worker {r} crashed silently.")
                    _check_msg(msg, r)
                    obs[r] = msg['obs']
                    pending_r.discard(r)
            return obs

        # ---- 初始化：向全部 worker 发 RESET，收齐首帧观测 ----
        for r in range(num_workers):
            opp_info, settings = make_episode_setup(r)
            pipes[r].send(('RESET', (opp_info, settings)))
        ready_obs = _collect_ready(range(num_workers))
        lane_obs.update(ready_obs)
        _register_ready(ready_obs)
        active_ranks.update(range(num_workers))

        while True:
            # 目标达成且所有 worker 已空闲（回合自然结束、未再派新局）→ 退出到提示符
            if stopping and len(idle_ranks) == num_workers:
                break
            if not active_ranks:
                break

            # ---- 1) master 集中推理：跨 worker 批量前向（红方学生共享同一份权重）----
            act_batch = {k: [] for k in ('fixed', 'enm', 'ally')}
            mask_batch = {k: [] for k in ('enm', 'ally')}
            rule_batch = []
            order = []   # [(rank, uid)]，与 batch 行索引一一对应
            for r in sorted(active_ranks):
                for uid in sorted(lane_obs[r].keys()):
                    obs = lane_obs[r][uid]
                    for k in act_batch:
                        act_batch[k].append(np.asarray(obs['agent_input'][k]))
                    for k in mask_batch:
                        mask_batch[k].append(np.asarray(obs['mask_input'][k]))
                    rule_batch.append(obs['rule_input'])
                    order.append((r, uid))

            action_packets = {r: {} for r in active_ranks}
            if order:
                actor_out = ac.get_action(
                    {k: np.stack(v) for k, v in act_batch.items()},
                    {k: np.stack(v) for k, v in mask_batch.items()},
                    stack_field(rule_batch))
                for i, (r, uid) in enumerate(order):
                    action_packets[r][uid] = {
                        'flat_idx': int(actor_out['flat_idx'][i]),
                        'enm_idx': int(actor_out['enm_idx'][i]),
                        'vert': int(actor_out['vert_idx'][i]),
                        'hor': int(actor_out['hor_idx'][i]),
                        'fire': int(actor_out['fire'][i]),
                        'fire_target': int(actor_out['fire_target'][i]),
                    }

            # ---- 2) 同步步级栅栏：所有活跃 worker 各推进一个决策点 ----
            stepped_ranks = sorted(active_ranks)
            for r in stepped_ranks:
                pipes[r].send(('STEP', {'actions': action_packets[r]}))

            replies = {}
            pending_r = set(stepped_ranks)
            while pending_r:
                for conn in conn_wait([pipes[r] for r in pending_r]):
                    r = pipes.index(conn)
                    try:
                        msg = conn.recv()
                    except EOFError:
                        for p in workers:
                            p.terminate()
                        raise RuntimeError(f"Worker {r} crashed silently.")
                    _check_msg(msg, r)
                    replies[r] = msg
                    pending_r.discard(r)

            # ---- 3) 收包 → master 把同一 worker 同一决策点的 N 条记录重组为一条联合决策 ----
            # num_timesteps 按 n_envs（实际被 step 的 worker 数）累加，与 SB3 口径一致
            if not stopping:
                perf_win_dp += len(replies)
                decision_points_total += len(replies)
                total_steps_of_maneuver = decision_points_total * action_cycle_multiplier

            reset_ranks = []
            for r in stepped_ranks:
                msg = replies[r]
                prev_obs = lane_obs[r]
                sent_actions = action_packets[r]
                next_obs = msg.get('next_obs')
                uids = lane_uids[r]

                if not stopping:
                    # 决策时刻存活：用上一步结算后的存活状态（本位死亡的 agent 仍计入本步奖励）
                    alive_pre = lane_alive[r]
                    decision = {
                        'obs': _stack_uid_obs(prev_obs, uids),
                        'actions': np.array([
                            int(sent_actions[u]['flat_idx']) * 2 + int(sent_actions[u]['fire'])
                            if u in sent_actions else 0 for u in uids], dtype=np.int64),
                        'rewards': np.array([float(msg['rewards'].get(u, 0.0)) for u in uids],
                                            dtype=np.float32),
                        'active': np.array([1.0 if alive_pre.get(u, False) else 0.0 for u in uids],
                                           dtype=np.float32),
                        'terminated': int(msg['terminated']),
                        'truncated': int(msg['truncated']),
                        # next_obs 只在链末端用于截断 bootstrap；终止帧不 bootstrap
                        'next_obs': None,
                    }
                    if msg['done'] and not msg['terminated'] and next_obs:
                        decision['next_obs'] = _stack_uid_obs(next_obs, uids)
                    ep_chains[r].append(decision)
                    # 结算后更新存活状态（死亡单调，决策时刻的存活已记入 active）
                    for u in uids:
                        lane_alive[r][u] = bool(msg['active'].get(u, False))

                if not msg['done']:
                    lane_obs[r] = next_obs
                    continue

                # ---- 回合结束：决策链整体交给 on-policy buffer ----
                if not stopping and ep_chains[r]:
                    agent.push_chain(ep_chains[r])
                ep_chains[r] = []
                ep = msg['episode_end']
                metrics = ep['metrics']
                opp_name = ep['opp_name']
                episodes_window.append((opp_name, metrics))
                fire_records = [x for x in ep.get('fire_hit_records', [])
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
                    active_ranks.discard(r)
                else:
                    opp_info, settings = make_episode_setup(r)
                    pipes[r].send(('RESET', (opp_info, settings)))
                    reset_ranks.append(r)

            # 回合结束的 worker 重置后重入屏障（下一轮与其余 worker 同时 step）
            if reset_ranks:
                ready_obs = _collect_ready(reset_ranks)
                lane_obs.update(ready_obs)
                _register_ready(ready_obs)
                active_ranks.update(reset_ranks)

            # ---- on-policy 触发：攒够 rollout_min_decisions 后消费整段 rollout 并清空 ----
            if len(agent.buffer) >= agent.rollout_min_decisions:
                _t_upd = time.perf_counter()
                last_update_metrics = agent.update().metrics
                perf_win_update_time += time.perf_counter() - _t_upd
                perf_win_updates += 1
                perf_win_flushes += 1
                if perf_win_flushes % 20 == 0:
                    _el = max(1e-9, time.perf_counter() - perf_win_t)
                    print(f"  [Perf] {perf_win_dp / _el:.1f} dp/s | "
                          f"updates {perf_win_updates / _el:.1f}/s | "
                          f"{1000 * perf_win_update_time / max(1, perf_win_updates):.1f} ms/update | "
                          f"master update duty {100 * perf_win_update_time / _el:.0f}% | "
                          f"rollout {len(agent.buffer)}")
                    perf_win_t = time.perf_counter()
                    perf_win_dp = 0
                    perf_win_updates = 0
                    perf_win_update_time = 0.0

            # ---- 测试窗口：每累计 test_interval_dp（默认 2000）决策点做一轮规则测试 + 日志 + checkpoint ----
            if decision_points_total >= next_test_dp:
                batch_idx += 1
                step_axis = float(decision_points_total)

                # [测试] 并行测试当前学生队 vs 规则对手（论文测试设定：argmax+阈值）
                if num_runs > 0:
                    print(f"\n>>> [Per-Batch Test] Testing student team vs rules (batch {batch_idx})...")
                    # 测试进程用 HanWrapper 加载：只下发放 actor 侧权重
                    current_ac_weights = {k: v.cpu() for k, v in _actor_only_state_dict(ac).items()}
                    # 三档规模串行：原规模 + 4v4 + 6v6（除场景规模外，规则/局数/温度/dt 等测试协议完全不变）
                    # 同一档内 4 个规则对手并行；等本档全部返回后再进入下一档，不同规模严格串行
                    test_scales = [(int(red_init_num), int(blue_init_num))]
                    for _scale_n in (4, 6):
                        if (_scale_n, _scale_n) not in test_scales:
                            test_scales.append((_scale_n, _scale_n))
                    for scale_red, scale_blue in test_scales:
                        scale_tag = f"test_{scale_red}v{scale_blue}"
                        print(f"  [Test Scale] {scale_tag} ...")
                        test_tasks = []
                        for r_idx in TEST_RULE_IDS:
                            obj = test_pool.apply_async(
                                test_worker_han,
                                kwds={'model_state_dict': current_ac_weights, 'rules_num': r_idx,
                                      'env_args': args, 'device_name': 'cpu', 'num_runs': num_runs,
                                      'vertices': vertices, 'dt_move': dt_move, 'dt_maneuver': 0.5, # dt_maneuver, # 测试的时候需要精细，训练的时候可以粗略
                                      'action_cycle_multiplier': action_cycle_multiplier,
                                      # test_worker_han 内部按 red/blue 机数重建 HanWrapper（敌机槽位=蓝方机数），无 alive 参数
                                      'red_init_num': scale_red, 'blue_init_num': scale_blue,
                                      'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo})
                            test_tasks.append(obj)
                        # 本档阻塞等待全部测试完成后再进入下一档（串行）
                        test_results = [t.get() for t in test_tasks]
                        outcomes = {r['rules_num']: r['result'] for r in test_results}
                        outcomes_perish = {r['rules_num']: r['perish_togethers'] / num_runs
                                           for r in test_results}
                        for r_num, score in outcomes.items():
                            logger.add(f"{scale_tag}/agent_vs_rule{r_num}", score, step_axis)
                            print(f"  [Test Result] {scale_tag} Rule_{r_num}: {score}")
                        avg_score = float(np.mean(list(outcomes.values())))
                        logger.add(f"{scale_tag}/avg_score", avg_score, step_axis)
                        logger.add(f"{scale_tag}/BVR perish together",
                                   float(np.mean(list(outcomes_perish.values()))), step_axis)
                        total_kills = sum(r['kill_count'] for r in test_results)
                        total_dies = sum(r['die_count'] for r in test_results)
                        logger.add(f"{scale_tag}/log_10_KD",
                                   float(np.log10(np.clip(total_kills / (total_dies + 1e-6), 1/20, 20))),
                                   step_axis)

                        # 分规模落盘：统一 test_vs_rules_{r}v{b}.csv（不再读写 test_vs_rules.csv）
                        test_csv_path = os.path.join(log_dir, f"test_vs_rules_{scale_red}v{scale_blue}.csv")
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
                logger.add("agent/rollout_decisions", agent.buffer.n_decisions, step_axis)

                if last_update_metrics:
                    m = last_update_metrics
                    logger.add("train/7 loss_q", m.get('loss_q', 0.0), step_axis)
                    logger.add("train/8 loss_pi", m.get('loss_pi', 0.0), step_axis)
                    logger.add("train/9 H_J", m.get('H_J', 0.0), step_axis)
                    logger.add("train/9 adv_abs", m.get('adv_abs', 0.0), step_axis)
                    logger.add("train/9 q_mean", m.get('q_mean', 0.0), step_axis)
                    logger.add("train/supervised_fire_loss", m.get('loss_fire', 0.0), step_axis)
                    logger.add("train_plus/fire_buffer_size", m.get('fire_buffer', 0.0), step_axis)
                    logger.add("train_plus/rollout_decisions", m.get('rollout_decisions', 0.0), step_axis)

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

                # 每个测试窗口都落盘：actor_rein{batch}.pt 供自博弈池加载（仅 actor 侧 key），
                # coma_agent.pt 供续训（含 critic / target / optimizer）
                actor_team_key = f"actor_rein{batch_idx}"
                torch.save(_actor_only_state_dict(ac), os.path.join(log_dir, f"{actor_team_key}.pt"))
                torch.save(agent.state_dict(), os.path.join(log_dir, "coma_agent.pt"))
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
    # 与《多机空战同构训练HANSAC.py》同口径的默认配置，SAC 专属超参换成 COMA 的
    # on-policy 节拍（rollout_min_decisions / update_epochs / TD(λ)）
    dt_decide = 2
    dt_maneuver = 0.2
    action_cycle_multiplier = int(round(dt_decide / dt_maneuver))
    MAX_DEC_STEPS = 1e6

    # --debug：小配置冒烟（2 workers、短回合便于尽快产生完整决策链），验证 spawn /
    # 联合决策组装 / push_chain / update / checkpoint 全链路；不带参数走正式配置
    debug = '--debug' in sys.argv
    if debug:
        run_kwargs = dict(
            num_workers=2,
            mission_name='COMA_homo_JPO_debug',
            lr=3e-4,
            max_steps_of_maneuver=200 * action_cycle_multiplier,
            gamma=0.9,
            lam=0.8,
            entropy_coef=0.001,
            update_epochs=2,
            batch_size=32,
            rollout_min_decisions=32,
            target_update_interval=50,
            max_grad_norm=10.0,
            action_cycle_multiplier=action_cycle_multiplier,
            dt_move=0.07,
            max_episode_duration=90,     # 调试：90s 截断，一回合约 45 个决策点
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
            test_interval_dp=40,         # 调试：40 个决策点就跑一次完整测试，覆盖测试链路
            num_runs=1,
            device=torch.device("cpu"),  # 调试：强制 CPU，避免 CUDA 初始化干扰小规模验证
            vertices=None,
            resume_dir=None,
            pause_on_finish=0,           # 调试：跑完直接退出，不进续训提示符
            fire_buffer_size=1000,
            fire_batch_size=128,
            fire_min_samples=100,
        )
    else:
        run_kwargs = dict(
            num_workers=15,
            mission_name='COMA_homo_JPO',
            lr=3e-4,
            max_steps_of_maneuver=MAX_DEC_STEPS * action_cycle_multiplier,
            gamma=0.9,
            lam=0.8,
            entropy_coef=0.001,
            update_epochs=3,
            batch_size=512,
            rollout_min_decisions=1024,
            target_update_interval=200,
            max_grad_norm=10.0,
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
            test_interval_dp=3333,
            num_runs=3,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            vertices=None,
            resume_dir=None,
            fire_buffer_size=1000,
            fire_batch_size=128,
            fire_min_samples=100,
        )

    run_COMA_simulation(**run_kwargs)
