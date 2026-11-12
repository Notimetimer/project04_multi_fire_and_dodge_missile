'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

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
from Algorithms.ActorWithATTN import ActorWrapper
from Algorithms.CriticWrapper import critic_wrapper
from Algorithms.MAPPOHybrid import MAPPOHybrid
from Math_calculates.sub_of_angles import sub_of_radian, sub_of_degree
from Visualize.tensorboard_visualize import TensorBoardLogger
from VsBaseline_MA import test_worker

    
dt_move = 0.04

# 测试对战初始弹药：统一从 init_states.json 读取，避免硬编码与训练配置脱节
_init_cfg = load_init_states()
red_init_ammo = _init_cfg['red_init_ammo']
blue_init_ammo = _init_cfg['blue_init_ammo']

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

from SelfPlayCalculation import calculate_expected_score, update_elo, get_opponent_probabilities

# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
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


def worker_process(rank, pipe, args, device_worker, dt_maneuver, seed, dt_move=0.07, pomdp=1, vertices=None):
    """采样 worker：红方学生共享一个 ActorWrapper（同构），蓝方为规则或历史 nn（对手）。
    经验只按学生 id 分组保存在内存，堆叠后通过 pipe 传回 master，不写磁盘。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.ActorWithATTN import ActorWrapper
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
                opp_name, opp_type, opp_data = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])
                enm_chosen_visable = settings.get('enm_chosen_visable', 0)  # 是否接收可见性标记
                ppo_with_bern = settings.get('ppo_with_bern', 1)  # 0 时 RL 的 log_prob/熵不含 fire 分支

                # 加载红方学生共享 actor 权重（学生 id 与环境红机 id 一致，全队共用一个实例）
                if student_actor is None:
                    student_actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                        device=device_worker, ppo_with_bern=ppo_with_bern,
                    )
                student_actor.load_state_dict(student_actor_weights)
                student_actor.eval()

                # 蓝方历史 nn 对手权重（master 加载时已把学生 id 重映射为蓝机 id 打包）
                blue_nn_weights = opp_data if (opp_type == 'nn' and opp_data) else {}
                for id, sd in blue_nn_weights.items():
                    if id not in actor_dict:
                        actor_dict[id] = ActorWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker, ppo_with_bern=ppo_with_bern,
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

                # 累加漏选目标情况用于平均
                ignore_num_count = 0
                ignore_enm_num_sum = 0
                
                # 预留位置
                red_enm_index_chosen = set()
                blue_enm_index_chosen = set()
                # # 队长 = 本队 id 最小者；它做决策时开启新一轮，清空本轮“已被选中”的敌机集合
                # red_leader_id = min(env.RUAV_ids) if env.RUAV_ids else None
                # blue_leader_id = min(env.BUAV_ids) if env.BUAV_ids else None
                
                maneuver_cmds_all = {}
                target_index_all = {}
                done = False
                ep_return = 0.0
                ep_return_by_id = {id: 0.0 for id in student_ids} # 个体回报（tensorboard 按 id 记录）
                ep_threat_delta_psi_abs_by_id = {id: [] for id in student_ids} # 告警状态下 |threat_delta_psi| 采样（tensorboard behavior 按 id 记录）
                ep_out_of_bounds_by_id = {id: 0 for id in student_ids} # 本回合是否出界（0/1，tensorboard behavior 按 id 记录）
                ep_ammo_exhausted_by_id = {id: 0 for id in student_ids} # 本回合是否打光中距弹 ammo=0（0/1）

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
                    whole_sec = count % action_cycle_multiplier == 0
                    if whole_sec:
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            
                            # 决策前把“本轮已被队友选中的敌机”写进本机观测；队长决策时先清空本轮记录
                            if enm_chosen_visable:
                                if UAV.side == 'r':
                                    our_alive_msl = env.alive_r_missiles
                                    # our_enm_index_chosen = red_enm_index_chosen
                                    # if id == red_leader_id:
                                    #     our_enm_index_chosen.clear()
                                    #     env.allocatedIDs_r2b.clear()
                                else:
                                    our_alive_msl = env.alive_b_missiles
                                    # our_enm_index_chosen = blue_enm_index_chosen
                                    # if id == blue_leader_id:
                                    #     our_enm_index_chosen.clear()
                                    #     env.allocatedIDs_b2r.clear()
                                our_enm_index_chosen = set()
                                our_enm_index_chosen.clear()
                                # 把在途导弹的目标index写入集合
                                for msl in our_alive_msl:
                                    our_enm_index_chosen.add(env.id2index(UAV.id, msl.target_id))
                                enm_chosen_vec = np.zeros_like(agent_inputs_[id]['enm_chosen'])
                                for chosen_i in our_enm_index_chosen:
                                    if 0 <= chosen_i < len(enm_chosen_vec):
                                        enm_chosen_vec[chosen_i] = 1
                                agent_inputs_[id]['enm_chosen'] = enm_chosen_vec
                            
                            is_student = UAV.side == 'r'
                            # 保存上一步经验（仅学生，决策步对齐，统一走 append_experience）
                            if is_student and Last_Obs[id] is not None and count > 0:
                                LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id],
                                                                   Last_Actions[id], Rewards[id], Curr_States[id],
                                                                   done=0, active_mask=not UAV.dead)

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
                                fire_target = enm_idx
                                maneuver_cmds_all[id] = [vert, hor]
                                target_index_all[id] = enm_idx
                                fire_flag = bool(fire)
                                # 记录动作供经验池使用（仅学生，log_prob 为行为策略真实值，update 直接读取）
                                if is_student:
                                    Last_Actions[id] = {
                                        "vert": vert, "hor": hor, "fire": fire,
                                        "enm_idx": enm_idx,
                                        "log_prob_enm": float(actor_out['log_prob_enm'].item()),
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

                            # # 选完改变后续观测，表示有些目标已经被选了，后续决策的时候能看到对应位置的enm_chosen发生变化
                            # # enm_idx 可能等于 enm_num（规则智能体的虚目标/null_token 槽位），越界的不计入
                            # if not UAV.dead and 0 <= enm_idx < len(enm_chosen_vec):
                            #     our_enm_index_chosen.add(enm_idx)
                            if enm_chosen_visable:
                                if id == env.RUAV_ids[-1]:
                                    env.allocatedIDs_r2b = set([env.index2id(id, i) for i in our_enm_index_chosen])
                                if id == env.BUAV_ids[-1]:
                                    env.allocatedIDs_b2r = set([env.index2id(id, i) for i in our_enm_index_chosen])
                            
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
                    Curr_States = {}
                    for id in env.UAV_ids:
                        done, reward_dict = env.combat_terminate_and_reward(
                            id=id, action_cycle_multiplier=action_cycle_multiplier)
                        if id in Rewards:
                            Rewards[id] = sum(reward_dict.values())
                            ep_return += Rewards[id]
                            ep_return_by_id[id] += Rewards[id]
                    # # 记录训练方漏选状况
                    # if whole_sec and len(env.alive_r_uavs) > 0:
                    #     ignore_num_count += 1
                        # ignore_enm_num_sum += env.r_ignored_enm_num
                    
                    # 记录当前状态
                    for id in env.UAV_ids:
                        Curr_States[id], _, state_obs = env.get_obs(id, reward_fn=1)
                        # 红方学生处于告警威胁时，记录 |threat_delta_psi|（观测中 RWR_infos 只保留最近的一个威胁）
                        if id in ep_threat_delta_psi_abs_by_id and not env.find_uav_with_id(id).dead:
                            rwr = state_obs.get('RWR_infos')
                            if rwr and rwr.get('warning', 0) > 0:
                                threat_abs = float(np.arccos(np.clip(rwr.get('cos_threat_delta_psi', 1.0), -1.0, 1.0)))
                                ep_threat_delta_psi_abs_by_id[id].append(threat_abs)
                        # 红方学生行为统计：出界 & 打光中距弹（逐帧锁存，一旦置 1 不再清零）
                        if id in ep_out_of_bounds_by_id:
                            u = env.find_uav_with_id(id)
                            # 出界即死，死后位置不变，out_cage 仍返回 True，故 dead & out_cage 可判定出界死因
                            if u.dead and env.out_cage(u):
                                ep_out_of_bounds_by_id[id] = 1
                            if u.ammo == 0:
                                ep_ammo_exhausted_by_id[id] = 1

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

                # 汇总所有飞机的开火样本
                fire_hit_records = []
                for uav in env.UAVs:
                    fire_hit_records += uav.fire_hit_records

                # 学生侧每 id 开火/命中统计（只统计已结算：hit_record 1=命中, 0=脱靶；None=未结算的一并丢弃）
                fire_count_by_id = {}
                hit_count_by_id = {}
                for id in student_ids:
                    recs = [r for r in getattr(env.find_uav_with_id(id), 'fire_hit_records', []) if r.get('hit_record') is not None]
                    fire_count_by_id[id] = len(recs)
                    hit_count_by_id[id] = sum(1 for r in recs if r['hit_record'] == 1)
                
                pipe.send({
                    'trans': stacked_trans,
                    'metrics': {
                        'steps': count,
                        'return': ep_return,
                        'return_by_id': ep_return_by_id, # 个体回报
                        # 告警状态下 |threat_delta_psi| 的回合均值（单位：rad；无告警样本时为 None）
                        'threat_delta_psi_abs_by_id': {
                            id: (float(np.mean(v)) if len(v) > 0 else None)
                            for id, v in ep_threat_delta_psi_abs_by_id.items()
                        },
                        # 本回合行为统计：出界标志、打光中距弹标志（0/1）
                        'out_of_bounds_by_id': ep_out_of_bounds_by_id,
                        'ammo_exhausted_by_id': ep_ammo_exhausted_by_id,
                        'fire_count_by_id': fire_count_by_id, # 每 id 本回合已结算开火次数
                        'hit_count_by_id': hit_count_by_id, # 每 id 本回合命中次数
                        'event_return': 0,
                        'win': win,
                        'lose': lose,
                        'draw': draw,
                        'BVR_perish_together': all_dead and env.t <= args.max_episode_len - 10,
                        # 'ignored_enm_num': ignore_enm_num_sum / max(ignore_num_count, 1)
                    },
                    'opp_name': opp_name,
                    'fire_hit_records': fire_hit_records,
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
    max_steps_of_maneuver=4 * 165e4,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=0.001,
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
    update_interval_of_batch=1,
    test_interval_dp=3333,     # 测试窗口触发间隔：累计 2000 决策点（decision_points）触发一次完整测试+日志+保存
    transition_dict_threshold=2048,
    mini_batch_size_mixed=256,
    num_runs=3,
    device=torch.device("cpu"),
    vertices=None,
    resume_dir=None,
    use_supervised_fire = 0, # [新增] 开火命中监督更新开关：仅用 fire_obs->hit_record 监督 fc_bern 头
    supervised_fire_buffer_size = 1000, # [新增] 开火记录滚动缓冲容量（FIFO，像off-policy一样滚动更新）
    supervised_fire_batch_size = 128, # [新增] 每次监督更新从缓冲中采样的批大小
    ppo_with_bern = 1, # [新增] 0 时 PPO 更新跳过开火头 fc_bern（log_prob 不计入 bern 项）
    enm_chosen_visable = 0, # 是否能看到队友的敌机选取
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
        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
        ppo_with_bern=ppo_with_bern)
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
    if resume_dir is not None and not os.path.exists(resume_dir):
        raise FileNotFoundError(f"resume_dir 不存在，已中止以防静默新训：{resume_dir}")
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
    # actor_rein0 只在首次创建时落盘；续训时若已存在则保留原始初始策略，避免被 latest 权重覆盖
    if not os.path.exists(os.path.join(log_dir, "actor_rein0.pt")):
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
    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
    else:
        actor_freeze_until = -1

    # 测试窗口变量：按 decision_points（total_steps_of_maneuver / action_cycle_multiplier）间隔触发完整测试+日志+保存
    next_test_dp = 0
    test_window_idx = 0   # 测试窗口计数器，用于 checkpoint 序号；与 batch_idx（同步 barrier 计数）解耦

    current_max_steps_of_maneuver = int(max_steps_of_maneuver)

    # 学生 id 独立的全局 Buffer
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [],
                             'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    transition_dicts = {uid: copy.deepcopy(empty_transition_dict) for uid in red_student_ids}

    # 开火记录滚动缓冲：跨 batch 累积（FIFO 逐条覆盖），不随 transition_dict 清空重记录
    fire_records_buffer = collections.deque(maxlen=int(supervised_fire_buffer_size))
    # 若中断续训，恢复已保存的开火监督 buffer
    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
    if os.path.exists(fire_buffer_path):
        try:
            with open(fire_buffer_path, "rb") as f:
                fb_data = pickle.load(f)
            if isinstance(fb_data, dict):
                loaded_records = fb_data.get("records", [])
                loaded_maxlen = fb_data.get("maxlen", int(supervised_fire_buffer_size))
            elif isinstance(fb_data, collections.deque):
                loaded_records = list(fb_data)
                loaded_maxlen = fb_data.maxlen
            else:
                loaded_records = list(fb_data)
                loaded_maxlen = int(supervised_fire_buffer_size)
            fire_records_buffer.clear()
            fire_records_buffer.extend(loaded_records[-loaded_maxlen:])
            print(f"Loaded fire records buffer from {fire_buffer_path}. Size: {len(fire_records_buffer)}")
        except Exception as e:
            print(f"Failed to load fire records buffer: {e}")
    
    # 基于胜率的在线 EMA
    ema_score = 0.5
    ema_step = 0
    target_p1 = 0.65

    # 主循环
    while True:
        while total_steps_of_maneuver < current_max_steps_of_maneuver:
            # 获取当前学生队共享策略权重 (CPU，单份同构)
            current_actor_weights = {k: v.cpu() for k, v in actor.state_dict().items()}
            do_test_window = 0 #  这个batch默认开始不测试
            # [测试] 按 decision_points 间隔触发：累计 2000 dp 做一次完整规则测试
            # 预判：本 batch 采完后累计 dp 约为当前 dp + action_cycle_multiplier * num_workers（每回合至少 1 决策点）
            dp_prev = total_steps_of_maneuver / action_cycle_multiplier
            if num_runs > 0 and dp_prev >= next_test_dp:
                do_test_window = 1  # 这个batch有测试
                test_window_idx += 1
                print(f"\n>>> [Per-Batch Test] Testing student team vs rules (window {test_window_idx})...")
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
                            'dt_maneuver': 0.2, # dt_maneuver,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'red_init_num': red_init_num,
                            'blue_init_num': blue_init_num,
                            'red_init_ammo': red_init_ammo,
                            'blue_init_ammo': blue_init_ammo,
                            'temperature': 0.3,  # 测试采样温度：<1趋于贪婪决策
                            'enm_chosen_visable': enm_chosen_visable,
                        }
                    )
                    test_tasks.append(obj)

                # 阻塞等待全部测试完成后再进入采样
                test_results = [t.get() for t in test_tasks]
                outcomes = {r['rules_num']: r['result'] for r in test_results}
                outcomes_perish = {r['rules_num']: r['perish_togethers'] / num_runs for r in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, dp_prev)
                    print(f"  [Test Result] Rule_{r_num}: {score}")

                # 所有对手的平均Score和超视距双杀率
                avg_score = np.mean(list(outcomes.values()))
                avg_perish_together = np.mean(list(outcomes_perish.values()))
                logger.add("test/avg_score", avg_score, dp_prev)
                logger.add("test/BVR perish together", avg_perish_together, dp_prev)

                # 当前策略的 KD 比统计（跨全部对手汇总击杀/死亡，不分对手编号）
                total_kills = sum(r['kill_count'] for r in test_results)
                total_dies = sum(r['die_count'] for r in test_results)
                log_10_KD = np.log10(np.clip(total_kills / (total_dies + 1e-6), 1/20, 20))
                logger.add("test/log_10_KD", log_10_KD, dp_prev)

                # 将本次测试的胜负平明细按对手追加到 csv（每测试窗口一轮，一行一个对手）
                test_csv_path = os.path.join(log_dir, "test_vs_rules.csv")
                csv_header = ["window_idx", "steps", "rule_num",
                              "wins", "loses", "draws", "score",
                              "perish_together", "num_runs"]
                _need_header = not os.path.exists(test_csv_path)
                with open(test_csv_path, 'a', newline='', encoding='utf-8-sig') as _f:
                    _w = csv.writer(_f)
                    if _need_header:
                        _w.writerow(csv_header)
                    for r in test_results:
                        _w.writerow([
                            batch_idx,
                            int(dp_prev),
                            r['rules_num'],
                            r['wins'], r['loses'], r['draws'],
                            float(r['result']),
                            float(r['perish_togethers'] / num_runs),
                            num_runs,
                        ])

            # 分发采样任务给 Worker
            worker_metrics_buffer = []

            # 对手池：采样一律从限定容量的 elite_elo_ratings（真实 Elo 表）取
            effective_pool = elite_elo_ratings

            selected_opponents = []
            for rank in range(num_workers):
                probs, opponent_keys = get_opponent_probabilities(
                    effective_pool, target_elo=main_agent_elo,
                    SP_type=self_play_type, sigma=sigma_elo, compete_old_rate=compete_old_rate,
                    deltaFSP_epsilon=deltaFSP_epsilon)
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
                        if not os.path.exists(ap):
                            # 权重文件缺失：直接抛错，禁止静默降级到 Rule_0
                            raise FileNotFoundError(
                                f"对手 {selected_opponent_name} 的权重文件不存在: {ap}")
                        shared_opp_sd = torch.load(ap, map_location='cpu')
                        # 蓝机 id = 学生 id + 200（同编队位次：1→201, 2→202）
                        opp_data = {uid + 200: shared_opp_sd for uid in red_student_ids}
                    else:
                        raise ValueError(
                            f"无法解析的 NN 对手名称: {selected_opponent_name}")

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
                    'enm_chosen_visable': enm_chosen_visable,
                    'ppo_with_bern': ppo_with_bern,
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
            batch_total_steps_of_maneuver = 0
            batch_wins = 0
            batch_loss_cnt = 0
            batch_draw_cnt = 0
            batch_bvr_perish_together_cnt = 0
            batch_fire_supervised = 0
            batch_total_return = 0
            batch_ignored_enm_sum = 0.0  # 各回合平均漏选敌机数累加
            batch_return_by_id = {uid: 0 for uid in red_student_ids} # 个体回报累加器
            batch_threat_delta_psi_abs_by_id = {uid: [] for uid in red_student_ids} # 个体告警方位偏差样本累加器
            batch_out_of_bounds_by_id = {uid: 0 for uid in red_student_ids} # 个体出界回合数累加器
            batch_ammo_exhausted_by_id = {uid: 0 for uid in red_student_ids} # 个体打光中距弹回合数累加器
            batch_fire_count_by_id = {uid: 0 for uid in red_student_ids} # 个体开火次数累加器
            batch_hit_count_by_id = {uid: 0 for uid in red_student_ids} # 个体命中次数累加器

            for res in batch_results:
                l_tr_by_id = res['trans']  # {id: transition_dict}
                metrics = res['metrics']
                opp_name = res['opp_name']

                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")

                batch_total_steps_of_maneuver += metrics['steps']
                batch_total_return += metrics['return']
                # 个体回报累加
                for uid, r_id in metrics.get('return_by_id', {}).items():
                    if uid in batch_return_by_id:
                        batch_return_by_id[uid] += r_id
                # 个体告警方位偏差累加
                for uid, threat_abs in metrics.get('threat_delta_psi_abs_by_id', {}).items():
                    if uid in batch_threat_delta_psi_abs_by_id and threat_abs is not None:
                        batch_threat_delta_psi_abs_by_id[uid].append(threat_abs)
                # 个体出界 / 打光中距弹 回合数累加
                for uid, oob in metrics.get('out_of_bounds_by_id', {}).items():
                    if uid in batch_out_of_bounds_by_id:
                        batch_out_of_bounds_by_id[uid] += int(oob)
                for uid, ae in metrics.get('ammo_exhausted_by_id', {}).items():
                    if uid in batch_ammo_exhausted_by_id:
                        batch_ammo_exhausted_by_id[uid] += int(ae)
                # 个体开火 / 命中次数累加
                for uid, fc in metrics.get('fire_count_by_id', {}).items():
                    if uid in batch_fire_count_by_id:
                        batch_fire_count_by_id[uid] += int(fc)
                for uid, hc in metrics.get('hit_count_by_id', {}).items():
                    if uid in batch_hit_count_by_id:
                        batch_hit_count_by_id[uid] += int(hc)
                if metrics.get('BVR_perish_together', False):
                    batch_bvr_perish_together_cnt += 1
                # batch_ignored_enm_sum += metrics.get('ignored_enm_num', 0.0)

                if metrics['win']: batch_wins += 1
                elif metrics['lose']: batch_loss_cnt += 1
                else: batch_draw_cnt += 1

                # 按 id 聚合经验（worker 已堆叠为 dict-of-arrays，递归合并；dict 字段禁用 extend）
                for uid, uid_trans in l_tr_by_id.items():
                    if uid not in transition_dicts:
                        continue
                    for k in transition_dicts[uid]:
                        transition_dicts[uid][k] = merge_transition_tree(transition_dicts[uid][k], uid_trans[k])

                # [新增] 收集本回合已结算的开火命中记录，直接拼接入滚动缓冲
                fire_records = [r for r in res.get('fire_hit_records', []) if r.get('hit_record') is not None]
                fire_records_buffer.extend(fire_records)
                batch_fire_supervised += len(fire_records)
                
                # ELO 更新：仅当对手已入 elite 池时才更新双方 Elo；未入池对手不参与 Elo 记账
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                if opp_name in elite_elo_ratings:
                    prev_main_elo = main_agent_elo
                    adv_elo = elite_elo_ratings[opp_name]
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                    available_opps["__CURRENT_MAIN__"] = main_agent_elo
                    elite_elo_ratings[opp_name] = new_adv_elo
            
        
            # 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps_of_maneuver += batch_total_steps_of_maneuver
            batch_idx += 1

            # 测试窗口精确触发：累计 decision_points 跨过阈值才算一次完整窗口
            dec_steps_total = total_steps_of_maneuver / action_cycle_multiplier

            # ---- 批次聚合指标（每测试窗口才记录） ----
            if do_test_window:
                # 记录批次聚合指标
                batch_score = (batch_wins + batch_draw_cnt * 0.5) / num_workers
                ema_step += 1
                alpha_ema = 0.1
                if ema_step == 1:
                    ema_score = batch_score
                else:
                    ema_score = (1 - alpha_ema) * ema_score + alpha_ema * batch_score

                filtered_score = ema_score
                logger.add("train_plus/batch_score", batch_score, dec_steps_total)
                logger.add("train_plus/filtered_score", filtered_score, dec_steps_total)
                logger.add("train_plus/target_p1", target_p1, dec_steps_total)

                # 推进测试窗口阈值
                next_test_dp = max(dp_prev + test_interval_dp, dec_steps_total)

            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, dec_steps_total)
            # 个体回报（每 id 一条）
            for uid, r_id in batch_return_by_id.items():
                logger.add(f"train/1 avg_episode_return_id{uid}", r_id / num_workers, dec_steps_total)
            # 告警威胁下的方位偏差均值（每 id 一条，单位：度）
            for uid, threat_abs_list in batch_threat_delta_psi_abs_by_id.items():
                if len(threat_abs_list) > 0:
                    logger.add(f"behavior/abs_threat_delta_psi_id{uid}", float(np.mean(threat_abs_list)) * 180 / np.pi, dec_steps_total)
            # 出界率 / 打光中距弹率（每 id 一条，本 batch 内回合占比）
            for uid, oob_cnt in batch_out_of_bounds_by_id.items():
                logger.add(f"behavior/out_of_bounds_rate_id{uid}", oob_cnt / num_workers, dec_steps_total)
            for uid, ae_cnt in batch_ammo_exhausted_by_id.items():
                logger.add(f"behavior/ammo_exhausted_rate_id{uid}", ae_cnt / num_workers, dec_steps_total)
            # 每回合已结算开火次数 / 命中率（每 id 一条；命中率 = batch 内总命中 / 总已结算开火，无已结算开火时跳过该点）
            for uid, fc_cnt in batch_fire_count_by_id.items():
                logger.add(f"behavior/fire_count_id{uid}", fc_cnt / num_workers, dec_steps_total)
                hc_cnt = batch_hit_count_by_id.get(uid, 0)
                if fc_cnt > 0:
                    logger.add(f"behavior/hit_rate_id{uid}", hc_cnt / fc_cnt, dec_steps_total)
            logger.add("train/2 win", batch_wins / num_workers, dec_steps_total)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, dec_steps_total)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, dec_steps_total)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, dec_steps_total)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, dec_steps_total)
            # logger.add("train/2 ignored_enm_num", batch_ignored_enm_sum / num_workers, dec_steps_total)
            logger.add("agent/ episode_step", batch_idx * num_workers, dec_steps_total)
            logger.add("agent/ batch_step", batch_idx, dec_steps_total)

            # 更新，保存与维护 (Checkpoint & Pool)
            if batch_idx % update_interval_of_batch == 0 and \
                len(transition_dicts[red_student_ids[0]]['dones']) >= transition_dict_threshold:

                # ELO 相对位置（基于 elite 池中的真实 Elo，elo_ratings 仅存版本号不可再用）
                all_keys = list(elite_elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                target_pool_keys = rule_keys + latest_rein_keys
                avg_pool_elo = np.mean([elite_elo_ratings[k] for k in target_pool_keys]) if target_pool_keys else main_agent_elo
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, dec_steps_total)

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
                    logger.add("train/7 actor_loss", m.get('actor_loss', 0.0), dec_steps_total)
                    logger.add("train/8 critic_loss", m.get('critic_loss', 0.0), dec_steps_total)
                    logger.add("train/9 entropy", m.get('entropy', 0.0), dec_steps_total)
                    logger.add("train/10 approx_kl", m.get('kl', 0.0), dec_steps_total)
                    logger.add("train/11 clip_frac", m.get('clip_frac', 0.0), dec_steps_total)
                    # 记录未禁火 mask 内的开火概率极值（全队共享策略，单条曲线）
                    logger.add("train/12 max_fire_prob", m.get('fire_prob_max', float('nan')), dec_steps_total)
                    logger.add("train/13 min_fire_prob", m.get('fire_prob_min', float('nan')), dec_steps_total)
                    
                # [新增] 开火命中监督更新：从滚动缓冲采样，仅更新 actor.net.fc_bern，不影响机动策略
                if use_supervised_fire and len(fire_records_buffer) > 100: #  and\
                    fire_sl_batch = random.sample(list(fire_records_buffer), min(int(supervised_fire_batch_size), len(fire_records_buffer)))
                    agent.supervised_fire_update(fire_sl_batch, epochs=1, batch_size=int(supervised_fire_batch_size), fire_lr_ratio=1)
                    logger.add("train_plus/supervised_fire_buffer_size", len(fire_records_buffer), dec_steps_total)
                    logger.add("train_plus/supervised_fire_samples", batch_fire_supervised, dec_steps_total)
                    logger.add("train/supervised_fire_loss", getattr(agent, 'last_supervised_fire_loss', 0.0), dec_steps_total)

                print(f"Step {dec_steps_total}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # A. 保存模型（同构单份，无 _uid 后缀）
                actor_team_key = f"actor_rein{batch_idx}"
                if batch_idx % 10 == 0:
                    torch.save(actor.state_dict(), os.path.join(log_dir, f"{actor_team_key}.pt"))
                    torch.save(critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                    print(f"Saved Checkpoint team: {actor_team_key}")
                    # 每次落盘新的 actor_rein 时记录：横轴当前步数，纵轴 actor_rein 后面的数字（batch_idx）
                    logger.add("checkpoint/actor_rein_idx", batch_idx, dec_steps_total)
                    # 只注册已落盘的版本到可用注册表（版本号=batch_idx），保证采样时总能加载到权重
                    available_opps[actor_team_key] = batch_idx

                # 清空各学生 id Buffer
                for uid in red_student_ids:
                    transition_dicts[uid] = copy.deepcopy(empty_transition_dict)

                # 全量历史记录
                available_opps["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                available_opps["__LAST_UPDATE_BATCH__"] = batch_idx

                # 保存全量日志
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(available_opps, f, ensure_ascii=False, indent=2)

                # 精英池快照
                save_elite = copy.deepcopy(elite_elo_ratings)
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)

                #  日志记录 (Logging) - 保持不变，展示的是精英池状态
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, dec_steps_total)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    
                    # 升序：Elo 低（弱对手）在前
                    rule_elos = sorted([v for k, v in elite_elo_ratings.items() if k.startswith("Rule")], reverse=False) # 升序排列
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, dec_steps_total)

                    # 口径：把 main 合入规则池一起归一化（不含历史 RL 对手），
                    # 0%=最弱 Rule（或 main 自己垫底），100%=最强 Rule（或 main 自己顶位）；
                    # 规则池只有一个元素或 main 独占时 spread=0，按约定输出 0
                    ref_values = [main_agent_elo] + list(rule_elos)
                    ref_min = float(np.min(ref_values))
                    ref_max = float(np.max(ref_values))
                    ref_spread = ref_max - ref_min
                    curr_rate = 0.0 if ref_spread == 0 else (main_agent_elo - ref_min) / ref_spread

                    # Elo分数没有稳定不许记录
                    if total_steps_of_maneuver >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rate_normed %", curr_rate * 100, dec_steps_total)

                    # 池容量占用（规则 + NN 总数）
                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    pool_total = len([k for k in elite_elo_ratings.keys() if not k.startswith("__")])
                    logger.add("Elo/History_Pool_Size", hist_count, dec_steps_total)
                    logger.add("Elo/Pool_Total_Size", pool_total, dec_steps_total)

                # B. 精英池维护（容量只对 NN 生效，规则对手永不踢出；不再定期全量重刷）
                if hist_agent_as_opponent and os.path.exists(os.path.join(log_dir, f"{actor_team_key}.pt")):
                    new_entry_elo = main_agent_elo
                    nn_keys = [k for k in elite_elo_ratings.keys() if not k.startswith("__") and not k.startswith("Rule")]
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

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, dec_steps_total)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, dec_steps_total)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, dec_steps_total)

                # 例行保存优化器状态（同构单份）
                opt_save = {
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }
                torch.save(opt_save, os.path.join(log_dir, "optimizers_state.pt"))
                
                # 保存开火监督数据 buffer
                if use_supervised_fire:
                    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
                    with open(fire_buffer_path, "wb") as f:
                        pickle.dump({"records": list(fire_records_buffer), "maxlen": fire_records_buffer.maxlen}, f)
                    print(f"Saved fire records buffer to {fire_buffer_path}. Size: {len(fire_records_buffer)}")


                available_opps["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                available_opps["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(available_opps, f, ensure_ascii=False, indent=2)

        #  [新增] 达到 max_steps_of_maneuver 后的交互逻辑
        print(f"\n--- Target steps reached: {total_steps_of_maneuver} / {current_max_steps_of_maneuver} ---")
        # 【新增】明确告知用户 Worker 当前状态
        print("All simulation workers are now idling safely. System is paused.") 
        
        inp = input(f"Enter new max_steps_of_maneuver (current {total_steps_of_maneuver}), or press Enter to exit: ")
        
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