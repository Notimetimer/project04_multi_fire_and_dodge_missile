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
import time  # 确保引入 time 模块
import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import traceback # [新增]
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
    """采样 worker：红方学生各 id 独立 ActorWrapper（不改 id），蓝方为规则或历史 nn（对手）。
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
    # 每架飞机各持独立 ActorWrapper（红方学生 + 蓝方历史 nn 对手共用此字典，按 id 区分）
    actor_dict = {}

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RUN_EPISODE':
                student_actor_weights = packet[0]   # {id: state_dict} 红方学生各 id 权重（不改 id）
                opp_name, opp_type, opp_data = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])
                ppo_with_bern = settings.get('ppo_with_bern', 1)  # 0 时 RL 的 log_prob/熵不含 fire 分支

                # 加载红方学生各 id 的 actor 权重（学生 id 与环境红机 id 一致）
                for id, sd in student_actor_weights.items():
                    if id not in actor_dict:
                        actor_dict[id] = ActorWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker, ppo_with_bern=ppo_with_bern,
                        )
                    actor_dict[id].load_state_dict(sd)
                    actor_dict[id].eval()

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
                                # 神经网络决策（红方学生 / 蓝方历史 nn）
                                actor_rule = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                                with torch.no_grad():
                                    actor_out = actor_dict[id].get_action(
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

                # 汇总所有飞机的开火样本
                fire_hit_records = []
                for uav in env.UAVs:
                    fire_hit_records += uav.fire_hit_records
                
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
    update_interval_of_batch=1,
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
):
    """多智能体 MAPPO 训练主函数。
    异构：每个 UAV id 维护独立的 ActorWrapper/critic_wrapper/MAPPOHybrid。
    保存命名：actor_rein{batch_idx}_{id}.pt / critic_{id}.pt。"""
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

    # 4. 为每个学生 id 创建独立的 actor/critic/MAPPOHybrid（完全异构）
    actors = {}      # 学生 id -> ActorWrapper
    critics = {}     # 学生 id -> critic_wrapper
    agents = {}      # 学生 id -> MAPPOHybrid
    for uid in red_student_ids:
        actors[uid] = ActorWrapper(
            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
            ppo_with_bern=ppo_with_bern)
        critics[uid] = critic_wrapper(
            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
        agents[uid] = MAPPOHybrid(
            actor=actors[uid], critic=critics[uid],
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

    # 中断续训：按 id 加载 actor_rein{latest_batch}_{id}.pt 和 critic_{id}.pt
    if resume_dir is not None and os.path.exists(resume_dir):
        actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
        if actor_files:
            def extract_batch(f):
                m = re.search(r'actor_rein(\d+)_', os.path.basename(f))
                return int(m.group(1)) if m else -1
            latest_batch = max(extract_batch(f) for f in actor_files)
            for uid in red_student_ids:
                ap = os.path.join(log_dir, f"actor_rein{latest_batch}_{uid}.pt")
                if os.path.exists(ap):
                    actors[uid].load_state_dict(torch.load(ap, map_location=device))
                    print(f"Loaded actor id={uid} from: {ap}")
                cp = os.path.join(log_dir, f"critic_{uid}.pt")
                if os.path.exists(cp):
                    critics[uid].load_state_dict(torch.load(cp, map_location=device))
                    print(f"Loaded critic id={uid} from: {cp}")
            batch_idx = latest_batch
        else:
            batch_idx = 0

        opt_path = os.path.join(log_dir, "optimizers_state.pt")
        if os.path.exists(opt_path):
            try:
                opt_states = torch.load(opt_path, map_location=device)
                for uid in red_student_ids:
                    if str(uid) in opt_states:
                        agents[uid].actor_optimizer.load_state_dict(opt_states[str(uid)]['actor_optimizer'])
                        agents[uid].critic_optimizer.load_state_dict(opt_states[str(uid)]['critic_optimizer'])
                print("Loaded optimizer states.")
            except Exception as e:
                print(f"Failed to load optimizers: {e}")
    else:
        batch_idx = 0

    # logger
    logger = TensorBoardLogger(
        log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 存储在线训练前的网络参数（异构命名，仅学生）
    for uid in red_student_ids:
        torch.save(actors[uid].state_dict(), os.path.join(log_dir, f"actor_rein0_{uid}.pt"))
        torch.save(critics[uid].state_dict(), os.path.join(log_dir, f"critic_{uid}.pt"))

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
    total_steps_of_maneuver = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
    else:
        actor_freeze_until = -1

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
            # 获取当前学生队各 id 策略权重 (CPU)
            current_actor_weights = {uid: {k: v.cpu() for k, v in actors[uid].state_dict().items()}
                                     for uid in red_student_ids}

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
                            'red_init_ammo': red_init_ammo,
                            'blue_init_ammo': blue_init_ammo,
                            'temperature': 0.3,  # 测试采样温度：<1趋于贪婪决策
                        }
                    )
                    test_tasks.append(obj)

                # 阻塞等待全部测试完成后再进入采样
                test_results = [t.get() for t in test_tasks]
                outcomes = {r['rules_num']: r['result'] for r in test_results}
                outcomes_perish = {r['rules_num']: r['perish_togethers'] / num_runs for r in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps_of_maneuver/action_cycle_multiplier)
                    print(f"  [Test Result] Rule_{r_num}: {score}")

                # 所有对手的平均Score和超视距双杀率
                avg_score = np.mean(list(outcomes.values()))
                avg_perish_together = np.mean(list(outcomes_perish.values()))
                logger.add("test/avg_score", avg_score, total_steps_of_maneuver/action_cycle_multiplier)
                logger.add("test/BVR perish together", avg_perish_together, total_steps_of_maneuver/action_cycle_multiplier)

                # 将本次测试的胜负平明细按对手追加到 csv（每 batch 测试一轮，一行一个对手）
                test_csv_path = os.path.join(log_dir, "test_vs_rules.csv")
                csv_header = ["batch_idx", "steps", "rule_num",
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
                            int(total_steps_of_maneuver / action_cycle_multiplier),
                            r['rules_num'],
                            r['wins'], r['loses'], r['draws'],
                            float(r['result']),
                            float(r['perish_togethers'] / num_runs),
                            num_runs,
                        ])

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
                    # 历史团队：按学生（红方）id 读盘，再重映射为蓝机 id 打包（采样对手需要改 id）
                    m = re.match(r'^actor_rein(\d+)$', selected_opponent_name)
                    if m:
                        old_batch = int(m.group(1))
                        opp_data = {}
                        for uid in red_student_ids:
                            ap = os.path.join(log_dir, f"actor_rein{old_batch}_{uid}.pt")
                            if os.path.exists(ap):
                                # 蓝机 id = 学生 id + 200（同编队位次：1→201, 2→202）
                                opp_data[uid + 200] = torch.load(ap, map_location='cpu')
                        if not opp_data:
                            opp_type = 'rule'
                            opp_data = 0
                    else:
                        opp_type = 'rule'
                        opp_data = 0

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
            batch_return_by_id = {uid: 0 for uid in red_student_ids} # 个体回报累加器

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

                # [新增] 收集本回合已结算的开火命中记录，直接拼接入滚动缓冲
                fire_records = [r for r in res.get('fire_hit_records', []) if r.get('hit_record') is not None]
                fire_records_buffer.extend(fire_records)
                batch_fire_supervised += len(fire_records)
                
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
            total_steps_of_maneuver += batch_total_steps_of_maneuver
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
            logger.add("train_plus/batch_score", batch_score, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train_plus/filtered_score", filtered_score, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train_plus/target_p1", target_p1, total_steps_of_maneuver/action_cycle_multiplier)

            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            # 个体回报（每 id 一条）
            for uid, r_id in batch_return_by_id.items():
                logger.add(f"train/1 avg_episode_return_id{uid}", r_id / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train/2 win", batch_wins / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps_of_maneuver/action_cycle_multiplier)
            logger.add("agent/ batch_step", batch_idx, total_steps_of_maneuver/action_cycle_multiplier)

            # 更新，保存与维护 (Checkpoint & Pool)
            if batch_idx % update_interval_of_batch == 0 and \
                len(transition_dicts[red_student_ids[0]]['dones']) >= transition_dict_threshold:

                # ELO 相对位置
                all_keys = list(elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                target_pool_keys = rule_keys + latest_rein_keys
                avg_pool_elo = np.mean([elo_ratings[k] for k in target_pool_keys])
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, total_steps_of_maneuver/action_cycle_multiplier)

                # 各学生 id 独立 PPO 更新
                update_metrics_by_id = {}
                for uid in red_student_ids:
                    td = transition_dicts[uid]
                    if len(td['dones']) == 0:
                        continue
                    m = agents[uid].update(td, adv_normed=1, mini_batch_size=mini_batch_size_mixed)
                    update_metrics_by_id[uid] = m
                    logger.add(f"train/7 actor_loss_id{uid}", m.get('actor_loss', 0.0), total_steps_of_maneuver/action_cycle_multiplier)
                    logger.add(f"train/8 critic_loss_id{uid}", m.get('critic_loss', 0.0), total_steps_of_maneuver/action_cycle_multiplier)
                    logger.add(f"train/9 entropy_id{uid}", m.get('entropy', 0.0), total_steps_of_maneuver/action_cycle_multiplier)
                    logger.add(f"train/10 approx_kl_id{uid}", m.get('kl', 0.0), total_steps_of_maneuver/action_cycle_multiplier)
                    logger.add(f"train/11 clip_frac_id{uid}", m.get('clip_frac', 0.0), total_steps_of_maneuver/action_cycle_multiplier)

                # 团队级聚合指标
                if update_metrics_by_id:
                    # 分 id 记录未禁火 mask 内的开火概率极值
                    for uid, m in update_metrics_by_id.items():
                        logger.add(f"train/12 max_fire_prob_id{uid}", m.get('fire_prob_max', float('nan')), total_steps_of_maneuver/action_cycle_multiplier)
                        logger.add(f"train/13 min_fire_prob_id{uid}", m.get('fire_prob_min', float('nan')), total_steps_of_maneuver/action_cycle_multiplier)
                # [新增] 开火命中监督更新：从滚动缓冲采样，仅更新 actor.net.fc_bern，不影响机动策略
                if use_supervised_fire and len(fire_records_buffer) > 100: #  and\
                    fire_sl_batch = random.sample(list(fire_records_buffer), min(int(supervised_fire_batch_size), len(fire_records_buffer)))
                    for uid in red_student_ids:
                        agents[uid].supervised_fire_update(fire_sl_batch, epochs=1, batch_size=int(supervised_fire_batch_size), fire_lr_ratio=1)
                    logger.add("train_plus/supervised_fire_buffer_size", len(fire_records_buffer), total_steps_of_maneuver)
                    logger.add("train_plus/supervised_fire_samples", batch_fire_supervised, total_steps_of_maneuver)
                print(f"Step {total_steps_of_maneuver}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # A. 保存模型（异构命名，仅学生）
                actor_team_key = f"actor_rein{batch_idx}"
                if batch_idx % 10 == 0:
                    for uid in red_student_ids:
                        torch.save(actors[uid].state_dict(), os.path.join(log_dir, f"{actor_team_key}_{uid}.pt"))
                        torch.save(critics[uid].state_dict(), os.path.join(log_dir, f"critic_{uid}.pt"))
                    print(f"Saved Checkpoint team: {actor_team_key}")
                    # 只注册已落盘的版本，保证自博弈采样对手时总能加载到对应权重
                    elo_ratings[actor_team_key] = main_agent_elo

                # 清空各学生 id Buffer
                for uid in red_student_ids:
                    transition_dicts[uid] = copy.deepcopy(empty_transition_dict)

                # B. 精英池维护
                if total_steps_of_maneuver >= WARM_UP_STEPS:
                    elo_ratings[opp_name] = main_agent_elo

                # 全量历史记录
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx

                # 定期重刷精英池
                if total_steps_of_maneuver > 0 and total_steps_of_maneuver % 1e6 == 0:
                    print(f"\n>>> [Elite Pool Refresh] Refreshing elite_elo_ratings at {total_steps_of_maneuver} steps...")
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
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps_of_maneuver/action_cycle_multiplier)
                    
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
                    logger.add("Elo_Centered/EloDiffToMean", elo_diff_to_thres, total_steps_of_maneuver/action_cycle_multiplier)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps_of_maneuver/action_cycle_multiplier)

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
                        logger.add("Elo_Centered/Current_rate_normed %", curr_rate * 100, total_steps_of_maneuver/action_cycle_multiplier)

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
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps_of_maneuver/action_cycle_multiplier)

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, total_steps_of_maneuver/action_cycle_multiplier)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps_of_maneuver/action_cycle_multiplier)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, total_steps_of_maneuver/action_cycle_multiplier)

                # 例行保存优化器状态（按学生 id）
                opt_save = {}
                for uid in red_student_ids:
                    opt_save[str(uid)] = {
                        'actor_optimizer': agents[uid].actor_optimizer.state_dict(),
                        'critic_optimizer': agents[uid].critic_optimizer.state_dict(),
                    }
                torch.save(opt_save, os.path.join(log_dir, "optimizers_state.pt"))
                
                # 保存开火监督数据 buffer
                if use_supervised_fire:
                    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
                    with open(fire_buffer_path, "wb") as f:
                        pickle.dump({"records": list(fire_records_buffer), "maxlen": fire_records_buffer.maxlen}, f)
                    print(f"Saved fire records buffer to {fire_buffer_path}. Size: {len(fire_records_buffer)}")


                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

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