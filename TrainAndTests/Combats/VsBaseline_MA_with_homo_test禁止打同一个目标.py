"同构基线测试 main：红方全队共享同一份 actor 权重（单份 state_dict），蓝方规则"
"""
复用 VsBaseline_MA.test_worker：其 _is_single_sd 检测到传入为单份 state_dict 时，
所有红机共享同一份权重。checkpoint 来自 CombatMAPPO_parall_homo：actor_rein{batch}.pt（无 _id 后缀）。
异构测试（actor_rein{batch}_{id}.pt）见 VsBaseline_MA.py 的 main。
"""

import os
import sys
import numpy as np
import torch
import argparse
import glob
import csv
import time
import random
import re
from math import *

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from VsBaseline_MA import test_worker, creat_initial_state
from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
from BasicRules_Homo import basic_rules
from Algorithms.ActorWithPointingNet import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field

use_tacview = 2 # 1 动态显示 2 静默写入 acmi_records

# 是否考虑视野限制
POMDP = 1


# ==========================================
# 多智能体测试 Worker：红方神经网络 vs 蓝方规则
# ==========================================
def test_worker(model_state_dict, rules_num,
                env_args, device_name='cpu', num_runs=3,
                vertices=None, save_record=False,
                dt_move=0.07,
                dt_maneuver=0.2,
                action_cycle_multiplier=10,
                min_alt_safe=2000,
                red_init_num=2, blue_init_num=2,
                red_init_ammo=2, blue_init_ammo=2,
                temperature=1.0,  # 玻尔兹曼采样温度，仅本测试函数使用；训练 worker_process 不传此参数
                ):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    红方各机独立 ActorWrapper（共享同一份 model_state_dict），蓝方统一 basic_rules。
    胜负直接读环境端导出的 r_outcome（1 胜 / -1 负 / 0 平，红方视角）。
    """
    device = torch.device(device_name)

    # save_record=False 时不向硬盘写任何东西：tacview 强制关闭、不清空也不生成 pkl/csv
    test_env = ChooseStrategyEnv(env_args, tacview_show=use_tacview if save_record else 0, vertices=vertices)
    test_env.dt_move = dt_move

    # 启动仿真前清空上次遗留的pkl和csv
    if save_record:
        event_log_dir = os.path.join(project_root, 'acmi_records')
        os.makedirs(event_log_dir, exist_ok=True)
        for pattern in ('*.pkl', '*.csv'):
            for f in glob.glob(os.path.join(event_log_dir, pattern)):
                os.remove(f)

    # 红方每架飞机各用一个独立 actor（加载同一份权重），按 id 索引；蓝方保持规则
    actor_dict = None

    wins = 0
    loses = 0
    draws = 0
    perish_togethers = 0
    episode_times = []  # 记录每场耗时
    t_bias = 0

    try:
        for i_episode in range(num_runs):
            ep_start = time.time()
            print(f"开始第{i_episode+1}场对抗")
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()
            test_env.reset(
                red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_num=red_init_num,
                blue_init_num=blue_init_num,
                red_init_ammo=red_init_ammo,
                blue_init_ammo=blue_init_ammo,
                # 红蓝各自独立的位置打乱种子（np.random 已在函数入口固定，序列可复现且场间不同）
                seed=(int(np.random.randint(0, 2**31 - 1)), int(np.random.randint(0, 2**31 - 1))),
                )

            # 首场 reset 后为每架红机各建一个独立 ActorWrapper 并加载权重，按 id 索引
            if actor_dict is None:
                actor_dict = {}
                red_ids = [i for i in test_env.UAV_ids if test_env.find_uav_with_id(i).side == 'r']
                # 网络维度统一从 env_obs_dims.json 读取（由 读取环境参数并存json.py 生成）
                fixed_dim, enm_shape, ally_shape = load_obs_dims()
                # 支持异构输入：model_state_dict 可以是单份 state_dict（同构，所有红机共享）
                # 或 {id: state_dict}（异构，每架红机加载自己的权重）
                _is_single_sd = isinstance(model_state_dict, dict) and \
                    all(isinstance(v, torch.Tensor) for v in model_state_dict.values())
                for rid in red_ids:
                    actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
                    )
                    if _is_single_sd:
                        # 同构：所有红机共享同一份权重
                        actor.load_state_dict(model_state_dict)
                    else:
                        # 异构：{id: state_dict}，按 id 加载；缺省则用第一份权重兜底
                        if rid in model_state_dict:
                            actor.load_state_dict(model_state_dict[rid])
                        else:
                            actor.load_state_dict(next(iter(model_state_dict.values())))
                    actor.eval()
                    actor_dict[rid] = actor

            # 各机上一决策步的动作缓存（按 id 保存，供执行阶段逐机取用）
            maneuver_cmds_all = {}
            target_index_all = {}

            done = False

            test_env.dt_move = dt_move
            test_env.dt_maneuver = dt_maneuver
            test_env.min_alt_safe = min_alt_safe

            # 每架飞机逐帧观测记录，mask调试信息合并到事件csv（save_record=False 时不记录）
            agent_obs_records = {
                _id: {
                    'agent_input': [],
                    'mask_input': [],
                    'rule_input': []
                } for _id in test_env.UAV_ids
            } if save_record else None

            # 环境运行一轮的情况
            steps_of_this_eps = -1

            # 一轮仿真开始
            for count in range(round(env_args.max_episode_len / test_env.dt_maneuver)):
                # 回合结束判断
                steps_of_this_eps += 1
                if test_env.running == False or done:
                    # print('回合结束，时间为：', test_env.t, 's')
                    break
                # 获取观测信息（测试不存经验池，每帧每机只调一次 get_obs）
                agent_inputs_ = {}
                mask_inputs_ = {}
                rule_inputs_ = {}

                for id in test_env.UAV_ids:
                    agent_input, mask_input, rule_input = test_env.get_obs(id, pomdp=POMDP)
                    agent_inputs_[id] = agent_input
                    mask_inputs_[id] = mask_input
                    rule_inputs_[id] = rule_input
                    if save_record and steps_of_this_eps % action_cycle_multiplier == 0:
                        agent_obs_records[id]['agent_input'].append(agent_input)
                        agent_obs_records[id]['mask_input'].append(mask_input)
                        agent_obs_records[id]['rule_input'].append(rule_input)

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    for id in test_env.UAV_ids:
                        UAV = test_env.find_uav_with_id(id)

                        # Agent 产生动作：红方走 actor（采样），蓝方走规则
                        if UAV.side == 'r':
                            # rule_inputs 用 take_some_of_rule_inputs 取子集后单样本堆叠成
                            # selector 期望的 dict-of-array（[B=1]）格式
                            actor_rule = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                            with torch.no_grad():
                                actor_out = actor_dict[id].get_action(
                                    agent_inputs_[id], mask_inputs_[id], actor_rule,
                                    temperature=temperature)
                            vert = int(actor_out['vert_idx'].item())
                            hor = int(actor_out['hor_idx'].item())
                            fire = int(actor_out['fire'].item())
                            enm_idx = int(actor_out['enm_idx'].item())
                            maneuver_cmds_all[id] = [vert, hor]
                            target_index_all[id] = enm_idx
                            fire_target = enm_idx
                            fire_flag = bool(fire)
                        else:
                            # 如果类型是规则智能体而非神经网络
                            fly_action, fire_action, radar_action = basic_rules(rule_inputs_[id], mask_inputs_[id], rules_num=rules_num)
                            maneuver_cmds_all[id] = fly_action['do']
                            target_index_all[id] = int(fly_action['at'])
                            fire_target = int(fire_action['at'])
                            fire_flag = bool(fire_action['do'])

                        # 不考虑同时攻击多个目标
                        if fire_flag:
                            try:
                                test_env.fire_120(id, fire_target, tabu=0)
                            except ValueError as e:
                                # 环境端对虚目标/越界索引抛错，这里仅打印不中断（规则智能体会输出 null_token 索引）
                                print(f"[fire_120 警告] {e}")

                        # 雷达全程开
                        test_env.radar_action(id, 1)

                        # 目标占用记录与禁选约束（与BasicRules_Homo调序一致）：存活敌机未被分配满时，
                        # 禁止队内后续id选择与前面uav相同的目标编号；红蓝双方都按各自 allocatedIDs 记录
                        if not UAV.dead:
                            if UAV.side == 'r':
                                enm_allocated_ids = test_env.allocatedIDs_r2b
                                our_id2index = test_env.r_id2indexinside
                                enm_id2index = test_env.b_id2indexinside
                                enm_alive_uavs = test_env.alive_b_uavs
                                alive_missiles = test_env.alive_r_missiles
                            else:
                                enm_allocated_ids = test_env.allocatedIDs_b2r
                                our_id2index = test_env.b_id2indexinside
                                enm_id2index = test_env.r_id2indexinside
                                enm_alive_uavs = test_env.alive_r_uavs
                                alive_missiles = test_env.alive_b_missiles
                            enm_alive_ids = {uav.id for uav in enm_alive_uavs}
                            # 优先使用在飞导弹锁定的目标作为占用集；无存活导弹时再用当前决策选中的目标
                            if alive_missiles:
                                for missile in alive_missiles:
                                    if missile.target_id is not None:
                                        enm_allocated_ids.add(missile.target_id)
                            else:
                                allocated_eid = test_env.index2id(UAV.id, target_index_all[id])
                                if allocated_eid is not None:
                                    enm_allocated_ids.add(allocated_eid)
                            # 如果当前飞机死了，不影响后面的飞机；没有分配完才对后续id号的飞机传入修改enm_mask
                            if not enm_alive_ids.issubset(enm_allocated_ids):
                                cur_index_in_side = our_id2index[UAV.id]
                                for fid in test_env.UAV_ids:
                                    fUAV = test_env.find_uav_with_id(fid)
                                    # 不对对手侧改obs
                                    if fUAV.side != UAV.side:
                                        continue
                                    # 不往回修改
                                    if our_id2index[fid] <= cur_index_in_side:
                                        continue
                                    # 不重复攻击同一个目标，除非有富余的火力资源
                                    for eid in enm_allocated_ids:
                                        if eid in enm_id2index:
                                            mask_inputs_[fid]['enm'][enm_id2index[eid]] = 0

                # 循环调用底层控制器，环境更新
                flight_cmds = {}
                for id in test_env.UAV_ids:
                    maneuver_cmds = maneuver_cmds_all[id]
                    target_index = target_index_all[id]
                    try:
                        flight_cmds[id] = test_env.maneuver_for_1_enm(id, target_index, maneuver_cmds, rule_inputs_[id])
                    except ValueError as e:
                        # 环境端对虚目标/越界索引抛错，规则智能体全灭时会输出 null_token 索引，用默认平飞指令兜底
                        print(f"[maneuver_for_1_enm 警告] {e}")
                        flight_cmds[id] = [0.0, 0.0, 0.0]
                test_env.step(flight_cmds.values())  # 环境统一更新

                for id in test_env.UAV_ids: # 每架飞机都调用一遍，保证各自的周期事件都被积累
                    done, reward_dict = test_env.combat_terminate_and_reward(id=id, action_cycle_multiplier=action_cycle_multiplier)

                '''显示运行轨迹'''
                # 可视化
                test_env.render(t_bias=t_bias)

            # 结果统计（红方视角）：直接读环境端导出的 r_outcome（1 胜 / -1 负 / 0 平）；
            # 平局且在时限前结束视为双杀（双方同归于尽），超时按剩余机数判 0 也是平
            outcome = test_env.r_outcome if test_env.r_outcome is not None else 0
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                loses += 1
            else:
                draws += 1
            perish_togethers += all(uav.dead for uav in test_env.UAVs) and test_env.t <= env_args.max_episode_len - 10

            # 仿真残余清理
            test_env.clear_render(t_bias=t_bias)
            t_bias += test_env.t

            # 时间记录
            ep_cost = time.time() - ep_start
            episode_times.append(ep_cost)
            print(f"第{i_episode+1}场对抗耗时：{ep_cost:.1f}s, 仿真倍率：{test_env.t/ep_cost:.1f}")

            if save_record:
                # 保存本回合各机的事件日志和mask调试信息，每架飞机一个csv
                for uav in test_env.UAVs:
                    events = getattr(uav, 'cycle_events', [])
                    if not events:
                        continue
                    masks = agent_obs_records[uav.id]['mask_input']
                    event_rows = []
                    for index, event in enumerate(events):
                        mask = masks[index] if index < len(masks) else {}
                        mask_ally = np.asarray(mask.get('ally', [])).tolist()
                        mask_enm = np.asarray(mask.get('enm', [])).tolist()
                        event_rows.append((*event, mask_ally, mask_enm))
                    csv_path = os.path.join(event_log_dir, f'uav_{uav.id}_events.csv')
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(['time', 'id', 'dense_reward', 'end_reward', 'event_reward', 'event',
                                            'mask_ally', 'mask_enm'])
                        writer.writerows(event_rows)

        result = (wins + 0.5 * draws) / num_runs
        return rules_num, result, wins, loses, draws, perish_togethers

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
        result = (wins + 0.5 * draws) / max(len(episode_times), 1)
        return rules_num, result, wins, loses, draws, perish_togethers
    finally:
        test_env.end_render() # 停止发送


if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    # 测试配置
    RULES_NUM = 5      # 蓝方规则编号
    N_EPISODES = 3    # 测试场次  10
    TEMPERATURE = 0.3  # 采样温度：<1更贪婪（机动/目标选取决策趋于确定）

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]

    # 加载同构训练目录下的单份 actor（CombatMAPPO_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "MAPPO_homo-run-20261019-185822"
                           )
    actor_files = glob.glob(os.path.join(LOG_DIR, "actor_rein*.pt"))
    if not actor_files:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到 actor_rein*.pt")
    # 只保留同构文件名 actor_rein{batch}.pt；异构 actor_rein{batch}_{id}.pt 因 _id 后缀不匹配
    homo = {}
    for f in actor_files:
        m = re.match(r'actor_rein(\d+)\.pt', os.path.basename(f))
        if m:
            homo[int(m.group(1))] = f
    if not homo:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到同构 actor_rein{{batch}}.pt（该目录只有异构命名）")
    latest_batch = max(homo.keys())

    # 加载的版本号
    loaded_version = latest_batch  # 可手动指定，如 10

    # 单份 state_dict：test_worker 检测到后所有红机共享同一份权重
    model_state_dict = torch.load(homo[loaded_version], map_location='cpu')
    print(f"加载同构 actor：batch={loaded_version}（单份，全队共享）")

    rules_num, result, wins, loses, draws, perish_togethers = test_worker(
        model_state_dict, RULES_NUM, args, num_runs=N_EPISODES, vertices=vertices,
        save_record=True, temperature=TEMPERATURE)

    print(f"\n红方(神经网络) vs 蓝方(规则{rules_num})，共{N_EPISODES}场：")
    print(f"胜 {wins} / 负 {loses} / 平 {draws}，综合得分率 {result:.2%}，死光 {perish_togethers} 场")
