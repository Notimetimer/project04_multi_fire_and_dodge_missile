"""
不让部署的时候遇到超出训练规模的敌机范围
手段：直接修改enm_mask，首先把敌机按照:
[未被选中的由近到远，被选中的由近到远]顺序排序
”选中“是逐id做的，不仅是”目标被导弹占用“，还有"前面有队友选了这个敌机"
随后在mask_inputs中只保留存活的2个（也就是训练规模）的敌机，
其余将mask_input设为0，当作其它敌机死了，在有限的排序下执行决策

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

from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
from BasicRules_Homo import basic_rules
from Algorithms.ActorWithATTN import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field, load_init_states
from _context import load_obs_dims, load_init_states

USE_TACVIEW = 2 # 1 动态显示 2 静默写入 acmi_records

# 是否考虑视野限制
POMDP = 1

def creat_initial_state():
    # 飞机出生状态指定
    blue_height = 9000
    red_height = 9000
    red_psi = pi/2
    blue_psi = - red_psi
    red_N = 0
    red_E = - 55e3
    blue_N = red_N
    blue_E = -red_E
    if np.random.uniform(0, 1) < 0.5: # 随机换边（与 CombatMAPPO_parall_homo 初始状态一致）
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


# 初始数量/挂载：显式传参优先，否则统一从 init_states.json 读取
init_cfg = load_init_states()

# ==========================================
# 多智能体测试 Worker：红方神经网络 vs 蓝方规则
# ==========================================
def test_worker(model_state_dict, rules_num,
                env_args, device_name='cpu', num_runs=3,
                vertices=None, save_record=False, use_tacview=USE_TACVIEW,
                dt_move=0.07,
                dt_maneuver=0.2,
                action_cycle_multiplier=1,
                min_alt_safe=2000,
                red_init_num=init_cfg['red_init_num'], blue_init_num=init_cfg['blue_init_num'],
                red_init_ammo=init_cfg['red_init_ammo'], blue_init_ammo=init_cfg['blue_init_ammo'],
                red_alive_num=None, blue_alive_num=None,
                red_init_formation = 'wall', blue_init_formation = 'wall', # 初始队形
                temperature=1.0,  # 玻尔兹曼采样温度，仅本测试函数使用；训练 worker_process 不传此参数
                enm_chosen_override=None,  # enm_chosen 支路开关覆盖：None 跟随 mask_config.json，0/1 直接生效
                
                ):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    红方各机独立 ActorWrapper（共享同一份 model_state_dict），蓝方统一 basic_rules。
    胜负直接读环境端导出的 r_outcome（1 胜 / -1 负 / 0 平，红方视角）。
    """
    device = torch.device(device_name)

    # 观测中允许保留的敌机数固定为训练规模，直接读取 init_states.json，
    # 不随 test_worker 的 red/blue_init_num、red/blue_alive_num 输入改变
    initial_alive_r = init_cfg['red_alive_num']
    initial_alive_b = init_cfg['blue_alive_num']

    # 防止传入None
    if None in [red_init_num, blue_init_num, red_alive_num, blue_alive_num]:
        red_init_num = init_cfg['red_init_num']
        blue_init_num = init_cfg['blue_init_num']
        red_alive_num = init_cfg['red_alive_num']
        blue_alive_num = init_cfg['blue_alive_num']
    
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
    R_kill_count = 0
    B_kill_count = 0
    
    try:
        for i_episode in range(num_runs):
            ep_start = time.time()
            print(f"开始第{i_episode+1}场对抗")
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()
            test_env.reset(
                red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_num=red_init_num, blue_init_num=blue_init_num,
                red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo,
                red_alive_num=red_alive_num, blue_alive_num=blue_alive_num,
                # 红蓝各自独立的位置打乱种子（np.random 已在函数入口固定，序列可复现且场间不同）
                seed=(int(np.random.randint(0, 2**31 - 1)), int(np.random.randint(0, 2**31 - 1))),
                red_init_formation = red_init_formation, blue_init_formation = blue_init_formation, # 初始队形
                )
            # 初始存活的飞机数量
            episode_initial_alive_r = len(test_env.alive_r_uavs)
            episode_initial_alive_b = len(test_env.alive_b_uavs)

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
                        enm_chosen_override=enm_chosen_override,
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

            # 每场重置“本轮已选敌机”集合；队长=本队 id 最小者，它做决策时开启新一轮并清空记录
            red_enm_index_chosen = set()
            blue_enm_index_chosen = set()
            # red_leader_id = min(test_env.RUAV_ids) if test_env.RUAV_ids else None
            # blue_leader_id = min(test_env.BUAV_ids) if test_env.BUAV_ids else None

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
                    # 找出所有被导弹选中的敌机
                    allocatedIDs_r2b = test_env.allocatedIDs_r2b
                    allocatedIDs_b2r = test_env.allocatedIDs_b2r
                    for id in test_env.UAV_ids:
                        UAV = test_env.find_uav_with_id(id)
                        # TODO 找出所有没被导弹选中且存活的敌机
                        our_enm_chosen_ids = allocatedIDs_r2b if UAV.side == 'r' else allocatedIDs_b2r
                        enm_uavs = test_env.BUAVs if UAV.side == 'r' else test_env.RUAVs
                        enm_infos = rule_inputs_[id]['enm_infos']
                        
                        # TODO 对敌机进行排序，确保顺序为：[未被分配的存活敌机按距离递增排列，被分配的存活敌机按距离递增排列，死亡敌机按id递增排列]
                        # 之后抽取有限个敌机index并修改mask_inputs_，对红方看蓝方，数量为json中的blue_alive_num,否则为red_alive_num，把超出
                        # 的部分修改mask_inputs_到0
                        enm_indices = sorted(
                            range(len(enm_uavs)),
                            key=lambda i: (
                                2 if enm_uavs[i].dead else (1 if enm_uavs[i].id in our_enm_chosen_ids else 0),
                                0.0 if enm_uavs[i].dead else float(enm_infos[i]['enm_dist']),
                                enm_uavs[i].id,
                            )
                        )
                        visible_enm_num = initial_alive_b if UAV.side == 'r' else initial_alive_r
                        visible_enm_indices = set(enm_indices[:visible_enm_num])
                        for enm_index in range(len(mask_inputs_[id]['enm'])):
                            if enm_index not in visible_enm_indices:
                                mask_inputs_[id]['enm'][enm_index] = 0
                        
                        # 决策前把“本轮已被队友选中的敌机”写进本机观测；队长决策时先清空本轮记录
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

                        # 如果这架飞机存活，那么our_enm_chosen_ids会被改变，后续飞机将看到这个目标被选取
                        if not UAV.dead:
                            our_enm_chosen_ids.add(test_env.index2id(UAV.id, fire_target))
                        
                            
                        # 不考虑同时攻击多个目标
                        if fire_flag:
                            try:
                                test_env.fire_120(id, fire_target, tabu=0)
                            except ValueError as e:
                                # 环境端对虚目标/越界索引抛错，这里仅打印不中断（规则智能体会输出 null_token 索引）
                                print(f"[fire_120 警告] {e}")

                        # 雷达全程开
                        test_env.radar_action(id, 1)

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

            # 记录测试方击杀/死亡数
            test_env.update_entity_alives() # 更新存活表
            R_kill_count += episode_initial_alive_b - len(test_env.alive_b_uavs)
            B_kill_count += episode_initial_alive_r - len(test_env.alive_r_uavs)
            
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
        return {
            'rules_num': rules_num,
            'result': result,
            'wins': wins,
            'loses': loses,
            'draws': draws,
            'perish_togethers': perish_togethers,
            'kill_count': R_kill_count,
            'die_count': B_kill_count,
        }

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
        result = (wins + 0.5 * draws) / max(len(episode_times), 1)
        return {
            'rules_num': rules_num,
            'result': result,
            'wins': wins,
            'loses': loses,
            'draws': draws,
            'perish_togethers': perish_togethers,
            'kill_count': R_kill_count,
            'die_count': B_kill_count,
        }
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
    AMOUNT_FOR_ONE_SIDE = 6
    
    args.max_episode_len = 15*60
    args.R_cage = 62.00e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = load_init_states()['vertices']

    # 加载指定训练目录下最新一批 actor（异构：每架红机一份权重，打包为 {id: state_dict}）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir, 
                           "2APPO_homo_SLA_1_blackboard_1-run-20261123-083404"
                           )
    # 异构命名：actor_rein{batch}_{id}.pt
    actor_files_het = glob.glob(os.path.join(LOG_DIR, "actor_rein*_*.pt"))
    # 同构命名：actor_rein{batch}.pt，锚定结尾避免把 actor_rein10_3.pt 误认成 batch=10
    actor_files_homo = [f for f in glob.glob(os.path.join(LOG_DIR, "actor_rein*.pt"))
                        if re.match(r'actor_rein\d+\.pt$', os.path.basename(f))]
    if actor_files_het:
        # 文件名形如 actor_rein{batch}_{id}.pt，解析出 (batch, id) 后取最大 batch 的一批
        parsed = []
        for f in actor_files_het:
            m = re.match(r'actor_rein(\d+)_(\d+)\.pt$', os.path.basename(f))
            if m:
                parsed.append((int(m.group(1)), int(m.group(2)), f))
        latest_batch = max(b for b, _, _ in parsed)
    
        # 加载的版本号
        loaded_version = latest_batch
    
        model_state_dict = {
            uid: torch.load(path, map_location='cpu')
            for b, uid, path in parsed if b == latest_batch
        }
        print(f"加载最新 actor：batch={loaded_version}，红机 id={sorted(model_state_dict.keys())}")
    elif actor_files_homo:
        # 同构：单份权重，所有红机共享（test_worker 内会按单份 state_dict 分发）
        parsed = []
        for f in actor_files_homo:
            m = re.match(r'actor_rein(\d+)\.pt$', os.path.basename(f))
            if m:
                parsed.append((int(m.group(1)), f))
        loaded_version = max(b for b, _ in parsed)
        model_state_dict = torch.load(dict(parsed)[loaded_version], map_location='cpu')
        print(f"加载最新 actor：batch={loaded_version}（同构单份权重）")
    else:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到 actor_rein{{batch}}_{{id}}.pt 或 actor_rein{{batch}}.pt")
    
    stats = test_worker(
        model_state_dict, RULES_NUM, args, num_runs=N_EPISODES, vertices=vertices,
        dt_maneuver=1,
        red_init_num=AMOUNT_FOR_ONE_SIDE, blue_init_num=AMOUNT_FOR_ONE_SIDE,
        red_alive_num=AMOUNT_FOR_ONE_SIDE, blue_alive_num=AMOUNT_FOR_ONE_SIDE,
        save_record=True, temperature=TEMPERATURE)

    print(f"\n红方(神经网络) vs 蓝方(规则{stats['rules_num']})，共{N_EPISODES}场：")
    print(f"胜 {stats['wins']} / 负 {stats['loses']} / 平 {stats['draws']}，综合得分率 {stats['result']:.2%}，死光 {stats['perish_togethers']} 场")
