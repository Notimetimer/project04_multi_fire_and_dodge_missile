"额外输出双杀率"
"""
多智能体基线测试：红方神经网络（ActorWrapper，采样决策） vs 蓝方规则（basic_rules）
以 BasicRules_Homo带Actor测试.py 为蓝本，流程向 BasicRules_Homo.py 靠拢：
测试不存经验池，每帧每机只调一次 get_obs；沿用 test_worker(model_state_dict, ...) 接口，
函数内把 state_dict load 到各红机独立的 ActorWrapper；主程序以随机初始化网络传入。
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
from math import *

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
from BasicRules_Homo import basic_rules
from Algorithms.ActorWithPointingNetClean import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field

use_tacview = 2 # 1 动态显示 2 静默写入 acmi_records

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
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                        'psi': red_psi
                        }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


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
                ):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    红方各机独立 ActorWrapper（共享同一份 model_state_dict），蓝方统一 basic_rules。
    胜负从红机 reward_dict['end'] 判读（end = outcome * 30，outcome 以 ego 视角给出）。
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
                )

            # 首场 reset 后为每架红机各建一个独立 ActorWrapper 并加载权重，按 id 索引
            if actor_dict is None:
                actor_dict = {}
                red_ids = [i for i in test_env.UAV_ids if test_env.find_uav_with_id(i).side == 'r']
                # 网络维度统一从 env_obs_dims.json 读取（由 读取环境参数并存json.py 生成）
                fixed_dim, enm_shape, ally_shape = load_obs_dims()
                for rid in red_ids:
                    actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
                    )
                    actor.load_state_dict(model_state_dict)
                    actor.eval()
                    actor_dict[rid] = actor

            # 各机上一决策步的动作缓存（按 id 保存，供执行阶段逐机取用）
            maneuver_cmds_all = {}
            target_index_all = {}

            done = False
            red_end_reward = 0 # 终局时红机 reward_dict['end']，>0 胜 / <0 负 / ==0 平

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
                                actor_out = actor_dict[id].get_action(agent_inputs_[id], mask_inputs_[id], actor_rule)
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
                    # 终局帧从红机视角记录结果奖励（end = outcome * 30）
                    if done and 'end' in reward_dict and test_env.find_uav_with_id(id).side == 'r':
                        red_end_reward = reward_dict['end']

                '''显示运行轨迹'''
                # 可视化
                test_env.render(t_bias=t_bias)

            # 结果统计（红方视角）：end>0 胜 / <0 负 / ==0 平；
            # 平局且在时限前结束视为双杀（双方同归于尽），超时按剩余机数判 0 也是平
            if red_end_reward > 0:
                wins += 1
            elif red_end_reward < 0:
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
    N_EPISODES = 10    # 测试场次

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]

    # 随机初始化的红方网络（验证权重加载通路；正式使用时替换为训练后的 state_dict）
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    random_actor = ActorWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device='cpu')
    model_state_dict = random_actor.state_dict()

    rules_num, result, wins, loses, draws, perish_togethers = test_worker(
        model_state_dict, RULES_NUM, args, num_runs=N_EPISODES, vertices=vertices, save_record=True)

    print(f"\n红方(神经网络) vs 蓝方(规则{rules_num})，共{N_EPISODES}场：")
    print(f"胜 {wins} / 负 {loses} / 平 {draws}，综合得分率 {result:.2%}，死光 {perish_togethers} 场")
