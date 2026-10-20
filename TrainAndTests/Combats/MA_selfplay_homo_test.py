"同构自博弈测试：红方最新策略 vs 蓝方指定检查点，双方均为神经网络（无规则对手）"
"""
以 MAselfplay_with_homo_test.py 为蓝本改造，用于观察“最终策略自博弈”的两类隐患：
1) 终止判定：完全沿用原流程（循环上界 = round(max_episode_len/dt_maneuver)，胜负读 env.r_outcome），
   逐场输出诊断，标注 r_outcome 是否为 None（即超时/终局判定是否被触发）。
2) 红蓝换边：红蓝两侧都加载神经网络（红=最新批次，蓝=指定批次），逐场打印双方存活数，
   用于观察同一套判定流程在两侧是否对称、是否出现异常。

与原脚本的差异：
- 决策分支不再区分“红=神经网络 / 蓝=规则”，两侧全部走 ActorWrapper.get_action；
- 蓝机 id = 红机 id + 200（环境约定），按侧分别加载各自 state_dict；
- 新增终止判定的诊断输出（不修改任何判定/循环逻辑，只读结果）。
"""

import os
import sys
import numpy as np
import torch
import argparse
import glob
import time
import random
import re
from math import *

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
from Algorithms.ActorWithPointingNet import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field

use_tacview = 2

# 是否考虑视野限制
POMDP = 1


def creat_initial_state():
    # 飞机出生状态指定（与基线脚本保持一致，便于对比）
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


def load_latest_actor(log_dir):
    """从 log_dir 取最大 batch 号的同构 actor_rein{batch}.pt，返回 (batch, state_dict)。
    只匹配同构命名 actor_rein{batch}.pt；异构 actor_rein{batch}_{id}.pt 因带 _id 后缀不匹配。"""
    actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    homo = {}
    for f in actor_files:
        m = re.match(r'actor_rein(\d+)\.pt', os.path.basename(f))
        if m:
            homo[int(m.group(1))] = f
    if not homo:
        raise FileNotFoundError(f"未在 {log_dir} 找到同构 actor_rein{{batch}}.pt")
    latest_batch = max(homo.keys())
    return latest_batch, torch.load(homo[latest_batch], map_location='cpu')


def load_batch_actor(log_dir, batch):
    """从 log_dir 取指定 batch 号的同构 actor_rein{batch}.pt，返回 state_dict。"""
    path = os.path.join(log_dir, f"actor_rein{batch}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到指定检查点：{path}")
    return torch.load(path, map_location='cpu')


# ==========================================
# 自博弈测试 Worker：红方最新策略 vs 蓝方指定检查点（双方均神经网络）
# ==========================================
def selfplay_worker(red_state_dict, blue_state_dict,
                    env_args, device_name='cpu', num_runs=3,
                    vertices=None,
                    dt_move=0.07,
                    dt_maneuver=0.2,
                    action_cycle_multiplier=10,
                    min_alt_safe=2000,
                    red_init_num=2, blue_init_num=2,
                    red_init_ammo=4, blue_init_ammo=4,
                    temperature=0.3,  # 采样温度：两侧共用同一温度，保证对比公平
                    ):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device(device_name)
    test_env = ChooseStrategyEnv(env_args, tacview_show=use_tacview, vertices=vertices)
    test_env.dt_move = dt_move

    # 每架飞机一个独立 ActorWrapper：红机加载 red_state_dict，蓝机加载 blue_state_dict
    actor_dict = None

    wins = 0
    loses = 0
    draws = 0
    perish_togethers = 0
    judge_miss = 0          # r_outcome 为 None（判定未触发）的场次数
    judge_miss_sign = []    # 上述场次中，若按存活数判定，本应得到的 sign(red_alive-blue_alive)
    judge_anomaly = 0       # done=True 但 r_outcome 为空（异常路径）的场次数
    episode_times = []
    t_bias = 0

    try:
        for i_episode in range(num_runs):
            ep_start = time.time()
            print(f"\n===== 第{i_episode+1}场 自博弈（红=最新策略 / 蓝=指定检查点） =====")
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

            # 首场 reset 后按侧建 actor：红机共享 red_state_dict，蓝机共享 blue_state_dict
            if actor_dict is None:
                actor_dict = {}
                fixed_dim, enm_shape, ally_shape = load_obs_dims()
                red_ids = []
                blue_ids = []
                for uid in test_env.UAV_ids:
                    side = test_env.find_uav_with_id(uid).side
                    actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
                    )
                    actor.load_state_dict(red_state_dict if side == 'r' else blue_state_dict)
                    actor.eval()
                    actor_dict[uid] = actor
                    (red_ids if side == 'r' else blue_ids).append(uid)
                print(f"  红机 id={red_ids} 载入【最新策略】；蓝机 id={blue_ids} 载入【指定检查点】")

            # 各机上一决策步的动作缓存（按 id 保存，供执行阶段逐机取用）
            maneuver_cmds_all = {}
            target_index_all = {}

            done = False

            test_env.dt_move = dt_move
            test_env.dt_maneuver = dt_maneuver
            test_env.min_alt_safe = min_alt_safe

            # 环境运行一轮的情况（沿用基线：步数从 -1 起，决策点在能被 action_cycle_multiplier 整除处）
            steps_of_this_eps = -1
            loop_bound = round(env_args.max_episode_len / test_env.dt_maneuver)
            broke_at_done = False

            # 一轮仿真开始（判定逻辑原样保留：不修改循环上界，用于暴露超时判定是否可达）
            for count in range(loop_bound):
                # 回合结束判断
                steps_of_this_eps += 1
                if test_env.running == False or done:
                    broke_at_done = True
                    break

                # 获取观测信息（每帧每机只调一次 get_obs）
                agent_inputs_ = {}
                mask_inputs_ = {}
                rule_inputs_ = {}

                for id in test_env.UAV_ids:
                    agent_input, mask_input, rule_input = test_env.get_obs(id, pomdp=POMDP)
                    agent_inputs_[id] = agent_input
                    mask_inputs_[id] = mask_input
                    rule_inputs_[id] = rule_input

                # --- 双方智能体决策 ---
                # 到达决策点（每 action_cycle_multiplier 步）时，红蓝两侧全部走神经网络
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    for id in test_env.UAV_ids:
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
                        # 不考虑同时攻击多个目标
                        if bool(fire):
                            try:
                                test_env.fire_120(id, enm_idx, tabu=0)
                            except ValueError as e:
                                # 环境端对虚目标/越界索引抛错，这里仅打印不中断
                                print(f"[fire_120 警告] {e}")
                        # 雷达全程开
                        test_env.radar_action(id, 1)

                # 循环调用底层控制器，环境更新
                flight_cmds = {}
                for id in test_env.UAV_ids:
                    try:
                        flight_cmds[id] = test_env.maneuver_for_1_enm(
                            id, target_index_all[id], maneuver_cmds_all[id], rule_inputs_[id])
                    except ValueError as e:
                        print(f"[maneuver_for_1_enm 警告] {e}")
                        flight_cmds[id] = [0.0, 0.0, 0.0]
                test_env.step(flight_cmds.values())  # 环境统一更新

                # 终止与奖励结算：每机都要调一遍，保证各自的周期事件都被累积
                for id in test_env.UAV_ids:
                    done, reward_dict = test_env.combat_terminate_and_reward(
                        id=id, action_cycle_multiplier=action_cycle_multiplier)

                test_env.render(t_bias=t_bias)

            # ---- 诊断：结束原因、判定是否触发、双方存活数（只读，不改判定逻辑） ----
            end_t = test_env.t
            raw_outcome = test_env.r_outcome
            red_alive = sum(1 for u in test_env.UAVs if u.side == 'r' and not u.dead)
            blue_alive = sum(1 for u in test_env.UAVs if u.side == 'b' and not u.dead)

            if raw_outcome is not None:
                end_reason = "判定触发（击杀/超时）"
            elif broke_at_done:
                end_reason = "done=True 但 r_outcome 为空（异常路径）"
                judge_anomaly += 1
            else:
                end_reason = "循环到上界自然结束（r_outcome 为空，超时判定未触发）"

            outcome = raw_outcome if raw_outcome is not None else 0
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                loses += 1
            else:
                draws += 1

            if raw_outcome is None:
                judge_miss += 1
                judge_miss_sign.append(int(np.sign(red_alive - blue_alive)))

            result_str = "胜" if outcome > 0 else ("负" if outcome < 0 else "平")
            print(f"  t_end={end_t:.1f}s / 时限={env_args.max_episode_len}s，"
                  f"r_outcome={raw_outcome} → 红方{result_str}")
            print(f"  结束原因: {end_reason}")
            print(f"  结束存活: 红 {red_alive} / 蓝 {blue_alive}"
                  + (f"  （若按存活数判定应={int(np.sign(red_alive - blue_alive)):+d}）"
                     if raw_outcome is None else ""))

            perish_togethers += all(uav.dead for uav in test_env.UAVs) and test_env.t <= env_args.max_episode_len - 10

            # 仿真残余清理
            test_env.clear_render(t_bias=t_bias)
            t_bias += test_env.t

            ep_cost = time.time() - ep_start
            episode_times.append(ep_cost)
            print(f"  本场耗时：{ep_cost:.1f}s, 仿真倍率：{test_env.t/ep_cost:.1f}")

        # ---- 汇总 ----
        result = (wins + 0.5 * draws) / num_runs
        print("\n================ 自博弈汇总（红方=最新策略 视角） ================")
        print(f"共 {num_runs} 场：胜 {wins} / 负 {loses} / 平 {draws}，红方得分率 {result:.2%}")
        print(f"判定未触发（r_outcome 为空、被记为平）场次：{judge_miss}")
        if judge_miss:
            print(f"  这些场次若按存活数判定，本应结果：{judge_miss_sign}")
        print(f"判定异常（done=True 但 r_outcome 为空）场次：{judge_anomaly}")
        print(f"双杀场次（双方全灭且提前结束）：{perish_togethers}")
        print("提示：若“判定未触发”场次较多，说明循环上界停在时限帧导致超时判定分支不可达。")

        return result, wins, loses, draws, perish_togethers, judge_miss, judge_miss_sign, judge_anomaly

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
        result = (wins + 0.5 * draws) / max(len(episode_times), 1)
        return result, wins, loses, draws, perish_togethers, judge_miss, judge_miss_sign, judge_anomaly
    finally:
        test_env.end_render()


if __name__ == '__main__':
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation selfplay")
    args = parser.parse_args()

    # 测试配置
    N_EPISODES = 8      # 自博弈场次
    TEMPERATURE = 1.0   # 采样温度：<1 更贪婪，两侧共用

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长 54km，东西宽 100km 的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]

    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"

    # 红方：该目录下最新一批同构 actor_rein{batch}.pt
    RED_LOG_DIR = os.path.join(upper_log_dir,
                               "MAPPO_homo-run-20261019-185822"
                               )
    RED_BATCH = None
    # 蓝方：另指定的检查点目录与批次
    #   BLUE_BATCH = None  → 取 BLUE_LOG_DIR 下的最新批次（默认与红方同目录，即为“最新 vs 最新”）
    #   BLUE_BATCH = 10    → 取该目录下 actor_rein10.pt（改成你要对比的版本）
    BLUE_LOG_DIR = os.path.join(upper_log_dir,
                                "MAPPO_homo-run-20261019-185822"
                                )
    BLUE_BATCH = None
    
    if RED_BATCH is None:
        red_batch, red_sd = load_latest_actor(RED_LOG_DIR)
    else:
        red_batch = RED_BATCH
        red_sd = load_batch_actor(RED_LOG_DIR, red_batch)
    print(f"红方【指定检查点】：batch={red_batch}（{RED_LOG_DIR}）")

    if BLUE_BATCH is None:
        blue_batch, blue_sd = load_latest_actor(BLUE_LOG_DIR)
    else:
        blue_batch = BLUE_BATCH
        blue_sd = load_batch_actor(BLUE_LOG_DIR, blue_batch)
    print(f"蓝方【指定检查点】：batch={blue_batch}（{BLUE_LOG_DIR}）")

    result, wins, loses, draws, perish, judge_miss, judge_miss_sign, judge_anomaly = selfplay_worker(
        red_sd, blue_sd, args, num_runs=N_EPISODES, vertices=vertices,
        red_init_ammo=4, blue_init_ammo=4, temperature=TEMPERATURE)

    print(f"\n红方(最新 batch={red_batch}) vs 蓝方(batch={blue_batch})，共{N_EPISODES}场：")
    print(f"胜 {wins} / 负 {loses} / 平 {draws}，红方得分率 {result:.2%}")
    print(f"判定未触发(记为平) {judge_miss} 场，判定异常 {judge_anomaly} 场，死光 {perish} 场")
