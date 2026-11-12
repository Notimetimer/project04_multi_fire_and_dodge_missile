"""同构 MAPPO 逐回合可视化复测：直接复用 CombatMAPPO_parall_homo.worker_process 的回合循环。

与 VsBaseline_MA_with_homo_test.py 的区别：
1. 不在主进程用 test_worker 跑 0.2s 的精细测试，而是 spawn 一个 worker_process，
   按训练侧同一套 dt_maneuver / action_cycle_multiplier 逐回合下发 RUN_EPISODE；
2. 每回合结束 worker 都会 clear_render，并在下一回合重新 handshake，
   因此每局都会在 acmi_records/ 下生成独立文件（文件名统一带 _ep1/_ep2/_ep3 后缀）；
3. 只做可视化与对局统计，不落盘经验池，也不参与训练。
"""

import os
import sys
import glob
import re
import time
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from CombatMAPPO_parall_homo import worker_process, create_initial_state_worker
from _context import load_init_states

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    # 复测配置
    RULES_NUM = 3            # 蓝方规则编号
    N_EPISODES = 3           # 复测局数
    DT_MANEUVER = 2          # 与训练一致（多机空战同构训练MAPPO.py: dt_maneuver = 2）
    DT_MOVE = 0.07
    ACTION_CYCLE_MULTIPLIER = 1
    USE_TACVIEW = 2          # 1 动态显示，2 静默写入 acmi_records，0 关闭
    ENM_CHOSEN_VISABLE = 0   # 与训练 checkpoint 的观测口径一致
    PPO_WITH_BERN = 1
    LOADED_VERSION = None    # actor_rein 的 batch 号；None 表示取目录内最新
    SEED = 42

    args.max_episode_len = 15 * 60
    args.R_cage = 60.3e3

    # 场地边界与默认机数/挂载统一来自 init_states.json
    init_cfg = load_init_states()
    vertices = init_cfg['vertices']
    red_init_num = init_cfg['red_init_num']
    blue_init_num = init_cfg['blue_init_num']
    red_init_ammo = init_cfg['red_init_ammo']
    blue_init_ammo = init_cfg['blue_init_ammo']

    # 加载同构 actor（CombatMAPPO_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "2APPO_homo_SLA_1_blackboard_0-run-20261112-171248",  # TODO 改成实际训练目录
                           )
    actor_files = glob.glob(os.path.join(LOG_DIR, "actor_rein*.pt"))
    if not actor_files:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到 actor_rein*.pt")
    homo = {}
    for f in actor_files:
        m = re.match(r'actor_rein(\d+)\.pt', os.path.basename(f))
        if m:
            homo[int(m.group(1))] = f
    if not homo:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到同构 actor_rein{{batch}}.pt")
    loaded_version = max(homo.keys()) if LOADED_VERSION is None else LOADED_VERSION

    model_state_dict = torch.load(homo[loaded_version], map_location='cpu')
    print(f"加载同构 actor：batch={loaded_version}（单份，全队共享）")
    print(f"复测规模：红 {red_init_num} vs 蓝 {blue_init_num}，dt_maneuver={DT_MANEUVER}s")

    # 单个 worker 常驻，主进程按回合下发 RUN_EPISODE（与训练侧 worker_process 完全同一套回合循环）
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=worker_process, kwargs={
        'rank': 0,
        'pipe': child_conn,
        'args': args,
        'device_worker': torch.device('cpu'),
        'dt_maneuver': DT_MANEUVER,
        'seed': SEED,
        'dt_move': DT_MOVE,
        'pomdp': 1,
        'vertices': vertices,
        'use_tacview': USE_TACVIEW,
    })
    p.start()

    wins = loses = draws = 0
    total_sim_time = 0.0
    total_wall_time = 0.0
    for k in range(1, N_EPISODES + 1):
        rb, bb = create_initial_state_worker(randomized=1)
        settings = {
            'action_cycle_multiplier': ACTION_CYCLE_MULTIPLIER,
            'red_init_num': red_init_num, 'blue_init_num': blue_init_num,
            'red_init_ammo': red_init_ammo, 'blue_init_ammo': blue_init_ammo,
            'red_birth': rb, 'blue_birth': bb,
            # 红蓝各自独立的出生位置打乱种子，保证场间不同且可复现
            'red_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
            'blue_shuffle_seed': int(np.random.randint(0, 2**31 - 1)),
            'enm_chosen_visable': ENM_CHOSEN_VISABLE,
            'ppo_with_bern': PPO_WITH_BERN,
        }
        print(f"\n开始第{k}场对抗")
        t0 = time.time()
        parent_conn.send(('RUN_EPISODE',
                          (model_state_dict, (f'Rule_{RULES_NUM}', 'rule', RULES_NUM), settings)))
        res = parent_conn.recv()
        wall_time = time.time() - t0
        if isinstance(res, dict) and 'error' in res:
            raise RuntimeError(f"Worker 回合异常:\n{res['error']}")

        m = res['metrics']
        wins += int(m['win'])
        loses += int(m['lose'])
        draws += int(m['draw'])
        hits = sum(1 for rec in res.get('fire_hit_records', []) if rec.get('hit_record') == 1)
        # 仿真倍率：模拟时长(steps * dt_maneuver) / 实际墙钟耗时
        sim_time = m['steps'] * DT_MANEUVER
        total_sim_time += sim_time
        total_wall_time += wall_time
        print(f"第{k}场结束：steps={m['steps']}，"
              f"{'胜' if m['win'] else ('负' if m['lose'] else '平')}，"
              f"return={m['return']:.3f}，命中 {hits}，"
              f"仿真 {sim_time:.1f}s / 耗时 {wall_time:.2f}s = {sim_time / wall_time:.1f}x")

    parent_conn.send(('EXIT', None))
    p.join(timeout=10)

    print(f"\n红方(MAPPO) vs 蓝方(规则{RULES_NUM})，共{N_EPISODES}场："
          f"胜 {wins} / 负 {loses} / 平 {draws}")
    print(f"平均仿真倍率：{total_sim_time / total_wall_time:.2f}x "
          f"（总仿真 {total_sim_time:.1f}s / 总耗时 {total_wall_time:.2f}s）")
    print(f"acmi 文件位于：{os.path.join(project_root, 'acmi_records')}")
