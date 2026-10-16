import os, sys
from _context import *
from TrainAndTests.Combats.CombatMAPPO_parall_hetero import *
from datetime import datetime
from prepare_il_datas_hierarchical import run_rules

# 指定中断续训的目录。如果为 None，则正常开启新训练。
resume_target_dir = None
# resume_target_dir = os.path.join(r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat",
#     r"PPO0.3_flymask_v0h0-run-20260924-145600")
collape_recover={ # 是否是崩盘后恢复
            "collapsed": False,
            "best_actor_name": None,
            "actor_frozen_batchs": 5,
        }

mission_name = f'MAPPO'

# 超参数
dt_decide = 2 # 2 # 6
dt_maneuver=0.2
action_cycle_multiplier = int(round(dt_decide /dt_maneuver)) # 6s 决策一次
# 仿真环境参数
max_episode_duration = 15*60 # 回合最长时间，单位s
dt_action_cycle = dt_maneuver * action_cycle_multiplier
transition_dict_threshold = 0 # 8 * max_episode_duration//dt_action_cycle + 1  # 5*

if __name__=='__main__':
    print('Hello')
    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    run_MLP_simulation(
        collape_recover=collape_recover,
        num_workers = 10,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        actor_lr = 1e-4,
        critic_lr = 5e-4,
        max_steps = 20e6,
        gamma=0.97,
        lmbda=0.985,
        epochs=4,
        eps=0.2,
        k_entropy=0.001,
        action_cycle_multiplier=action_cycle_multiplier,
        dt_move = 0.07,
        max_episode_duration = max_episode_duration,
        R_cage = 62.00e3, # 55e3 # 场地半径，单位m,
        dt_maneuver = dt_maneuver,
        transition_dict_threshold=transition_dict_threshold,
        mini_batch_size_mixed=256, # 混合更新minibatch大小,
        init_elo_ratings = {
            'Rule_0': 1200, # debug
            "Rule_1": 1200,
            "Rule_2": 1200,
            'Rule_3': 1200,
            # 'Rule_4': 1200,
            # 'Rule_5': 1200,
            # 'Rule_6': 1200,
            },
        self_play_type = 'PFSP_balanced', # PFSP_balanced, PFSP_challenge, FSP, SP, None 表示非自博弈
        hist_agent_as_opponent = 1, # 奖励函数调试禁止自博弈
        WARM_UP_STEPS = 0e3, # 500e3, # 1e3 为debug
        ADMISSION_THRESHOLD = -1,  # 0.5,
        MAX_HISTORY_SIZE = 50, # 150  # 300
        compete_old_rate = 0.2, # “复习”概率
        K_FACTOR = 16,  # 32 原先振荡太大了
        randomized_birth = 1,
        save_interval = 1, # 触发更新至少要经过多少批采样
        num_runs = 3, # 测试回合重复次数  3
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        vertices = None,
        resume_dir=resume_target_dir, # 指定中断续训目录
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")