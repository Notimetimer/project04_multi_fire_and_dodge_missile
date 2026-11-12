import os, sys
from _context import *
from TrainAndTests.Combats.CombatMAPPO_parall_homo import *
from datetime import datetime

# 指定中断续训的目录。如果为 None，则正常开启新训练。
resume_target_dir = None
resume_target_dir = os.path.join(r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat",
    r"MAPPO_homo_SLA_1_blackboard_1-run-20261110-130531")
collape_recover={ # 是否是崩盘后恢复
            "collapsed": False,
            "best_actor_name": None,
            "actor_frozen_batchs": 5,
        }

USE_SUPERVISED_FIRE = 1 # [新增] 开火命中监督更新开关：仅用 fire_obs->hit_record 监督 fc_bern 头
ENM_CHOSEN_VISABLE = 1 # 是否能看到导弹的选择 1
PPO_WITH_BERN = 1 # 是否开启PPO的开火训练梯度
mission_name = f'MAPPO_homo_SLA_{USE_SUPERVISED_FIRE}_blackboard_{ENM_CHOSEN_VISABLE}'

# 超参数
dt_decide = 2
dt_maneuver = 2 # 0.2
action_cycle_multiplier = int(round(dt_decide / dt_maneuver))  # 2s 决策一次
MAX_DEC_STEPS = 1e6  # 按决策步定训练开支

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
        num_workers = 15,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        actor_lr = 1e-4,
        critic_lr = 5e-4,
        max_steps = MAX_DEC_STEPS*action_cycle_multiplier,  # 20e3
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
        MAX_HISTORY_SIZE = 50, # 150  # 300
        compete_old_rate = 0.2, # “复习”概率
        K_FACTOR = 16,  # 32 原先振荡太大了
        randomized_birth = 1,
        save_interval = 1, # 触发更新至少要经过多少批采样
        num_runs = 3, # 测试回合重复次数  3
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        # 场地边界：与测试脚本一致，统一从 init_states.json 加载方形顶点（不再传 None 走环境兜底）
        vertices = load_init_states()['vertices'],
        resume_dir=resume_target_dir, # 指定中断续训目录
        use_supervised_fire = USE_SUPERVISED_FIRE,
        supervised_fire_buffer_size = 1000, # [新增] 开火记录滚动缓冲容量（FIFO，像off-policy一样滚动更新）
        supervised_fire_batch_size = 128, # [新增] 每次监督更新从缓冲中采样的批大小
        ppo_with_bern = PPO_WITH_BERN, # [新增] 0 时 PPO 更新跳过开火头 fc_bern（log_prob 不计入 bern 项）
        enm_chosen_visable = ENM_CHOSEN_VISABLE, # 是否能看到队友的选择
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")