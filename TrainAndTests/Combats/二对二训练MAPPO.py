import os, sys
# from CombatPPOWithIL3_parallel_hierarch_Classic import *
from CombatPPOWithIL4_parallel_hierarch import *
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
# 读取 mask_config.json 中的 ver/hor 开关状态 (相对路径)
mask_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Algorithms', 'mask_config.json')
with open(mask_config_path, 'r', encoding='utf-8') as f:
    _mask_cfg = json.load(f)
    ver = _mask_cfg.get('ver', 0)
    hor = _mask_cfg.get('hor', 0)


mission_name = f'MAPPO0.3_flymask_v{ver}h{hor}'

# 超参数
actor_lr = 1e-4 # 4 1e-4
critic_lr = actor_lr * 5 # * 5
IL_epoches= 0
max_steps = 20e6 # 1320e4
hidden_dim = [128, 128, 128]
gamma = 0.97 # 0.995
lmbda = 0.985 # 0.995
epochs = 4 # 10
eps = 0.2 # 0.2
k_entropy={'cont':0.01, 'cat':0.008, 'bern': 0.003} # cat:0.005, bern:0.001 是常数熵系数几乎完美的设定值。
alpha_il = 0.0  # 设置为0就是纯强化学习
il_batch_size=128 # 模仿学习minibatch大小
il_buffer_max_size= 5e3 # il_batch_size 2e4
mini_batch_size_mixed = 256 # 混合更新minibatch大小  64
beta_mixed = 1.0
label_smoothing=0.3 # 0.2 # 0.3 改为 1-0.4，而p1=0.4对应3.4附近的策略熵
label_smoothing_mixed=0.01
dt_decide = 2 # 2 # 6
dt_maneuver=0.2
action_cycle_multiplier = int(round(dt_decide /dt_maneuver)) # 6s 决策一次
trigger0 = 50e3  #  / 10
trigger_delta = 50e3  #  / 10
weight_reward_0 = np.array([1,0,0]) # 1,1,1 引导奖励很难说该不该有
IL_rule = 2 # 3 # 初始模仿对象
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 仿真环境参数
no_crash = 1 # 是否开启环境级别的防撞地系统
dt_move = 0.07 # 0.05 # 0.1 # 0.04 # 动力学解算步长, dt_maneuver=0.2 这是常数，不许改
max_episode_duration = 15*60 # 回合最长时间，单位s
R_cage= 62.00e3 # 55e3 # 场地半径，单位m
dt_action_cycle = dt_maneuver * action_cycle_multiplier
transition_dict_threshold = 8 * max_episode_duration//dt_action_cycle + 1  # 5*

require_new_IL_data = 0 # 是否需要现场产生示范数据

# # 现场产生奖励函数一致的示范数据
if require_new_IL_data:
    run_rules(gamma=gamma, weight_reward=weight_reward_0, action_cycle_multiplier=action_cycle_multiplier, current_rule=IL_rule)

# 加载数据
original_il_transition_dict, transition_dict = None, None

if __name__=='__main__':
    print('Hello')
    
    
    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    run_MLP_simulation(
        k_nonlinear=0.0,
        collape_recover=collape_recover,
        num_workers=15,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        IL_epoches=IL_epoches,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        k_entropy=k_entropy,
        alpha_il=alpha_il,
        il_batch_size=il_batch_size,
        il_buffer_max_size=il_buffer_max_size,
        mini_batch_size_mixed=mini_batch_size_mixed,
        beta_mixed=beta_mixed,
        label_smoothing=label_smoothing,
        label_smoothing_mixed=label_smoothing_mixed,
        action_cycle_multiplier=action_cycle_multiplier,
        trigger0=trigger0,
        trigger_delta=trigger_delta,
        weight_reward_0=weight_reward_0,
        IL_rule=IL_rule,
        no_crash=no_crash,
        dt_move=dt_move,
        max_episode_duration=max_episode_duration,
        R_cage=R_cage,
        dt_maneuver=dt_maneuver,
        transition_dict_threshold=transition_dict_threshold,
        should_kick=0, # False,  # 是否踢走不合规的对手
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
        use_sil = 0,
        p_factor = 0.23,
        WARM_UP_STEPS = 0e3, # 500e3, # 1e3 为debug
        ADMISSION_THRESHOLD = -1,  # 0.5,
        MAX_HISTORY_SIZE = 50, # 150  # 300
        compete_old_rate = 0.2, # “复习”概率
        K_FACTOR = 16,  # 32 原先振荡太大了
        randomized_birth = 1,
        save_interval = 1, # 触发更新至少要经过多少批采样
        opp_greedy_rate = 0, # 对手贪婪率
        num_runs = 3, # 测试回合重复次数
        device = device,
        R_cage_range = (R_cage, R_cage), # 固定场地大小
        vertices = None,
        resume_dir=resume_target_dir, # 指定中断续训目录
        init_il_data = original_il_transition_dict, # 传入模仿数据集
        POMDP=0,
        adj_r_w=0, # 奖励函数权重可调
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")