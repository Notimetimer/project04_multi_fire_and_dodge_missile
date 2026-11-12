import os, sys, runpy
from _context import *
from TrainAndTests.Combats.CombatHanSAC_parall_homo import *
from datetime import datetime

# 指定中断续训的目录。如果为 None，则正常开启新训练。
resume_target_dir = None
# resume_target_dir = os.path.join(r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat",
#     r"HanSAC_homo_JPO-run-2026xxxx-xxxxxx")
mission_name = 'HanSAC_homo_JPO'

# 机数统一从 init_states.json 读取（与 MAPPO 链路同源），训练与周期测试共用同一组值
_init_cfg = load_init_states()
red_init_num = _init_cfg['red_init_num']
blue_init_num = _init_cfg['blue_init_num']

# 超参数（JPO 论文口径：γ=0.9, τ=0.05, lr=1e-4, batch=128, buffer 1e5）
dt_decide = 2
dt_maneuver = 2 # 0.2
action_cycle_multiplier = int(round(dt_decide / dt_maneuver))  # 2s 决策一次
MAX_DEC_STEPS = 1e6  # 按决策步定训练开支

# 仿真环境参数
max_episode_duration = 15 * 60  # 回合最长时间，单位s

if __name__ == '__main__':
    # 训练前自动按 init_states.json 刷新 env_obs_dims.json，省去人工重跑生成器
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '读取环境参数并存json.py'),
                   run_name='__main__')
    print('Hello')
    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    run_HanSAC_simulation(
        num_workers=15,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        lr=1e-4,
        max_steps_of_maneuver=MAX_DEC_STEPS * action_cycle_multiplier,
        gamma=0.9,
        tau=0.05,
        batch_size=128,
        buffer_size=int(1e5),
        min_terminal_samples=32,      # 末端门控：池中 terminated+truncated 帧达 32 才开始更新
        update_batch_points=16,       # 每攒够 16 个决策点触发一次 lane flush + 梯度更新
        points_per_update=1,          # 论文口径：1 决策点 : 1 次更新
        action_cycle_multiplier=action_cycle_multiplier,
        dt_move=0.07,
        max_episode_duration=max_episode_duration,
        R_cage=62.00e3,  # 场地半径，单位m
        dt_maneuver=dt_maneuver,
        init_elo_ratings={
            'Rule_0': 1200,
            'Rule_1': 1200,
            'Rule_2': 1200,
            'Rule_3': 1200,
        },
        self_play_type='PFSP_balanced',  # PFSP_balanced, PFSP_challenge, FSP, SP, None 表示非自博弈
        hist_agent_as_opponent=1,
        WARM_UP_STEPS=0e3,
        MAX_HISTORY_SIZE=50,
        compete_old_rate=0.2,  # "复习"概率
        K_FACTOR=16,
        randomized_birth=1,
        update_interval_of_batch=1,
        num_runs=3,  # 测试回合重复次数
        red_init_num=red_init_num,    # 与 env_obs_dims.json 的 ally 槽位对齐
        blue_init_num=blue_init_num,  # 与 env_obs_dims.json 的 enm 槽位对齐
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        # 场地边界：与 MAPPO 训练脚本一致，统一从 init_states.json 加载方形顶点
        vertices=load_init_states()['vertices'],
        resume_dir=resume_target_dir,  # 指定中断续训目录
        pause_on_finish=1,             # 达标后进入续训提示符（前台运行时手动续训/退出）
        fire_buffer_size=1000,         # 论文 |D_f|=1e3
        fire_batch_size=128,
        kI=0.8, kD=0.8,                # 论文期望熵系数 kI=kD=0.8
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")
