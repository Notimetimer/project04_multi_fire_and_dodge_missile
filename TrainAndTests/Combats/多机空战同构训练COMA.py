import os, sys, runpy
from _context import *
from TrainAndTests.Combats.CombatCOMA_parall_homo import *
from datetime import datetime

# 指定中断续训的目录。如果为 None，则正常开启新训练。
resume_target_dir = None
# resume_target_dir = os.path.join(r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat",
#     r"COMA_homo_JPO-run-20261119-120000")
mission_name = 'COMA_homo_JPO'

# 机数统一从 init_states.json 读取（与 HANSAC/MAPPO 链路同源），训练与周期测试共用同一组值
_init_cfg = load_init_states()
red_init_num = _init_cfg['red_init_num']
blue_init_num = _init_cfg['blue_init_num']

# 超参数（COMA 口径：on-policy，γ=0.9，TD(λ=0.8)，固定小熵系数 0.001）
# 节拍：按"回合"把联合决策链 push 进 buffer，攒够 rollout_min_decisions 个决策点后
# 做 update_epochs 轮梯度更新并清空（不做跨 rollout 复用）；目标 critic 每
# target_update_interval 个梯度步硬拷贝一次。
dt_decide = 2
dt_maneuver = 2  # 0.2
action_cycle_multiplier = int(round(dt_decide / dt_maneuver))  # 2s 决策一次
MAX_DEC_STEPS = 1e6  # 按决策步定训练开支
NUM_OF_TEST_POINTS = 50  # 测试点数

# 仿真环境参数
max_episode_duration = 15 * 60  # 回合最长时间，单位s

if __name__ == '__main__':
    # 自博弈总开关：True = 历史权重入池 + PFSP 自博弈；
    # False = 关闭自博弈，对手池只保留 init_elo_ratings 里的预设规则对手
    SELF_PLAY = 1
    # 训练前自动按 init_states.json 刷新 env_obs_dims.json，省去人工重跑生成器
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '读取环境参数并存json.py'),
                   run_name='__main__')
    print('Hello')
    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    run_COMA_simulation(
        num_workers=15,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        lr=3e-4,                      # on-policy 更新次数多，取 3e-4（COMA 论文 1e-4）
        max_steps_of_maneuver=MAX_DEC_STEPS * action_cycle_multiplier,
        gamma=0.9,                    # 论文 γ=0.9
        lam=0.8,                      # TD(λ)
        entropy_coef=0.001,           # 固定小熵系数（声明偏差，COMA 原版无熵正则）
        update_epochs=3,              # 每段 rollout 复用 3 个 epoch
        batch_size=512,               # minibatch 大小（critic/actor 共用）
        rollout_min_decisions=1024,   # 攒够 1024 个联合决策触发一次 update
        target_update_interval=200,   # 目标 critic 硬拷贝间隔（按梯度步）
        max_grad_norm=10.0,           # 梯度裁剪阈值
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
        self_play_type='PFSP_balanced',  # PFSP_balanced, PFSP_challenge, FSP, SP
        hist_agent_as_opponent=1 if SELF_PLAY else 0,  # 0：历史权重永不入池
        WARM_UP_STEPS=0e3,
        MAX_HISTORY_SIZE=50,
        compete_old_rate=0.2,  # "复习"概率
        K_FACTOR=16,
        randomized_birth=1,
        test_interval_dp=int(MAX_DEC_STEPS / NUM_OF_TEST_POINTS),  # 采集50个测试点
        num_runs=3,  # 测试回合重复次数
        red_init_num=red_init_num,    # 与 env_obs_dims.json 的 ally 槽位对齐
        blue_init_num=blue_init_num,  # 与 env_obs_dims.json 的 enm 槽位对齐
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        # 场地边界：与 MAPPO/HANSAC 训练脚本一致，统一从 init_states.json 加载方形顶点
        vertices=load_init_states()['vertices'],
        resume_dir=resume_target_dir,  # 指定中断续训目录
        pause_on_finish=1,             # 达标后进入续训提示符（前台运行时手动续训/退出）
        fire_buffer_size=1000,         # 论文 |D_f|=1e3
        fire_batch_size=128,
        fire_min_samples=100,
        seed=42,
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")
