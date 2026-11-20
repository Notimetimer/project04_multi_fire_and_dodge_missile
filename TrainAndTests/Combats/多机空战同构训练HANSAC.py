import os, sys, runpy
from _context import *
from TrainAndTests.Combats.CombatHanSAC_parall_homo import *
from datetime import datetime

# 指定中断续训的目录。如果为 None，则正常开启新训练。
resume_target_dir = None
# resume_target_dir = os.path.join(r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat",
#     r"HanSAC_homo_JPO-run-20261118-202604")
mission_name = 'HanSAC_homo_JPO'

# 机数统一从 init_states.json 读取（与 MAPPO 链路同源），训练与周期测试共用同一组值
_init_cfg = load_init_states()
red_init_num = _init_cfg['red_init_num']
blue_init_num = _init_cfg['blue_init_num']

# 超参数（JPO 论文口径：γ=0.9，网络 lr/α lr=1e-4，buffer 4e4）
# 更新节奏：每 points_per_update=5 个决策点触发一次，每次做 update_rounds=3 次更新
# （≈1.7 决策点 : 1 次更新；1M 决策点 ≈ 60 万次总更新，其中 actor ≈ 30 万步）
# batch=1024、τ=0.005、actor_update_every=2、梯度裁剪 1.0（均为 SAC 内部超参）
dt_decide = 2
dt_maneuver = 2 # 0.2
action_cycle_multiplier = int(round(dt_decide / dt_maneuver))  # 2s 决策一次
MAX_DEC_STEPS = 1e6  # 按决策步定训练开支
NUM_OF_TEST_POINTS = 50 # 300 # 测试点数

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
    run_HanSAC_simulation(
        num_workers=15,  # 并行进程数，根据CPU核数调整，建议 10-20
        mission_name=mission_name,
        lr=1e-4,                      # 3e-5 → 1e-4：Q 学得太慢而 target 跟得太快会外漂，配合 τ↓ 一起修正
        alpha_lr=1e-4,                # 3e-4 → 1e-4：α 步长远大于网络 lr 会让温度一两个窗口就贴底
        alpha_min=0.02,               # 温度下界：α_D 曾塌到 0.003，熵正则实际失效
        max_steps_of_maneuver=MAX_DEC_STEPS * action_cycle_multiplier,
        gamma=0.85, # 0.9
        tau=0.005,                    # 0.05 → 0.005 → 0.001：目标网络软更新步更小，避免外漂
        # [A方案·2026-11-18] 0.001 → 0.005：火控改由 BCE 单独更新后，V_t 里的 p_fire 来自目标侧
        # fire_head，τ 太小会让目标侧长期滞后在线 BCE（τ=0.001 约 4600 次更新才跟到 99%，
        # τ=0.005 约 920 次）；取 SB3 默认 0.005，既跟得上又比 0.05 稳。
        batch_size=1024,              # 512 → 1024：降梯度噪声（SAC 内部超参，不改环境/奖励/评测）
        buffer_size=100000,  # 约容纳 80 局（2V2：每决策点 2 条转移）
        # [降振荡·2026-11-19] 40000 → 100000：扩大回放在时间上的覆盖，降低"只对最近十几分钟
        # 数据自举"的方差；40k 在 2V2 下约 20k 决策点就转满一圈（≈14.5 min 墙钟），
        # 与一局最长 15 min 同量级，末端奖励还没回传就被挤掉。1e5 与论文 D_BFM 口径一致。
        min_terminal_samples=32,      # 末端门控：池中 terminated+truncated 帧达 32 才开始更新
        points_per_update=5,          # 每攒够 5 个决策点触发一次 lane flush + 梯度更新
        update_rounds=3,              # 每次触发做 3 次梯度更新（≈1.7 决策点 : 1 次；1M ≈ 60 万次）
        actor_update_every=3,         # 5 → 2：策略长期落后于 Q 会让策略梯度基于已过拟合的 Q
        # [降振荡·2026-11-19] 2 → 3：延迟策略更新再放缓一步，抑制"actor 追着 Q 跑"的来回横跳；
        # 温度与 actor 同步降频，代价是策略收敛变慢（先稳后快，若仍振荡再加到 4）。
        fire_update_every=1,          # 开火 BCE 监督每次更新都做
        max_grad_norm=1.0,            # 5.0 → 1.0：actor/critic 独立裁剪，收紧单步幅度
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
        self_play_type='PFSP_challenge',  # PFSP_balanced, PFSP_challenge, FSP, SP
        hist_agent_as_opponent=1 if SELF_PLAY else 0,  # 0：历史权重永不入池
        WARM_UP_STEPS=0e3,
        MAX_HISTORY_SIZE=50,
        compete_old_rate=0.2,  # "复习"概率
        K_FACTOR=16,
        randomized_birth=1,
        update_interval_of_batch=1,
        test_interval_dp=int(MAX_DEC_STEPS / NUM_OF_TEST_POINTS),  # 采集50个测试点
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
        kI=0.4, kD=0.4,                # kD：机动目标熵占最大熵的比例（0.4；kI 已废弃）
    )
    end_time = datetime.now()
    print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    print(f"Simulation duration: {elapsed_hours:.4f} hours")
