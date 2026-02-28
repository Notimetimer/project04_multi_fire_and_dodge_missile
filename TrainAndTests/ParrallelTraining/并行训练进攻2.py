import argparse
import time
import sys
import os
import numpy as np
import torch
from datetime import datetime
from math import pi, sqrt, atan2, exp, cos, sin

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# debug0 排除环境的问题
from Envs.Tasks.AttackManeuverEnv import AttackTrainEnv, dt_maneuver
# from Envs.Tasks.AttackManeuverEnv_old import AttackTrainEnv, dt_maneuver
# 引入我们刚才写的并行 Wrapper
# debug1 排除ParallelEnv未修改部分的问题
# from Algorithms.ParallelEnv_old import ParallelPettingZooEnv
from Algorithms.ParallelEnv import ParallelPettingZooEnv
from Algorithms.PPOHybrid22 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
# debug2 排除HybridBuffer未修改部分的问题
# from Algorithms.HybridBuffer_old import HybridReplayBuffer
from Algorithms.HybridBuffer import HybridReplayBuffer # [新增]
from Visualize.tensorboard_visualize import TensorBoardLogger
from Math_calculates.ScaleLearningRate import scale_learning_rate
from Math_calculates.sub_of_angles import sub_of_radian

# --- 参数配置 ---
parser = argparse.ArgumentParser("UAV Parallel Training")
parser.add_argument("--max-episode-len", type=float, default=120,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")
parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量 (根据CPU核数设定)")
parser.add_argument("--max-steps", type=float, default=2e6, help="总训练步数")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 5
hidden_dim = [128, 128, 128]
gamma = 0.9
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy = 0.01
mission_name = 'Attack_Parallel_new'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ==============================================================================
#  Red Rule-Based Logic (提取自 Battle Class，用于计算对手动作)
# ==============================================================================

def track_behavior(ego_height, delta_psi, speed_cmd=1.5*340):
    """追踪行为"""
    height_cmd = 7e3 - ego_height
    heading_cmd = delta_psi
    return np.array([height_cmd, heading_cmd, speed_cmd])

def escape_behavior(ego_height, enm_delta_psi, warning, threat_delta_psi, speed_cmd=1.5 * 340):
    """逃逸行为"""
    height_cmd = 7e3 - ego_height
    if warning: 
        heading_cmd = np.clip(sub_of_radian(threat_delta_psi, pi), -pi / 2, pi / 2)
    else:
        heading_cmd = np.clip(sub_of_radian(enm_delta_psi, pi), -pi / 2, pi / 2)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def left_crank_behavior(ego_height, delta_psi, speed_cmd = 1.1 * 340):
    """左 Crank 行为"""
    height_cmd = 7e3 - ego_height
    heading_cmd = np.clip(delta_psi - pi / 4, -pi / 2, pi / 2)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def right_crank_behavior(ego_height, delta_psi, speed_cmd = 1.1 * 340):
    """右 Crank 行为"""
    height_cmd = 7e3 - ego_height
    heading_cmd = np.clip(delta_psi + pi / 4, -pi / 2, pi / 2)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def wander_behavior(speed_cmd = 300):
    """随机漫步行为"""
    alt_cmd = 3000 * np.random.uniform(-1, 1)
    heading_cmd = np.random.normal(0, 25 * pi / 180)
    return np.array([alt_cmd, heading_cmd, speed_cmd])

def back_in_cage(cmd, ego_pos_, ego_psi, R_cage=70e3):
    """回笼行为 (使用绝对坐标)"""
    height_cmd, heading_cmd, speed_cmd = cmd
    ego_height = ego_pos_[1]
    R_to_o00 = sqrt(ego_pos_[0] ** 2 + ego_pos_[2] ** 2)
    if ego_height > 13e3:
        height_cmd = -5000
    elif ego_height < 3e3:
        height_cmd = 5000
    if R_cage - R_to_o00 < 8e3:
        beta_of_o00 = atan2(-ego_pos_[2], -ego_pos_[0])
        heading_cmd = sub_of_radian(beta_of_o00, ego_psi)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def get_red_rule_decision(obs_flat, raw_info, env_index, args):
    """
    红方高层决策函数
    此函数替换了 AttackManeuverEnv.py 中被删除的 get_red_rule_action
    """
    
    # --- 1. 状态还原 ---
    # 距离 (dist) 在 AttackObs 中是第 4 个元素 (索引 3)，且被 10e3 缩放
    dist = obs_flat[3] * 10e3 
    
    # 相对方位角 (enm_delta_psi)
    cos_dpsi = obs_flat[0]
    sin_dpsi = obs_flat[1]
    enm_delta_psi = atan2(sin_dpsi, cos_dpsi)
    
    # 从 raw_info 获取绝对导航信息 (Worker 进程中提取)
    ego_pos_ = raw_info['pos']
    ego_psi = raw_info['psi']
    
    # 威胁感知 (AttackObs 屏蔽了 Threat，这里假定无威胁，简化处理)
    warning = False 
    threat_delta_psi = 0
    
    # --- 2. 策略选择 ---
    use_wander = False
    if env_index % 2 == 1:
        use_wander = True
        
    # --- 3. 核心决策逻辑 ---
    if use_wander:
        cmd = wander_behavior()
    else:
        # 简化版拦截逻辑 (与串行脚本中的 decision_rule 效果类似)
        if dist > 40e3:
             cmd = track_behavior(ego_pos_[1], enm_delta_psi)
        elif dist > 10e3:
            if enm_delta_psi >= 0:
                cmd = left_crank_behavior(ego_pos_[1], enm_delta_psi)
            else:
                cmd = right_crank_behavior(ego_pos_[1], enm_delta_psi)
        else:
            cmd = track_behavior(ego_pos_[1], enm_delta_psi)
             
    # 最高优先级：强制回笼逻辑
    cmd = back_in_cage(cmd, ego_pos_, ego_psi, R_cage=args.R_cage)
    
    # --- 4. 发射逻辑 ---
    fire = False
    # obs_ATA = obs_flat[4] # ATA (未缩放，弧度)
    
    # if raw_info['ammo'] > 0:
    #     # 发射条件：距离 < 30km, ATA < 30度 (简化逻辑)
    #     if dist < 30e3 and abs(obs_ATA) < 30 * pi / 180:
    #          if np.random.rand() < 0.05: 
    #              fire = True
                 
    return cmd, fire
# ==============================================================================


# --- 环境工厂函数 ---
def make_env():
    # 这里可以添加不同种子的设置
    def _thunk():
        env = AttackTrainEnv(args, tacview_show=0) 
        return env
    return _thunk

if __name__ == "__main__":
    # 1. 启动并行环境
    print(f"Starting {args.n_envs} parallel environments...")
    env_fns = [make_env() for _ in range(args.n_envs)]
    vec_env = ParallelPettingZooEnv(env_fns)
    
    # 2. 获取维度信息 (用一个临时环境)
    tmp_env = AttackTrainEnv(args)
    # ================= [新增这一行] =================
    tmp_env.reset()  # 必须先重置，生成飞机对象，否则无法获取观测
    # ===============================================
    b_action_dim = tmp_env.b_action_spaces[0].shape[0]
    # 获取观测维度
    tmp_obs, _ = tmp_env.attack_obs('b')
    state_dim = tmp_obs.shape[0]
    del tmp_env

    # 动作空间定义 (Hybrid)
    action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
    action_dims_dict = {'cont': b_action_dim, 'cat': [], 'bern': 0}

    # 3. 初始化 PPO Agent
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

    agent = PPOHybrid(
        actor=actor_wrapper, critic=critic_net, 
        actor_lr=actor_lr, critic_lr=critic_lr,
        lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma, 
        device=device, k_entropy=k_entropy, max_std=0.3
    )
    
    # 学习率缩放
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)
    agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    # 日志
    log_dir = os.path.join(project_root, f"logs/attack/{mission_name}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    # 保存 Meta 信息
    import json
    with open(os.path.join(log_dir, "actor.meta.json"), "w") as f:
        json.dump({k: list(v.shape) for k, v in agent.actor.state_dict().items()}, f)

    # 新增：按保存次数计数（从1开始）
    save_count = 0

    # 4. 训练主循环
    total_steps = 0
    # 每次 PPO 更新前收集的步数 = update_steps * n_envs
    # 例如 update_steps=256, n_envs=8 -> 每次更新基于 2048 个样本
    steps_per_rollout = 600 
    
    # [新增] 初始化高效 Buffer
    # 注意：actor_hidden_dim 暂设为 None，如果你用了 GRU 请填入维度
    replay_buffer = HybridReplayBuffer(
        n_envs=args.n_envs,
        buffer_size=steps_per_rollout,
        obs_dim=state_dim,   # 假设 obs_dim = state_dim (如果用了 attack_obs)
        state_dim=state_dim, # Critic 用的状态
        action_dims_dict=action_dims_dict,
        use_truncs=True, use_active_masks=True,
        device=device
    )

    # 保存间隔：每隔 steps_per_rollout * 10 步保存一次
    save_interval = steps_per_rollout * 10

    # Reset 全部环境
    obs_dict = vec_env.reset()
    b_obs = obs_dict['b_obs'] 

    # ================= [修正 1：必须初始化 b_state] =================
    # 从 reset 结果中获取 t=0 的状态
    if 'b_state' in obs_dict:
        b_state = obs_dict['b_state']
    else:
        # 如果 ParallelEnv 还没准备好 state，暂时用 obs 代替
        b_state = b_obs 
    # =============================================================

    infos = obs_dict['infos'] # 捕获初始 info (包含 red_raw_info)

    try:
        while total_steps < args.max_steps:
            # [修改] 每次循环前只需清空 Buffer 指针
            replay_buffer.clear()
            
            # --- 数据收集阶段 (Rollout) ---
            for _ in range(steps_per_rollout):
                # A. 蓝方动作 (神经网络批量推理)
                # agent.take_action 支持 (Batch, Dim) 输入
                # 返回 actions_exec (用于环境), actions_raw (用于训练)
                b_actions_exec, b_actions_raw, _, _ = agent.take_action(b_obs, explore=True)
                
                # B. 红方动作 (主进程循环处理不同策略)
                r_obs = obs_dict['r_obs']
                r_actions_list = []
                
                # *** 核心修改：并行化红方规则逻辑 ***
                for i in range(args.n_envs):
                    # 获取当前环境的 红方观测 和 Raw Info
                    curr_r_obs = r_obs[i]
                    curr_info = infos[i]
                    
                    # 提取 worker 发送的原始状态信息
                    curr_raw = curr_info.get('red_raw_info', {'pos': np.array([0,5000,0]), 'psi': 0, 'ammo': 0})
                    
                    # 调用规则逻辑，获取 (maneuver_vector, fire_boolean)
                    maneuver, fire = get_red_rule_decision(curr_r_obs, curr_raw, env_index=i, args=args)
                    
                    # r_action_list 的元素是 (maneuver_vector, fire_boolean)
                    r_actions_list.append((maneuver, fire))
                
                # C. 构造动作字典
                # 'r' 动作是 [(maneuver, fire), ...] 的列表
                actions_dispatch = {
                    'r': r_actions_list, 
                    'b': b_actions_exec['cont'] # (N_envs, 3)
                }
                
                # D. 并行步进
                results = vec_env.step(actions_dispatch)
                
                next_b_obs = results['b_obs']
                next_b_state = results['b_obs'] # 这是 S_{t+1}
                rewards = results['b_reward'] 
                dones = results['dones']

                # [新增] 获取新数据
                # 如果你的 Buffer 需要 state，现在可以直接拿
                b_masks = 1 # results['b_active_masks']
                truncs = 0 # results['truncs']
                
                # ================= [修正 2：存入 Buffer] =================
                replay_buffer.add(
                    obs=b_obs,       # 使用循环当前的 b_obs (S_t)
                    state=b_state,   # <--- 【关键】使用循环当前的 b_state (S_t)
                    action_dict=b_actions_raw,
                    reward=rewards,
                    done=dones,
                    next_state=next_b_state, # 存入 S_{t+1}
                    trunc=truncs, # 无法调换truncs和dones的位置，环境就不是这回事
                    active_mask=b_masks,
                )
                # ========================================================
                
                # 更新循环变量
                obs_dict = results
                infos = results['infos']
                b_obs = next_b_obs
                # t 变成 t+1
                # ================= [修正 3：更新状态] =================
                b_state = next_b_state   # <--- 更新 b_state 为 S_{t+1}
                # ====================================================
                total_steps += 1 # 这里 +1 代表所有环境都走了一步
                
            # --- 更新阶段 (Update) ---
            
            # [修改] 使用 PPOHybrid2 内置的并行数据预处理
            # 1. 计算 GAE (在展平前)
            # 2. 展平所有数据 (States, Actions, Rewards...)
            # 3. 返回包含 'advantages' 和 'td_targets' 的字典

            # [修改] 使用 Buffer 计算 GAE 并导出为 Dict
            # 传入 critic 网络用于价值评估
            transition_dict = replay_buffer.compute_estimates_and_flatten(
                critic_net=agent.critic, 
                gamma=gamma, 
                lmbda=lmbda
            )

            # 2. 执行 PPO 更新
            # 现在 update 内部会检测到 advantages 已存在，从而跳过重复计算
            # 并且 actions 已經是优化过的 Dict of Arrays 格式
            agent.update(transition_dict, mini_batch_size=64)
            
            # 3. 记录日志
            # 注意：transition_dict['rewards'] 已经被展平成 numpy array 了
            # 计算 avg_return：按照用户要求的归一化方式
            rewards_arr = np.array(transition_dict['rewards'])
            total_reward_all_envs = rewards_arr.sum()   # rollout 内所有 env、所有步的 reward 之和
            n_envs = args.n_envs
            # 将每步 reward 积分到回合尺度，且把 rollout 平均到每个回合/每个子环境上
            avg_return = (total_reward_all_envs / n_envs) * dt_maneuver * (round(args.max_episode_len / dt_maneuver) / steps_per_rollout)
            logger.add("train/1 episode_return", avg_return, total_steps)
            logger.add("train/actor_loss", agent.actor_loss, total_steps)
            logger.add("train/critic_loss", agent.critic_loss, total_steps)
            
            # 显示训练进度（当前步数 / 最大步数）及平均回报
            max_steps_val = float(args.max_steps) if getattr(args, "max_steps", None) is not None else 1.0
            progress = total_steps / max_steps_val if max_steps_val > 0 else 0.0
            print(f"Step {total_steps}/{int(max_steps_val)} ({progress:.2%}) - Reward={avg_return:.4f}")

            # 4. 保存模型
            # 按保存间隔保存（每隔 save_interval 步）
            if total_steps > 0 and total_steps % save_interval == 0:
                # 先自增保存计数
                save_count += 1
                fname = os.path.join(log_dir, f"actor_rein{save_count}.pt")
                torch.save(agent.actor.state_dict(), fname)

                # 更新/写入 elo_ratings.json：优先使用已有的同名条目，否则回退到上一次保存的分数，再无则默认1200
                elo_path = os.path.join(log_dir, "elo_ratings.json")
                try:
                    if os.path.exists(elo_path):
                        with open(elo_path, "r") as ef:
                            elo_dict = json.load(ef)
                    else:
                        elo_dict = {}
                except Exception:
                    elo_dict = {}

                key = f"actor_rein{save_count}"
                if key in elo_dict:
                    # 已有评估过程提前写入该保存点的 Elo，保留不变
                    pass
                else:
                    # 优先回退到上一次保存的 Elo（若存在）
                    prev_key = f"actor_rein{save_count-1}"
                    if prev_key in elo_dict:
                        elo_score = elo_dict[prev_key]
                    else:
                        # 若没有上一次记录，尝试取文件中最大的已有 actor_reinX 的分数
                        actor_keys = [k for k in elo_dict.keys() if k.startswith("actor_rein")]
                        if len(actor_keys) > 0:
                            # 取最大编号对应的分数
                            try:
                                nums = [(int(k.replace("actor_rein", "")), k) for k in actor_keys]
                                nums.sort()
                                elo_score = elo_dict[nums[-1][1]]
                            except Exception:
                                elo_score = 1200.0
                        else:
                            elo_score = 1200.0

                    elo_dict[key] = float(elo_score)
                    with open(elo_path, "w") as ef:
                        json.dump(elo_dict, ef, indent=2)
                
    finally:
        vec_env.close()
        logger.close()