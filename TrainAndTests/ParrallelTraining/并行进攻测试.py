import sys
import os
import argparse
import numpy as np
import torch as th
from math import pi, cos, sin, atan2, sqrt
import time
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 从 TrainAndTests.Attacks.PPOAttack__Train 导入 args，但如果找不到，我们手动创建
try:
    from TrainAndTests.Attacks.PPOAttack__Train import args
except ImportError:
    # 如果串行训练脚本找不到，手动创建 args
    print("Warning: Could not import args from PPOAttack__Train. Manual args created.")
    args = argparse.Namespace(max_episode_len=120, R_cage=70e3, n_envs=1, max_steps=1)


from Envs.Tasks.AttackManeuverEnv import AttackTrainEnv, dt_maneuver
# [注意] 不再需要 ParallelEnv，但我们需要它的 PPOHybrid 和模型结构
from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Utilities.LocateDirAndAgents import get_latest_log_dir, load_actor_from_log

# [新增] 导入并行训练中定义的规则逻辑 (为简化，我们将逻辑直接内嵌到测试脚本中)
from Math_calculates.sub_of_angles import sub_of_radian
# ----------------------------------------------------------------------
# 规则逻辑辅助函数 (从并行训练脚本中提取，用于红方决策)
# ----------------------------------------------------------------------

def track_behavior(ego_height, delta_psi, speed_cmd=1.5*340):
    height_cmd = 7e3 - ego_height
    heading_cmd = delta_psi
    return np.array([height_cmd, heading_cmd, speed_cmd])

def left_crank_behavior(ego_height, delta_psi, speed_cmd = 1.1 * 340):
    height_cmd = 7e3 - ego_height
    heading_cmd = np.clip(delta_psi - pi / 4, -pi / 2, pi / 2)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def right_crank_behavior(ego_height, delta_psi, speed_cmd = 1.1 * 340):
    height_cmd = 7e3 - ego_height
    heading_cmd = np.clip(delta_psi + pi / 4, -pi / 2, pi / 2)
    return np.array([height_cmd, heading_cmd, speed_cmd])

def wander_behavior(speed_cmd = 300):
    alt_cmd = 3000 * np.random.uniform(-1, 1)
    heading_cmd = np.random.normal(0, 25 * pi / 180)
    return np.array([alt_cmd, heading_cmd, speed_cmd])

def back_in_cage(cmd, ego_pos_, ego_psi, R_cage):
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

def get_red_rule_decision(obs_flat, raw_info, args, use_wander=True):
    """红方决策：返回 (maneuver_vector, fire_boolean)"""
    
    # 状态还原 (AttackObs 已缩放)
    dist = obs_flat[3] * 10e3 
    cos_dpsi = obs_flat[0]
    sin_dpsi = obs_flat[1]
    enm_delta_psi = atan2(sin_dpsi, cos_dpsi)
    
    # 从 raw_info 获取绝对导航信息 (在串行环境中，我们需要手动传入)
    ego_pos_ = raw_info['pos']
    ego_psi = raw_info['psi']
    ammo = raw_info['ammo']
    
    # 决策逻辑
    if use_wander:
        cmd = wander_behavior()
    else:
        if dist > 40e3:
             cmd = track_behavior(ego_pos_[1], enm_delta_psi)
        elif dist > 10e3:
            if enm_delta_psi >= 0:
                cmd = left_crank_behavior(ego_pos_[1], enm_delta_psi)
            else:
                cmd = right_crank_behavior(ego_pos_[1], enm_delta_psi)
        else:
            cmd = track_behavior(ego_pos_[1], enm_delta_psi)
             
    # 强制回笼逻辑
    cmd = back_in_cage(cmd, ego_pos_, ego_psi, R_cage=args.R_cage)
    
    # 发射逻辑
    fire = False
    # obs_ATA = obs_flat[4] # ATA (未缩放，弧度)
    
    # if ammo > 0:
    #     if dist < 30e3 and abs(obs_ATA) < 30 * pi / 180:
    #          if np.random.rand() < 0.05: 
    #              fire = True
                 
    return cmd, fire
# ----------------------------------------------------------------------


dt_maneuver = 0.2 
hidden_dim = [128, 128, 128]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mission_name = 'Attack_Parallel' # 必须与训练脚本一致

# --- 1. 获取环境维度并初始化模型 ---

tmp_env = AttackTrainEnv(args)
tmp_env.reset()
b_action_dim = tmp_env.b_action_spaces[0].shape[0]
tmp_obs, _ = tmp_env.attack_obs('b')
state_dim = tmp_obs.shape[0]
del tmp_env

action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
action_dims_dict = {'cont': b_action_dim, 'cat': [], 'bern': 0}

actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

# 仅用于初始化 PPOHybrid 结构，推理时实际不使用 critic
critic_net = ValueNet(state_dim, hidden_dim).to(device)
agent = PPOHybrid(actor=actor_wrapper, critic=critic_net,
                  actor_lr=1e-4, critic_lr=5e-4, lmbda=0.95, epochs=10, eps=0.2, gamma=0.9, device=device)

# --- 2. 加载 checkpoint ---

pre_log_dir = os.path.join(project_root, "logs/attack")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
actor_path = None
if log_dir:
    actor_path = load_actor_from_log(log_dir, number=None, rein_prefix="actor_rein")
    if actor_path is None:
        actor_path = load_actor_from_log(log_dir, number=None, rein_prefix="actor_save")

if not actor_path:
    print(f"No actor checkpoint found in {log_dir}. Running with random weights.")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    try:
        actor_wrapper.load_state_dict(sd)
    except Exception:
        if 'actor' in sd:
            actor_wrapper.load_state_dict(sd['actor'])
        else:
            model_sd = actor_wrapper.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(filtered)
            actor_wrapper.load_state_dict(model_sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

try:
    # 保持单个串行环境，开启可视化
    env = AttackTrainEnv(args, tacview_show=1) 
    
    for i_episode in range(3): 
        print(f"--- Episode {i_episode + 1} ---")

        # --- 3. 环境初始化 ---
        init_distance = 90e3
        red_R_ = init_distance/2 
        blue_R_ = init_distance/2
        red_beta = pi 
        red_psi = 0 
        red_height = 8e3 
        red_N = red_R_*cos(red_beta)
        red_E = red_R_*sin(red_beta)
        blue_height = 8e3 

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_R_, blue_height, 0.0]),
                                    'psi': np.random.choice([pi/2, -pi/2])}
        
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        env.dt_maneuver = dt_maneuver
        
        done = False
        
        while not done:
            # --- 4. 获取观测和 Raw Info ---
            r_obs_n, r_obs_check = env.attack_obs('r')
            b_obs_n, b_obs_check = env.attack_obs('b')
            
            # (串行测试中不需要 memory/obs_check，但为了兼容性保留)
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()
            
            # --- 5. 红方 (Rule) 动作决策 ---
            # 提取 Red Raw Info (模拟 ParallelEnv Worker 提供的 'red_raw_info')
            red_raw_info = {
                'pos': env.RUAV.pos_,
                'psi': env.RUAV.psi,
                'ammo': env.RUAV.ammo
            }
            
            # 使用并行训练中定义的决策逻辑
            r_action_n_maneuver, r_fire = get_red_rule_decision(
                obs_flat=np.squeeze(r_obs_n), 
                raw_info=red_raw_info, 
                args=args,
                use_wander=True # 红方默认启用漫游
            )
            
            # --- 6. 蓝方 (RL Agent) 动作决策 ---
            # b_obs_n 已经是 (1, D) 形式
            b_action_n_exec, _, _, _  = agent.take_action(b_obs_n, explore=False)
            b_action_n_maneuver = b_action_n_exec['cont']
            
            # --- 7. 执行 Step (串行环境) ---
            # 在串行环境中，我们需要手动处理发射，因为 env.step 不包含发射逻辑
            if r_fire:
                # 必须从 battle6dof1v1_missile0919 导入此函数
                from Envs.battle6dof1v1_missile0919 import launch_missile_if_possible 
                launch_missile_if_possible(env, side='r') 
            # 蓝方发射逻辑（如果需要，也要在这里添加）
            # launch_missile_if_possible(env, side='b') # 蓝方发射默认关闭，除非 PPO 学习到发射
            
            env.step(r_action_n_maneuver, b_action_n_maneuver) 
            done, b_reward, _ = env.get_terminate_and_reward('b')
            
            env.render(t_bias=t_bias)
            time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    try:
        # 确保 Tacview 渲染结束
        if 'env' in locals() and hasattr(env, 'end_render'):
             env.end_render()
    except Exception:
        pass