'''
训练进攻策略:

红方闲庭信步
蓝方追赶红方
'''

'''
修改说明：
已将算法替换为 PPOHybrid2，仅使用 Continuous ('cont') 动作空间。
'''

import argparse
import time
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from math import pi, cos, sin, sqrt
from numpy.linalg import norm

# 路径设置
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 环境与工具
from Envs.Tasks.AttackManeuverEnv import *
from Algorithms.Rules import decision_rule
from Math_calculates.ScaleLearningRate import scale_learning_rate
from Visualize.tensorboard_visualize import TensorBoardLogger

# --- [修改 1] 引入 Hybrid PPO 相关模块 ---
from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet

use_tacview = 0  # 是否可视化

# matplotlib 设置
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')

parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=120, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=70e3, help="")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 5
max_steps = 65e4
hidden_dim = [128, 128, 128]
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
pre_train_rate = 0
k_entropy = 0.01
mission_name = 'Attack_Hybrid' # 修改任务名以便区分

env = AttackTrainEnv(args, tacview_show=use_tacview)
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

# 动作边界定义
# 注意：HybridActorWrapper 会使用这个 bound 进行 tanh -> 真实动作 的映射
action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])

state_dim = 8 + 7 + 2
action_dim = b_action_spaces[0].shape[0]

# --- [修改 2] 定义 Hybrid 动作空间字典 (仅使用 cont) ---
action_dims_dict = {
    'cont': action_dim,  # 连续动作维度
    'cat': [],           # 离散动作 (无)
    'bern': 0            # 伯努利动作 (无)
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    
    # --- [修改 3] 初始化 Agent (参考 RLIL_Combat_PFSP.py) ---
    # 1. 创建神经网络
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)

    # 2. Wrapper (在这里传入 action_bound)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

    # 3. PPOHybrid Agent
    agent = PPOHybrid(
        actor=actor_wrapper, 
        critic=critic_net, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device, 
        k_entropy=k_entropy,
        max_std=0.3 # PPOHybrid 默认 max_std
    )

    # 日志与保存路径
    from datetime import datetime
    logs_dir = os.path.join(project_root, "logs/attack")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    # 缩放学习率
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)
    agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    out_range_count = 0
    return_list = []
    win_list = []
    total_steps = 0
    i_episode = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    training_start_time = time.time()
    t_bias = 0

    try:
        # 强化学习训练循环
        while total_steps < int(max_steps * (1 - pre_train_rate)):
            i_episode += 1
            episode_return = 0
            
            # --- [修改 4] Transition Dict 初始化 ---
            # PPOHybrid 不需要 'action_bounds' 字段，因为 bounds 固定在 wrapper 中
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

            # 飞机出生状态指定
            red_R_ = random.uniform(30e3, 40e3)
            red_beta = random.uniform(0, 2 * pi)
            red_psi = random.uniform(0, 2 * pi)
            red_height = random.uniform(3e3, 10e3)
            red_N = red_R_ * cos(red_beta)
            red_E = red_R_ * sin(red_beta)
            blue_height = random.uniform(3e3, 10e3)

            DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                       'psi': red_psi}
            DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                        'psi': pi}
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                      red_init_ammo=0, blue_init_ammo=0)

            done = False
            r_action_list = []
            b_action_list = []
            
            # 环境运行一轮的情况
            for count in range(round(args.max_episode_len / dt_maneuver)):
                current_t = count * dt_maneuver
                if env.running == False or done:
                    break
                
                # 获取观测信息
                r_obs_n, r_obs_check = env.attack_obs('r')
                b_obs_n, b_obs_check = env.attack_obs('b')

                env.RUAV.obs_memory = r_obs_check.copy()
                env.BUAV.obs_memory = b_obs_check.copy()

                b_obs = np.squeeze(b_obs_n)
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)

                # 红方（规则）决策
                if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:
                    launch_missile_if_possible(env, side='r')
                    launch_missile_if_possible(env, side='b')

                r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                           enm_pos_=env.BUAV.pos_, distance=distance,
                                           ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                           o00=o00, R_cage=env.R_cage, wander=1)

                # --- [修改 5] 蓝方决策 (使用 PPOHybrid 接口) ---
                # take_action 返回: actions_exec(dict), actions_raw(dict), hidden_state, dist_check
                b_actions_exec, b_actions_raw, _, _ = agent.take_action(b_obs, explore=True)
                
                # 从 dict 中取出 continuous 动作给环境执行
                b_action_n = b_actions_exec['cont'] 

                r_action_list.append(r_action_n)
                b_action_list.append(b_action_n)

                # 环境步进
                env.step(r_action_n, b_action_n)
                done, b_reward, _ = env.get_terminate_and_reward('b')
                next_b_obs, _ = env.attack_obs('b')
                env.BUAV.act_memory = b_action_n.copy()
                total_steps += 1

                # --- [修改 6] 存储经验 ---
                transition_dict['states'].append(b_obs)
                # 注意：这里存储的是 actions_raw (包含 'cont' key 的字典)，PPOHybrid update 需要这种格式
                transition_dict['actions'].append(b_actions_raw) 
                transition_dict['next_states'].append(next_b_obs)
                transition_dict['rewards'].append(b_reward)
                transition_dict['dones'].append(done)
                # 不再需要存储 action_bounds，wrapper 内部已处理

                episode_return += b_reward * env.dt_maneuver

                # 可视化
                env.render(t_bias=t_bias)

            if env.lose == 1:
                out_range_count += 1
            return_list.append(episode_return)
            win_list.append(1 - env.lose)
            
            # --- [修改 7] 更新策略 ---
            # PPOHybrid.update 内部会自动处理 dict 形式的 action
            agent.update(transition_dict)

            env.clear_render(t_bias=t_bias)
            t_bias += env.t

            # --- 保存模型 ---
            os.makedirs(log_dir, exist_ok=True)
            critic_path = os.path.join(log_dir, "critic.pt")
            torch.save(agent.critic.state_dict(), critic_path)
            
            if i_episode % 10 == 0:
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                torch.save(agent.actor.state_dict(), actor_path)

            # 打印进度
            if (i_episode) >= 10:
                print(f"episode {i_episode}, 进度: {total_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]) :.3f}")
            else:
                print(f"episode {i_episode}, total_steps {total_steps}")

            # Tensorboard 记录
            logger.add("train/1 episode_return", episode_return, total_steps)
            logger.add("train/2 win", env.win, total_steps)

            # 获取 agent 内部指标
            actor_grad_norm = agent.actor_grad
            actor_pre_clip_grad = agent.pre_clip_actor_grad
            critic_grad_norm = agent.critic_grad
            critic_pre_clip_grad = agent.pre_clip_critic_grad

            logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
            logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
            logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
            logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
            logger.add("train/7 actor_loss", agent.actor_loss, total_steps)
            logger.add("train/8 critic_loss", agent.critic_loss, total_steps)
            logger.add("train/9 entropy", agent.entropy_mean, total_steps)
            logger.add("train/10 ratio", agent.ratio_mean, total_steps)

        # 训练结束
        training_end_time = time.time()
        elapsed = training_end_time - training_start_time
        from datetime import timedelta
        td = timedelta(seconds=elapsed)
        print(f"总训练时长: {td.days}天 {td.seconds//3600}小时 {(td.seconds//60)%60}分钟 {td.seconds%60}秒")

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")