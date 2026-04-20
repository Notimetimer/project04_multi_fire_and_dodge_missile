'''
当前做法的一个问题：
所有奖励是相加的，而且目标方向并不会随着姿态的变化而改变，当前训练的是什么？
是沿着一条安排好的轨迹动态跟踪，还是说只是照着指令里的速度、角速度和俯仰角去飞？
如果是后者，应该要区分“指令要你俯冲你俯冲撞地和指令没有要你俯冲你俯冲撞地的情况”
'''

use_tacview = 0

import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
from gym import spaces
import copy
import matplotlib.pyplot as plt
import json
import glob
import argparse
import time  # 确保引入 time 模块
from datetime import datetime

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from _context import *
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOHybrid23_0_distil2_one_step_KL import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from TrainAndTests.Controls.UPolicyWrapper import *

from TrainAndTests.Controls.FlightControl_Train_dual_a_out2 import track_env

import torch.multiprocessing as mp
import random
import traceback
import time

def worker_process(rank, pipe, args, state_dim, hidden_dim, action_dims_dict, action_bound, \
    device_worker, seed, dt_decide, dt_move, beta_ao, k_entropy={'cont':0.01, 'cat':0.01, 'bern':0.05}):
    try:
        worker_seed = seed + rank * 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        env = track_env(dt_move=dt_move, tacview_show=0, time_limit=args.max_episode_len)
        # dt_decide 已作为参数传入
        
        local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        from Algorithms.MLP_heads import ValueNet
        local_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)

        local_agent = PPOHybrid(
            actor=HybridActorWrapper(local_actor, action_dims_dict, action_bounds=action_bound, device=device_worker).to(device_worker),
            critic=local_dummy_critic,
            actor_lr=0, critic_lr=0,
            lmbda=0, eps=0, gamma=0, epochs=0,
            device=device_worker,
            k_entropy=k_entropy,
        )
        
        while True:
            cmd, packet = pipe.recv()
            if cmd == 'EXIT':
                break
            if cmd == 'RUN_EPISODE':
                actor_weights, warm_up = packet
                local_agent.actor.load_state_dict(actor_weights)
                
                init_height = np.random.uniform(4000, 10000)
                birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)}
                
                # 使用传入的 warm_up 调整难度
                height_req = np.clip(init_height + 1 * np.random.uniform(-1, 1) * 5000, 1000, 15000)
                psi_req = np.random.uniform(-pi, pi) # * warm_up
                v_req = np.random.uniform(0.5, 2.5) * 340  # 根本不可能跑到2.5Ma, 1.1 就封顶了，但是在训练的时候就要处理好这个数据.

                env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
                
                obs, obs_check = env.get_obs()
                done = False
                
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                episode_return = 0
                steps_run = 0
                ao_ema_episode = 0.0
                v_error_ema_episode = 0.0
                psi_error_ema_episode = 0.0
                theta_error_ema_episode = 0.0
                


                while not done:
                    # 目标会跑
                    height_req += np.random.randn() * 80 * dt_decide # 每秒动 80m
                    env.height_req = np.clip(height_req, 1000, 13000)
                    psi_req += np.random.randn() * 10 *pi/180 * dt_decide # 每秒动10°
                    env.psi_req = sub_of_radian(psi_req)
                    v_req += np.random.randn() * 3 * dt_decide # 速度目标也在动
                    env.v_req = np.clip(v_req, 0.5 * 340, 2.5 * 340)

                    obs, obs_check = env.get_obs()
                    action, u, _, _ = local_agent.take_action(obs, explore=True)
                    steps_run += 1
                    
                    next_obs, reward, done = env.step(action)
                    ao_ema_episode = beta_ao * ao_ema_episode + (1 - beta_ao) * (env.AO)
                    v_error_ema_episode = beta_ao * v_error_ema_episode + (1 - beta_ao) * abs(env.v_error)
                    psi_error_ema_episode = beta_ao * psi_error_ema_episode + (1 - beta_ao) * abs(env.psi_error)
                    theta_error_ema_episode = beta_ao * theta_error_ema_episode + (1 - beta_ao) * abs(env.theta_error)
                    
                    transition_dict['states'].append(obs)
                    transition_dict['actions'].append(u)
                    transition_dict['next_states'].append(next_obs)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['action_bounds'].append(action_bound)
                    
                    obs = next_obs
                    episode_return += reward
                
                metrics = {
                    'return': episode_return,
                    'steps': steps_run,
                    'fail': env.fail,
                    'crash': env.crash,
                    'stall': env.stall,
                    'break_up': env.break_up,
                    't': env.t,
                    'ao_ema': ao_ema_episode,
                    'v_error_ema': v_error_ema_episode,
                    'psi_error_ema': psi_error_ema_episode,
                    'theta_error_ema': theta_error_ema_episode,
                    'height_overshoot': env.height_overshoot,
                    'heading_overshoot': env.heading_overshoot
                }
                pipe.send({'trans': transition_dict, 'metrics': metrics})
                
    except Exception as e:
        tb = traceback.format_exc()
        try: pipe.send({'error': tb})
        except: pass

# 网络结构参数
state_dim = 7+8+4  # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
hidden_dim = [128, 128] # [128, 128]
action_dim = 4 # test
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
action_bound = np.array([[-1.1,1.1],[-1.1,1.1],[-1.1,1.1],[-0.2,1.2]])  # aileron, elevator, rudder, throttle
mission_name = 'FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr'

if __name__=='__main__':
    # dof = 3
    # 超参数
    actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
    critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
    max_steps = 50 * 65e4
    
    gamma = 0.95
    lmbda = 0.95
    epochs = 5  # 10
    eps = 0.2
    dt_decide = 0.16 # 0.15 对 0.01, 0.16 对 0.02
    dt_move = 0.02
    k_entropy={'cont':0.5, 'cat':0.0, 'bern':0.0} # 0.01

    # --- EMA 统计参数设定 ---
    # 学术设定：将 10s 设为 95% 权重的历史长度 (Cumulative Weight = 0.95)
    # 计算公式: beta^k = 0.05, 其中 k = 10s / dt_decide
    beta_ao_95_time = 10.0
    beta_ao = 0.05 ** (dt_decide / beta_ao_95_time) # 分配给旧数值的权重
    parser = argparse.ArgumentParser("UAV flight control training parallel")
    parser.add_argument("--num_workers", type=int, default=20, help="number of parallel workers")  # 10
    parser.add_argument("--max-episode-len", type=float, default=5*60, help="maximum episode time length") # 7分钟有些长
    args = parser.parse_args()

    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    # log_dir = "./logs"
    from datetime import datetime
    log_dir = os.path.join(project_root, "./logs/control", mission_name + "-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    start_time = datetime.now()
    print(f"Simulation start: {start_time.isoformat(sep=' ', timespec='seconds')}")
    mp.set_start_method('spawn', force=True)
    # 创建一个 dummy env 获取维度
    dummy_env = track_env(dt_move=dt_move, time_limit=args.max_episode_len)
    teacher_agent = UnifiedPolicyWrapper(dummy_env, dt_decide=dt_decide)

    action_dims_dict = {'cont': action_dim, 'cat': [], 'bern': 0}
    policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
    from Algorithms.MLP_heads import ValueNet
    critic = ValueNet(state_dim, hidden_dim).to(device)
    
    agent = PPOHybrid(actor, critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        
    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    def save_meta_once(path, state_dict):
        if os.path.exists(path):
            return
        meta = {k: list(v.shape) for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(meta, f)

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Visualize.tensorboard_visualize import TensorBoardLogger
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    
    # 启动多进程
    workers = []
    pipes = []
    worker_device = torch.device('cpu')  # Worker一般用CPU采样
    seed = 42

    print(f"Initializing {args.num_workers} training workers...")
    for i in range(args.num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(
            i, child_conn, args, state_dim, hidden_dim, 
            action_dims_dict, action_bound, worker_device, 
            seed, dt_decide, dt_move, beta_ao, k_entropy
        ))
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    try:
        rl_steps = 0
        i_episode = 0
        while rl_steps < max_steps:
            # 1. 深度拷贝当前 Actor 权重到 CPU
            current_weights = {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()}
            
            # 计算预热参数 (同步到 Worker)
            warm_up = 1 # np.clip(rl_steps / 30e4, 0, 1)

            # 2. 分发任务
            for rank in range(args.num_workers):
                pipes[rank].send(('RUN_EPISODE', (current_weights, warm_up)))
            
            # 3. 阻塞等待结果
            batch_results = []
            for rank in range(args.num_workers):
                try: 
                    res = pipes[rank].recv() # 阻塞等待
                except EOFError: 
                    print(f"[Error] Worker {rank} crashed silently.")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed.")
                    
                if isinstance(res, dict) and 'error' in res:
                    print(f"--- Master received error from Worker {rank}, aborting. ---")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed with error:\n{res['error']}")
                    
                batch_results.append(res)
            
            # 4. 汇总数据
            master_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
            batch_return_list = []
            batch_fail_cnt = 0
            batch_crash_cnt = 0
            batch_stall_cnt = 0
            batch_breakup_cnt = 0
            batch_steps_run = 0
            
            for res in batch_results:
                tr = res['trans']
                metrics = res['metrics']
                
                batch_return_list.append(metrics['return'])
                if metrics['fail']: batch_fail_cnt += 1
                if metrics.get('crash', 0): batch_crash_cnt += 1
                if metrics.get('stall', 0): batch_stall_cnt += 1
                if metrics.get('break_up', 0): batch_breakup_cnt += 1
                batch_steps_run += metrics['steps']
                
                for k in master_transition_dict:
                    master_transition_dict[k].extend(tr[k])
                    
            rl_steps += batch_steps_run
            i_episode += args.num_workers
            
            # 计算蒸馏参数 (Master 使用)
            alpha_distill = 0.0 # 1.0 * (1 - 0.8 * warm_up)
            distil_epochs = 0 # max(int(10 * (1 - 0.9 * warm_up)), 1)

            # 5. 模型更新
            agent.update(master_transition_dict, adv_normed=1, mini_batch_size=512)
            # agent.distil(master_transition_dict, teacher_agent=teacher_agent, epochs=distil_epochs, alpha=alpha_distill)
            
            # --- 保存模型 ---
            if (i_episode // args.num_workers) % 10 == 0:
                critic_path = os.path.join(log_dir, "critic.pt")
                th.save(agent.critic.state_dict(), critic_path)
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)
                
                # [新增] 记录当前模型的详细性能到 JSON
                # 由于 mean_return 等变量在下方定义，这里先提前计算一下基础指标
                summary_data = {
                    "actor_name": actor_name,
                    "rl_steps": int(rl_steps),
                    "i_episode": int(i_episode),
                    "mean_return": float(np.mean(batch_return_list)),
                    "survive_rate": float(1.0 - (batch_fail_cnt / args.num_workers)),
                    # 下面这些如果在这还没算，就用当时的全局记录
                    "max_std": float(agent.max_std),
                    "k_entropy": {k: float(v) for k, v in agent.k_entropy.items()}
                }
                summary_log_path = os.path.join(log_dir, "checkpoints_summary.json")
                with open(summary_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary_data, ensure_ascii=False) + "\n")

            # --- 日志和控制台输出 ---
            mean_return = np.mean(batch_return_list)
            survive_rate = 1.0 - (batch_fail_cnt / args.num_workers)
            print(f"Episodes {i_episode}, 进度: {rl_steps / max_steps:.3f}, batch_return: {mean_return:.3f}, survive_rate: {survive_rate:.2f}")

            logger.add("train/0 episode_return", mean_return, rl_steps)
            logger.add("train/0 survive", survive_rate, rl_steps)
            
            # 记录平均高度超调量
            mean_height_overshoot = np.mean([abs(res['metrics']['height_overshoot']) for res in batch_results])
            logger.add("train_plus/0 height_overshoot", mean_height_overshoot, rl_steps)
            
            # 记录失败分类比率
            logger.add("train_plus/fail_rate_crash", batch_crash_cnt / args.num_workers, rl_steps)
            logger.add("train_plus/fail_rate_stall", batch_stall_cnt / args.num_workers, rl_steps)
            logger.add("train_plus/fail_rate_breakup", batch_breakup_cnt / args.num_workers, rl_steps)

            # 记录平均航向超调量
            mean_heading_overshoot = np.mean([abs(res['metrics']['heading_overshoot']) for res in batch_results])
            logger.add("train_plus/0 heading_overshoot", mean_heading_overshoot * 180 / pi, rl_steps)

            # 记录平均 AO (回合结束时的 EMA 的算术平均，消除初始为0的偏差)
            mean_ao_batch = np.mean([res['metrics']['ao_ema'] / (1 - beta_ao ** max(1, res['metrics']['steps'])) for res in batch_results])
            logger.add("train_plus/0 avg AO", mean_ao_batch, rl_steps)

            # [重构] 平滑衰减逻辑：
            # 当 mean_ao_batch 从 20 减小到 5 时，线性地压制 max_std (0.2 -> 0.1) 和 k_entropy (0.5 -> 1e-5)
            # 因子 alpha: 5->0.0, 20->1.0
            alpha_decay = np.clip((mean_ao_batch - 5.0) / (20.0 - 5.0), 0.0, 1.0)
            
            # 平滑调整 max_std
            agent.max_std = 0.1 + (0.2 - 0.1) * alpha_decay
            agent.actor.net.clamp_log_std(agent.max_std)
            
            # 平滑调整熵系数 (假设初始 cont 熵为 0.5)
            agent.k_entropy['cont'] = 1e-5 + (0.5 - 1e-5) * alpha_decay
            
            # [新增] 动态调整学习率：从初始 lr 下降到 1/4
            current_actor_lr = actor_lr * (1/4 + (3/4) * alpha_decay)
            current_critic_lr = critic_lr * (1/4 + (3/4) * alpha_decay)
            agent.set_learning_rate(actor_lr=current_actor_lr, critic_lr=current_critic_lr)

            # logger 记录一下当前的干预值
            logger.add("train_plus/agent_max_std", agent.max_std, rl_steps)
            logger.add("train_plus/agent_k_entropy_cont", agent.k_entropy['cont'], rl_steps)
            logger.add("train_plus/actor_lr", current_actor_lr, rl_steps)
            logger.add("train_plus/critic_lr", current_critic_lr, rl_steps)

            # 记录平均速度误差 (回合 EMA 的算术平均)
            mean_v_error_batch = np.mean([res['metrics']['v_error_ema'] / (1 - beta_ao ** max(1, res['metrics']['steps'])) for res in batch_results])
            logger.add("train_plus/0 avg v_e", mean_v_error_batch, rl_steps)

            # 记录平均航向偏差 (回合 EMA 的算术平均)
            mean_psi_error_batch = np.mean([res['metrics']['psi_error_ema'] / (1 - beta_ao ** max(1, res['metrics']['steps'])) for res in batch_results])
            logger.add("train_plus/0 avg psi_e", mean_psi_error_batch, rl_steps)

            # 记录平均俯仰偏差 (回合 EMA 的算术平均)
            mean_theta_error_batch = np.mean([res['metrics']['theta_error_ema'] / (1 - beta_ao ** max(1, res['metrics']['steps'])) for res in batch_results])
            logger.add("train_plus/0 avg theta_e", mean_theta_error_batch, rl_steps)

            actor_grad_norm = model_grad_norm(agent.actor)
            critic_grad_norm = model_grad_norm(agent.critic)
            logger.add("train/1 actor_grad_norm", actor_grad_norm, rl_steps)
            logger.add("train/2 critic_grad_norm", critic_grad_norm, rl_steps)
            logger.add("train/3 actor_loss", agent.actor_loss, rl_steps)
            logger.add("train/4 critic_loss", agent.critic_loss, rl_steps)
            logger.add("train/5 entropy", agent.entropy_mean, rl_steps)

            if hasattr(agent.actor.net, 'log_std_cont'):
                current_std = torch.exp(agent.actor.net.log_std_cont).mean().item()
                logger.add("train/5 std", current_std, rl_steps)
            
            logger.add("train/6 ratio", agent.ratio_mean, rl_steps)
            logger.add("train/7 steps", i_episode, rl_steps)
            if hasattr(agent, 'dis_actor_loss') and agent.dis_actor_loss != 0:
                logger.add("train/8 distil_loss", agent.dis_actor_loss, rl_steps)

            logger.add("train_plus/warm_up", warm_up, rl_steps)
            logger.add("train_plus/distil_epochs", distil_epochs, rl_steps)
            logger.add("train_plus/alpha_distill", alpha_distill, rl_steps)

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        for pipe in pipes:
            try: pipe.send(('EXIT', None))
            except: pass
                
        for p in workers:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
            
        logger.close()


        print(f"日志已保存到：{logger.run_dir}")

        end_time = datetime.now()
        print(f"Simulation end: {end_time.isoformat(sep=' ', timespec='seconds')}")
        elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
        print(f"Simulation duration: {elapsed_hours:.4f} hours")

