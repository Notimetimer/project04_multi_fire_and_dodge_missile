'''
通信流程（简化）

ParallelPettingZooEnv 创建 pipes + processes，把 CloudpickleWrapper(env_fn) 传入子进程；
子进程的 worker 从 wrapper 中恢复 env 并等待命令；
主进程调用 vec_env.step(...) 或 reset()，通过 remotes 发送 ('step', data)；
worker 执行环境逻辑，send 回结果；主进程收集并返回批量结果。
'''

import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time
import multiprocessing as mp
import cloudpickle
from datetime import datetime
from math import pi

def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))
cur_dir = get_current_file_dir()
project_root = os.path.dirname(os.path.dirname(cur_dir))
sys.path.append(project_root)

# 引入必要的模块
from Envs.Tasks.ChooseStrategyEnv2 import *
from Algorithms.PPOHybrid22 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Algorithms.HybridBuffer import HybridReplayBuffer  # 确保你有这个文件
from Visualize.tensorboard_visualize import TensorBoardLogger
from BasicRules import *



# ==============================================================================
# Part 1: ParallelEnv 定义 (Worker & Manager)
# ==============================================================================
# （子进程函数）
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    # 获取动作周期倍率 (Frame Skip)
    # 假设 BasicRules 或 env 中有定义，默认为 1
    try:
        cycle_mult = action_cycle_multiplier
    except NameError:
        cycle_mult = 1
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # data: {'r': r_action, 'b': b_action}
                r_action_label = data['r']
                b_action_exec = data['b'] # 这是一个字典 {'cat': [idx], 'bern': [val]}
                
                # 解析蓝方动作
                b_action_label = b_action_exec['cat'] # 已经是标量了
                b_fire = b_action_exec['bern']
                
                # 触发开火
                b_m_id = None
                if b_fire:
                    b_m_id = launch_missile_immediately(env, 'b')
                
                # 红方开火由 Rule 决定，这里暂且假设 r_action_label 只包含机动
                # 如果 Rule 返回了 (maneuver, fire)，需要在主进程拆解，或者这里处理
                # 这里假设传入的 r_action_label 仅为机动索引
                
                r_maneuver = env.maneuver14(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14(env.BUAV, b_action_label)
                
                step_reward = 0
                done = False
                win_flag = False
                lose_flag = False
                draw_flag = False
                
                # --- Frame Skipping Loop (物理步进) ---
                for _ in range(cycle_mult):
                    env.step(r_maneuver, b_maneuver)
                    
                    # 累计奖励 (每一步物理步都检查)
                    # 注意：reward_fn=1 是全局状态, 这里我们需要 combat_terminate_and_reward
                    is_done, r_val, r_assist = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None)
                    step_reward += (r_val - r_assist) # 累加净奖励
                    
                    if is_done:
                        done = True
                        # 记录胜负状态用于主进程统计
                        if env.win: win_flag = True
                        elif env.lose: lose_flag = True
                        else: draw_flag = True # 假设非胜非负即为平/超时
                        break
                
                # --- 获取 Next State / Obs ---
                # 1. 局部观测 (Actor)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                
                # 2. 全局状态 (Critic) - 注意这里是 Next State
                b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                
                # 3. 红方观测 (用于主进程计算 Rule)
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                
                # 4. 计算 Masks
                # 只有当蓝方存活时，Mask=1；死了 Mask=0
                b_active_mask = 1.0 if not env.BUAV.dead else 0.0
                
                # 5. Info
                info = {
                    'win': env.win,
                    'lose': env.lose,
                    'r_check_obs': r_check_obs # 传回主进程供 Rule 使用
                }
                
                if done:
                    # 如果结束，自动 Reset (PettingZoo 风格)
                    # 重新生成随机初始条件
                    # 这里简化处理：直接调用 reset，如果需要随机化需要在 worker 内部实现 create_initial_state
                    # 为保持并行训练的多样性，建议 env.reset() 内部包含随机性
                    red_psi = -pi/2
                    blue_psi = pi/2
                    red_N = 0
                    red_E = 45e3
                    blue_N = red_N
                    blue_E = -red_E
                    # 添加一些随机扰动
                    noise_pos = np.random.uniform(-5000, 5000, 3)
                    
                    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, 9000, red_E]), 'psi': red_psi}
                    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, 9000, blue_E]), 'psi': blue_psi}
                    
                    env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                             red_init_ammo=6, blue_init_ammo=6)
                    
                    # Reset 后重新获取初始 Obs
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                    b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                    b_active_mask = 1.0
                    
                    info['r_check_obs'] = r_check_obs # 更新 info
                
                remote.send({
                    'b_obs': b_obs,
                    'b_state': b_state_global,
                    'r_check_obs': r_check_obs, # 可以放在 info 里，也可以单独传
                    'b_reward': step_reward,
                    'dones': done,
                    'b_active_masks': b_active_mask,
                    'truncs': 0.0, # 暂时不处理截断
                    'infos': info
                })

            elif cmd == 'reset':
                # 初始化 Reset
                env.reset(red_init_ammo=6, blue_init_ammo=6)
                
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                
                remote.send({
                    'b_obs': b_obs,
                    'b_state': b_state_global,
                    'r_check_obs': r_check_obs,
                    'b_active_masks': 1.0,
                    'infos': {'r_check_obs': r_check_obs}
                })

            elif cmd == 'close':
                remote.close()
                break
    except KeyboardInterrupt:
        print('Worker KeyboardInterrupt')
    finally:
        remote.close()

# 实现 getstate 和 setstate，用 cloudpickle 将可调用对象（env factory）序列化为字节流。
# 解决 multiprocessing 在传递复杂可调用对象（如闭包或局部定义的函数/类）时的 pickle 问题，使 worker 能在子进程中重建 env
class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

# 进程管理器
# 在主进程中创建若干 mp.Pipe() 对和 mp.Process（target=worker），
# 每个 process 接收一个 CloudpickleWrapper 包裹的 env_fn
# 保存 remotes（主端管道）并在 step/reset 时把命令与数据发给各子进程，
# 收集各子进程返回的结果并做批量拼接/打包返回（_stack_results）。
# 提供 close() 清理子进程并结束管道。
class ParallelPettingZooEnv:
    def __init__(self, env_fns):
        self.closed = False
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_dict):
        # actions_dict: {'r': [act1, ...], 'b': [{'cat':..., 'bern':...}, ...]}
        for i, remote in enumerate(self.remotes):
            b_act_i = {k: v[i] for k, v in actions_dict['b'].items()} # 拆解 batch action
            worker_action = {
                'r': actions_dict['r'][i],
                'b': b_act_i
            }
            remote.send(('step', worker_action))

        results = [remote.recv() for remote in self.remotes]
        
        return self._stack_results(results)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return self._stack_results(results)
    
    def _stack_results(self, results):
        return {
            'b_obs': np.stack([r['b_obs'] for r in results]),
            'b_state': np.stack([r['b_state'] for r in results]), # Next State if step, Init State if reset
            'r_check_obs': np.stack([r['r_check_obs'] for r in results]),
            'b_reward': np.stack([r.get('b_reward', 0) for r in results]),
            'dones': np.stack([r.get('dones', False) for r in results]),
            'b_active_masks': np.stack([r['b_active_masks'] for r in results]).reshape(-1), # Buffer usually expects (N,) or (N,1)
            'truncs': np.stack([r.get('truncs', 0) for r in results]).reshape(-1),
            'infos': [r.get('infos', {}) for r in results]
        }

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except: pass
        for p in self.ps:
            p.join()
        self.closed = True

# ==============================================================================
# Part 2: 辅助函数
# ==============================================================================

def load_il_and_transitions(folder, il_name, rl_name):
    if folder is None: folder = os.getcwd()
    il_path = os.path.join(folder, il_name)
    il = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
        print(f"Loaded IL data from: {il_path}")
    else:
        print(f"File NOT found: {il_path}")
    return il

def restructure_actions(actions_data):
    if isinstance(actions_data, dict): return actions_data
    if isinstance(actions_data, list) and len(actions_data) > 0:
        new_actions = {'cat': [], 'bern': []}
        for item in actions_data:
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item()
            if isinstance(act, dict):
                if 'fly' in act: new_actions['cat'].append(act['fly'])
                if 'fire' in act: new_actions['bern'].append(act['fire'])
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                 new_actions['cat'].append(act[0])
                 new_actions['bern'].append(act[1])
        cat_arr = np.array(new_actions['cat'], dtype=np.int64)
        if cat_arr.ndim == 1: cat_arr = cat_arr.reshape(-1, 1)
        bern_arr = np.array(new_actions['bern'], dtype=np.float32)
        if bern_arr.ndim == 1: bern_arr = bern_arr.reshape(-1, 1)
        return {'cat': cat_arr, 'bern': bern_arr}
    return actions_data

def save_meta_once(path, state_dict):
    if os.path.exists(path): return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f: json.dump(meta, f)

# ==============================================================================
# Part 3: 主逻辑
# ==============================================================================

# 参数配置
parser = argparse.ArgumentParser("UAV swarm confrontation Parallel")
parser.add_argument("--max-episode-len", type=float, default=10*60)
parser.add_argument("--R-cage", type=float, default=55e3)
parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments") # 并行数量
parser.add_argument("--max-steps", type=float, default=2e6)
args = parser.parse_args()

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 5
IL_epoches = 0
hidden_dim = [128, 128, 128]
gamma = 0.95
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy = 0.05
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建一个临时环境获取维度
tmp_env = ChooseStrategyEnv(args)
state_dim = tmp_env.obs_dim
action_dims_dict = {'cont': 0, 'cat': tmp_env.fly_act_dim, 'bern': tmp_env.fire_dim}
del tmp_env

# 1. 创建神经网络
actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
critic_net = ValueNet(state_dim, hidden_dim).to(device)
actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)

# 2. Agent
student_agent = PPOHybrid(
    actor=actor_wrapper, critic=critic_net, 
    actor_lr=actor_lr, critic_lr=critic_lr,
    lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma, 
    device=device, k_entropy=k_entropy, max_std=0.3
)

if __name__ == "__main__":
    # --- IL Training Phase ---
    il_transition_dict = load_il_and_transitions(os.path.join(cur_dir, "IL"), "il_transitions_combat.pkl", None)
    
    logs_dir = os.path.join(project_root, "logs/combat_parallel")
    log_dir = os.path.join(logs_dir, f"Parallel_Run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    save_meta_once(os.path.join(log_dir, "actor.meta.json"), student_agent.actor.state_dict())
    
    if il_transition_dict is not None and IL_epoches > 0:
        print("Start MARWIL Training...")
        il_transition_dict['actions'] = restructure_actions(il_transition_dict['actions'])
        il_transition_dict['states'] = np.array(il_transition_dict['states'], dtype=np.float32)
        il_transition_dict['returns'] = np.array(il_transition_dict['returns'], dtype=np.float32)

        for epoch in range(IL_epoches): 
            avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
                il_transition_dict, beta=1.0, batch_size=128, label_smoothing=0.3
            )
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: A_Loss: {avg_actor_loss:.4f}, C_Loss: {avg_critic_loss:.4f}")
        print("IL Training Finished.")

    # --- Parallel RL Phase ---
    print(f"Starting {args.n_envs} Parallel Environments...")
    
    # 环境工厂
    def make_env():
        def _thunk():
            env = ChooseStrategyEnv(args, tacview_show=0) # 并行训练通常不开可视化
            env.shielded = 1
            env.dt_move = 0.05
            return env
        return _thunk
    
    env_fns = [make_env() for _ in range(args.n_envs)]
    vec_env = ParallelPettingZooEnv(env_fns)
    
    # 初始化 Buffer
    steps_per_rollout = 400 # 每个环境采集多少步更新一次
    buffer_size = steps_per_rollout
    
    replay_buffer = HybridReplayBuffer(
        n_envs=args.n_envs,
        buffer_size=buffer_size,
        obs_dim=state_dim,
        state_dim=state_dim, # Critic 使用 Global State (这里假设维度相同，若不同需修改)
        action_dims_dict=action_dims_dict,
        use_truncs=True, # 代码要求 truncs
        use_active_masks=True,
        device=device
    )

    total_steps = 0
    save_interval = 20000
    
    # 初始 Reset
    reset_results = vec_env.reset()
    b_obs = reset_results['b_obs']
    
    # [关键修正] 初始化 b_state (Current State for T=0)
    # Reset 返回的 b_state 即为初始状态
    b_state = reset_results['b_state']
    
    infos = reset_results['infos']
    r_check_obs = reset_results['r_check_obs'] # 用于红方规则
    
    # 用于记录 Rule 动作的持久变量 (PFSP中Rule可能需要历史信息)
    last_r_action_labels = [0] * args.n_envs

    print("Start Parallel RL Loop...")
    
    try:
        while total_steps < args.max_steps:
            replay_buffer.clear()
            
            # --- 统计计数器 (收集周期内) ---
            rollout_total_reward = 0       # 总奖励
            rollout_missile_counts = 0     # 导弹发射总数
            rollout_wins = 0               # 胜场数
            rollout_loses = 0              # 负场数
            rollout_draws = 0              # 平场数
            
            # --- Rollout Loop ---
            for step_idx in range(steps_per_rollout):
                # 1. 蓝方决策 (NN)
                # PPOHybrid2 的 take_action 返回字典
                b_action_exec, b_action_raw, _, _ = student_agent.take_action(b_obs, explore=True)
                
                # 统计导弹 (bern > 0.5)
                # b_action_exec['bern'] 是 (N_envs, 1) 的 numpy 数组
                fired_this_step = np.sum(b_action_exec['bern'] > 0.5)
                rollout_missile_counts += fired_this_step
                
                r_actions_list = []
                for i in range(args.n_envs):
                    # 从 unscale_state 获取真实物理状态
                    # 注意：这里需要 env 实例的方法，但主进程没有 env 实例
                    # 解决方案：假设 ChooseStrategyEnv2.unscale_state 是静态逻辑 
                    # 或者直接在 Worker 里传回 unscaled state
                    # 既然旧代码是 unscale_state(r_check_obs)，我们这里模拟一下
                    # 简单起见，假设 Worker 返回的 r_check_obs 可以直接用于 basic_rules，
                    # 或者我们在 make_env 时创建一个 dummy env 来调用 unscale
                    
                    # 为了严谨，建议在 Worker 里做 unscale，这里暂且用 tmp_env (需要重新实例化一个 dummy)
                    if 'dummy_env' not in locals():
                        dummy_env = ChooseStrategyEnv(args)
                    
                    r_state_real = dummy_env.unscale_state(r_check_obs[i])
                    
                    # 调用 Basic Rule
                    r_act, _ = basic_rules(r_state_real, rules_num=0, last_action=last_r_action_labels[i])
                    r_actions_list.append(r_act)
                    last_r_action_labels[i] = r_act # 更新历史
                
                # 3. 构造动作字典
                actions_dispatch = {
                    'r': r_actions_list,
                    'b': b_action_exec # 包含 'cat' 和 'bern'
                }
                
                # 4. 并行步进
                results = vec_env.step(actions_dispatch)
                
                # 获取 T+1 数据
                next_b_obs = results['b_obs']
                next_b_state = results['b_state'] # Next State
                rewards = results['b_reward']
                dones = results['dones']
                b_active_masks = results['b_active_masks']
                truncs = results['truncs']
                result_infos = results['infos']
                
                # 累加奖励 (Sum rewards across all envs for this step)
                rollout_total_reward += np.sum(rewards)
                
                # 统计胜负平 (遍历 infos)
                for info in result_infos:
                    if info.get('win', False): rollout_wins += 1
                    elif info.get('lose', False): rollout_loses += 1
                    elif info.get('draw', False): rollout_draws += 1
                
                # 5. 存入 Buffer
                # [关键修正] state=b_state (T), next_state=next_b_state (T+1)
                replay_buffer.add(
                    obs=b_obs,
                    state=b_state,
                    action_dict=b_action_raw, # 存入 Raw Action (Logits/Idx)
                    reward=rewards,
                    done=dones,
                    next_state=next_b_state,
                    active_mask=b_active_masks,
                    trunc=truncs
                )
                
                # 6. 更新循环变量
                b_obs = next_b_obs
                b_state = next_b_state # 更新当前状态为 T+1
                r_check_obs = results['r_check_obs'] # 更新红方状态
                
                total_steps += 1 # 每个环境走了一步(决策步)
            
            # --- 统计检查与计算 ---
            total_battles = rollout_wins + rollout_loses + rollout_draws
            
            # 计算一个回合最大的决策步数: T_max / T_decide
            T_decide = action_cycle_multiplier * dt_maneuver
            max_episode_decision_steps = args.max_episode_len / T_decide if T_decide > 0 else 1.0

            if total_battles == 0:
                raise RuntimeError("Error: No episodes finished in this rollout cycle. Please increase buffer size (steps_per_rollout)!")
            
            # 【修正 3：平均奖励缩放】
            # 计算缩放因子： (最大决策步数 / 经验收集步数)
            scaling_factor = max_episode_decision_steps / steps_per_rollout
            
            # Rollout 总奖励平均到每个环境/每个完整回合的尺度上
            avg_rollout_reward = (rollout_total_reward / args.n_envs) * scaling_factor
            
            # 导弹数平均到每个环境/每个完整回合的尺度上
            avg_missiles = (rollout_missile_counts / args.n_envs) * scaling_factor
            
            # 【修正 2：胜负平比例】
            # 注意：这里计算的是总的胜场/败场/平场数。如果目标是比例，则除以总战斗场次
            win_ratio = rollout_wins / total_battles
            lose_ratio = rollout_loses / total_battles
            draw_ratio = rollout_draws / total_battles
            
            # --- Update Phase ---
            # 计算 GAE 并展平数据
            transition_dict = replay_buffer.compute_estimates_and_flatten(
                critic_net=student_agent.critic,
                gamma=gamma,
                lmbda=lmbda
            )
            
            # PPO 更新
            student_agent.update(transition_dict, adv_normed=True, mini_batch_size=128)
            
            # --- Logging (恢复所有监控项) ---
            # 1. 业务指标
            logger.add("train/1 episode_return (avg)", avg_rollout_reward, total_steps) 
            logger.add("train/2 win (ratio)", win_ratio, total_steps)
            logger.add("train/2 lose (ratio)", lose_ratio, total_steps)
            logger.add("train/2 draw (ratio)", draw_ratio, total_steps)
            logger.add("special/0 missiles (avg_count)", avg_missiles, total_steps)
            
            # 2. 训练进度
            logger.add("train/11 episode/step", total_steps / args.max_episode_len, total_steps) # 估算 Episode 数
            
            # 3. 梯度与网络监控 (PPOHybrid 内部属性)
            logger.add("train/5 actor_pre_clip_grad", student_agent.pre_clip_actor_grad, total_steps)
            logger.add("train/6 critic_pre_clip_grad", student_agent.pre_clip_critic_grad, total_steps)
            
            # 4. 损失函数
            logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
            logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
            
            # 5. 熵 (总熵及分项)
            logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
            logger.add("train/9 entropy_cat", student_agent.entropy_cat, total_steps)
            logger.add("train/9 entropy_bern", student_agent.entropy_bern, total_steps)
            
            # 6. PPO 诊断指标 (Advantage, Explained Var, Clip Frac)
            # 注意: student_agent.advantage 需要在 PPOHybrid 中正确计算并保存
            logger.add("train/10 advantage", student_agent.advantage, total_steps)
            logger.add("train/10 explained_var", student_agent.explained_var, total_steps)
            logger.add("train/10 clip_frac", student_agent.clip_frac, total_steps)

            print(f"Step {total_steps}: Reward {avg_rollout_reward:.2f}, Win Ratio {win_ratio:.2f}, Loss {student_agent.actor_loss:.3f}")
            
            # Save Model
            if total_steps % save_interval < args.n_envs * steps_per_rollout:
                save_idx = total_steps // save_interval
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"actor_rein_{save_idx}.pt"))
                torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        vec_env.close()
        logger.close()
        print("Done.")