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
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from BasicRules import *
from Envs.Tasks.Wated.ChooseStrategyEnv3 import *

def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()

def load_il_and_transitions(folder, il_name, rl_name):
    if folder is None:
        folder = os.getcwd()
    il_path = os.path.join(folder, il_name)
    trans_path = os.path.join(folder, rl_name)
    il = None
    trans = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
        print(f"Loaded IL data from: {il_path}")
    else:
        print(f"File NOT found: {il_path}")
        
    if os.path.isfile(trans_path):
        with open(trans_path, "rb") as f:
            trans = pickle.load(f)
    return il, trans

def restructure_actions(actions_data):
    """
    将 list of dicts 转换为 dict of arrays
    """
    if isinstance(actions_data, dict):
        return actions_data
    
    if isinstance(actions_data, list) and len(actions_data) > 0:
        print("Restructuring actions from List[Dict] to Dict[Array]...")
        new_actions = {'cat': [], 'bern': []}
        
        for item in actions_data:
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item()
            
            if isinstance(act, dict):
                if 'fly' in act:
                    new_actions['cat'].append(act['fly'])
                if 'fire' in act:
                    new_actions['bern'].append(act['fire'])
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

def summarize(il_dict):
    if il_dict is None: return
    print("\nData Summary:")
    for k in ('states', 'actions', 'returns'):
        v = il_dict.get(k)
        if isinstance(v, dict):
            for sk, sv in v.items():
                print(f"  {k}['{sk}']: shape={sv.shape}")
        else:
            print(f"  {k}: shape={getattr(v, 'shape', 'Unknown')}")
            
def append_agent_experience(td, obs, state, action_dict, reward, next_state, done):
    """
    通用经验添加函数，用于将 Maverick 和 Goose 的独立经验追加到各自的 Buffer 中。
    """
    td['obs'].append(obs)
    td['states'].append(state)
    td['actions'].append(action_dict)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    return td

# ==============================================================================
# Setup & Config
# ==============================================================================

# 加载数据
il_transition_dict, transition_dict = load_il_and_transitions(
    os.path.join(cur_dir, "IL"),
    "il_transitions_combat.pkl",
    "transition_dict_combat.pkl"
)

# 数据重构
if il_transition_dict is not None:
    il_transition_dict['actions'] = restructure_actions(il_transition_dict['actions'])
    il_transition_dict['states'] = np.array(il_transition_dict['states'], dtype=np.float32)
    il_transition_dict['returns'] = np.array(il_transition_dict['returns'], dtype=np.float32)

parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=10*60, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=55e3, help="")
args = parser.parse_args()

# 超参数
actor_lr = 1e-3
critic_lr = actor_lr * 5
IL_epoches = 0
max_steps = 165e4
hidden_dim = [128, 128, 128]
gamma = 0.95
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy = 0.05

env = ChooseStrategyEnv(args)
state_dim = env.obs_dim
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
action_bound = None

# ==============================================================================
# Agent Initialization (Double Agent Setup)
# ==============================================================================

# 1. Maverick (Pilot) - 负责机动
# 动作空间：仅 cat (机动)
maverick_dims = {'cont': 0, 'cat': env.fly_act_dim, 'bern': 0}

maverick_actor = PolicyNetHybrid(state_dim, hidden_dim, maverick_dims).to(device)
maverick_critic = ValueNet(state_dim, hidden_dim).to(device)
maverick_wrapper = HybridActorWrapper(maverick_actor, maverick_dims, action_bound, device).to(device)

maverick_agent = PPOHybrid(
    actor=maverick_wrapper, 
    critic=maverick_critic, 
    actor_lr=actor_lr, 
    critic_lr=critic_lr,
    lmbda=lmbda, 
    epochs=epochs, 
    eps=eps, 
    gamma=gamma, 
    device=device, 
    k_entropy=k_entropy, 
    max_std=0.3
)

# 2. Goose (WSO) - 负责开火
# 动作空间：仅 bern (开火)
goose_dims = {'cont': 0, 'cat': [], 'bern': env.fire_dim}

goose_actor = PolicyNetHybrid(state_dim, hidden_dim, goose_dims).to(device)
goose_critic = ValueNet(state_dim, hidden_dim).to(device)
goose_wrapper = HybridActorWrapper(goose_actor, goose_dims, action_bound, device).to(device)

goose_agent = PPOHybrid(
    actor=goose_wrapper, 
    critic=goose_critic, 
    actor_lr=actor_lr, 
    critic_lr=critic_lr,
    lmbda=lmbda, 
    epochs=epochs, 
    eps=eps, 
    gamma=gamma, 
    device=device, 
    k_entropy=k_entropy, 
    max_std=0.3
)

# 3. Adversary (Red) - 保持原有的混合结构以兼容代码，但不训练
# 如果红方也是 AI，它需要完整的动作空间
adv_dims = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
adv_actor = PolicyNetHybrid(state_dim, hidden_dim, adv_dims).to(device)
adv_critic = ValueNet(state_dim, hidden_dim).to(device)
adv_wrapper = HybridActorWrapper(adv_actor, adv_dims, action_bound, device).to(device)
adv_agent = PPOHybrid(adv_wrapper, adv_critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)


if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # Imitation Learning (MARWIL) - Split Data
    # ------------------------------------------------------------------
    if il_transition_dict is None:
        print("No il_transitions_combat file found.")
        sys.exit(1)

    summarize(il_transition_dict)

    # 准备分离的 IL 数据
    print("Splitting IL data for Maverick and Goose...")
    # Maverick 只需要 cat 动作
    maverick_il_data = copy.deepcopy(il_transition_dict)
    maverick_il_data['actions'] = {'cat': il_transition_dict['actions']['cat']}
    
    # Goose 只需要 bern 动作
    goose_il_data = copy.deepcopy(il_transition_dict)
    goose_il_data['actions'] = {'bern': il_transition_dict['actions']['bern']}

    # 日志
    logs_dir = os.path.join(project_root, "logs/combat")
    mission_name = 'ILRL_打rule0_MavGoose'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    print("Start Dual-Agent MARWIL Training...")

    for epoch in range(IL_epoches): 
        # Update Maverick (Pilot)
        mav_al, mav_cl, mav_c = maverick_agent.MARWIL_update(
            maverick_il_data, beta=1.0, batch_size=128, label_smoothing=0.1
        )
        
        # Update Goose (WSO)
        goose_al, goose_cl, goose_c = goose_agent.MARWIL_update(
            goose_il_data, beta=1.0, batch_size=128, label_smoothing=0.1
        )
        
        if epoch % 5 == 0:
            logger.add("il_mav/actor_loss", mav_al, epoch)
            logger.add("il_mav/critic_loss", mav_cl, epoch)
            logger.add("il_goose/actor_loss", goose_al, epoch)
            logger.add("il_goose/critic_loss", goose_cl, epoch)
            print(f"Ep {epoch}: Mav Loss(A/C): {mav_al:.3f}/{mav_cl:.3f} | Goose Loss(A/C): {goose_al:.3f}/{goose_cl:.3f}")

    print("IL Training Finished.")

    # ==============================================================================
    # Reinforcement Learning (Self-Play / Rule)
    # ==============================================================================
    
    tacview_input = input("Enable tacview visualization? (0=no, 1=yes) [default 0]: ").strip()
    tacview_show = 1 if tacview_input == "1" else 0
    
    env = ChooseStrategyEnv(args, tacview_show=tacview_show)
    env.shielded = 1 
    env.dt_move = 0.05 
    
    t_bias = 0

    def create_initial_state():
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 45e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier
    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 

    i_episode = 0 
    total_steps = 0
    
    # 两个独立的 Buffer
    mav_transition_dict = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
    goose_transition_dict = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
    
    while total_steps < int(max_steps):
        i_episode += 1
        
        # 对手设置 (强制 Rule 0)
        adv_is_rule = True
        rule_num = 1
        print(f"Eps {i_episode}: Opponent is Rule {rule_num}")
        
        episode_return = 0
        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = create_initial_state()
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE)
        
        # 状态变量初始化
        last_decision_obs = None 
        last_decision_state = None
        current_mav_action = None   # {'cat': ...}
        current_goose_action = None # {'bern': ...}
        
        # 奖励暂存 (用于累积 step 间的奖励)
        accum_reward_fly = 0
        accum_reward_fire = 0
        
        dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}
        done = False
        env.dt_maneuver = dt_maneuver
        
        last_r_action_label = 0
        steps_of_this_eps = -1
        m_fired = 0
        
        # Episode Loop
        for count in range(round(args.max_episode_len / dt_maneuver)):
            steps_of_this_eps += 1
            if env.running == False or done: break
            
            # 获取观测
            r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
            b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)         # Actor Input
            b_state_global, _ = env.obs_1v1('b', reward_fn=1)      # Critic Input
            
            # --- 决策点 (每 10 步) ---
            if steps_of_this_eps % action_cycle_multiplier == 0:
                # 1. 存储上一周期的经验
                if steps_of_this_eps > 0 and not dead_dict['b']:
                    # Maverick 存储 (Reward Fly)
                    mav_transition_dict = append_agent_experience(
                        mav_transition_dict, last_decision_obs, last_decision_state, 
                        current_mav_action, accum_reward_fly, b_state_global, False
                    )
                    # Goose 存储 (Reward Fire)
                    goose_transition_dict = append_agent_experience(
                        goose_transition_dict, last_decision_obs, last_decision_state, 
                        current_goose_action, accum_reward_fire, b_state_global, False
                    )
                
                # 重置累积奖励
                accum_reward_fly = 0
                accum_reward_fire = 0
                
                # 2. 产生新动作
                last_decision_obs = b_obs
                last_decision_state = b_state_global
                
                # --- Red Side ---
                r_state_check = env.unscale_state(r_check_obs)
                if adv_is_rule:
                    r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=last_r_action_label)
                else:
                    r_out, _, _, _ = adv_agent.take_action(r_obs, explore=1)
                    r_action_label = r_out['cat'][0]
                    r_fire = r_out['bern'][0]
                last_r_action_label = r_action_label

                # --- Blue Side (Maverick & Goose) ---
                # Maverick 决策
                mav_exec, mav_raw, _, mav_check = maverick_agent.take_action(b_obs, explore=1)
                b_action_label = mav_exec['cat'][0]
                
                # Goose 决策
                goose_exec, goose_raw, _, goose_check = goose_agent.take_action(b_obs, explore=1)
                b_fire = goose_exec['bern'][0]
                
                # 记录动作字典用于 Buffer
                current_mav_action = {'cat': mav_exec['cat']}   # shape (1,) or array
                current_goose_action = {'bern': goose_exec['bern']} # shape (1,)

                # 执行开火
                b_m_id = None
                if b_fire:
                    b_m_id = launch_missile_immediately(env, 'b')
                if b_m_id is not None:
                    m_fired += 1
            
            # --- 环境步进 ---
            r_maneuver = env.maneuver14(env.RUAV, r_action_label)
            b_maneuver = env.maneuver14(env.BUAV, b_action_label)
            env.step(r_maneuver, b_maneuver)
            
            # --- 获取奖励 ---
            # done, reward_total, reward_assisted, reward_fly, reward_fire
            done, b_reward_total, b_rew_assist, b_rew_fly, b_rew_fire = env.combat_terminate_and_reward(
                'b', b_action_label, b_m_id is not None
            )
            
            # 累积奖励
            if steps_of_this_eps % action_cycle_multiplier == 0:
                # 仅用于统计
                episode_return += b_reward_total 
            
            # 累积用于训练的奖励 (Step-wise accumulation)
            # 注意：env 返回的是瞬时奖励，我们需要在整个 action_cycle 期间累积它，或者像原代码一样只取最后时刻?
            # 原代码逻辑：episode_return += b_reward (每step都加?? 不，原代码只在 % 10 == 0 时加)
            # check original code: 
            #   accumulate logic was: `if steps... % 10 == 0: episode_return += b_reward`
            #   And `append_b_experience` used `b_reward`.
            #   BUT `combat_terminate_and_reward` is called EVERY STEP.
            #   If we only record reward at decision step, we might miss intermediate penalties?
            #   原代码逻辑确实是：每步 step，每步 get reward，但只在 decision step 时 append experience。
            #   并且 append 的 reward 是 `b_reward` (当前step的瞬时值)。
            #   这意味着 step 1-9 的奖励被丢弃了？
            #   -> 这是一个常见的 Bug 或 特性。如果奖励函数是稠密的且平滑的，取采样点的值也没问题。
            #   -> 我们保持原代码逻辑：只使用 Decision Step 的 reward。
            
            accum_reward_fly = b_rew_fly      # 覆盖，只取决策点当下的瞬时值 (保持原逻辑)
            accum_reward_fire = b_rew_fire
            
            # 死亡检测
            if dead_dict['b'] == 0:
                next_b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                if env.BUAV.dead:
                    dead_dict['b'] = 1

            total_steps += 1
            env.render(t_bias=t_bias)

        # --- End of Episode Handling ---
        # 存储最后一步
        if last_decision_state is not None:
            mav_transition_dict = append_agent_experience(
                mav_transition_dict, last_decision_obs, last_decision_state, 
                current_mav_action, accum_reward_fly, next_b_state_global, True
            )
            goose_transition_dict = append_agent_experience(
                goose_transition_dict, last_decision_obs, last_decision_state, 
                current_goose_action, accum_reward_fire, next_b_state_global, True
            )
            episode_return += b_reward_total

        print(f"  Result: Win={env.win}, Lose={env.lose}, Draw={env.draw}, Return={episode_return:.2f}, Fired={m_fired}")
        
        # Logging
        logger.add("train/return", episode_return, total_steps)
        logger.add("train/missiles_fired", m_fired, total_steps)
        logger.add("train/win_rate", int(env.win), total_steps)

        # --- Update Networks ---
        if len(mav_transition_dict['dones']) >= transition_dict_threshold:
            print("Updating Agents...")
            # Update Maverick
            maverick_agent.update(mav_transition_dict, adv_normed=1)
            # Update Goose
            goose_agent.update(goose_transition_dict, adv_normed=1)
            
            # Log Training Stats
            logger.add("mav/actor_loss", maverick_agent.actor_loss, total_steps)
            logger.add("mav/critic_loss", maverick_agent.critic_loss, total_steps)
            logger.add("mav/entropy", maverick_agent.entropy_mean, total_steps)
            
            logger.add("goose/actor_loss", goose_agent.actor_loss, total_steps)
            logger.add("goose/critic_loss", goose_agent.critic_loss, total_steps)
            logger.add("goose/entropy", goose_agent.entropy_mean, total_steps)

            # Reset Buffers
            mav_transition_dict = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
            goose_transition_dict = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

        # Clear Render & Save
        env.clear_render(t_bias=t_bias)
        t_bias += env.t
        
        if i_episode % 10 == 0:
            # Maverick Actor (带回合数)
            torch.save(maverick_agent.actor.state_dict(), os.path.join(log_dir, f"maverick_actor_{i_episode}.pt"))
            # Goose Actor (带回合数)
            torch.save(goose_agent.actor.state_dict(), os.path.join(log_dir, f"goose_actor_{i_episode}.pt"))
            
            # Maverick Critic (覆盖保存)
            torch.save(maverick_agent.critic.state_dict(), os.path.join(log_dir, "maverick_critic.pt"))
            # Goose Critic (覆盖保存)
            torch.save(goose_agent.critic.state_dict(), os.path.join(log_dir, "goose_critic.pt"))
            print(f"Saved models at episode {i_episode}")

    env.end_render()
    print("Training Finished.")
    logger.close()