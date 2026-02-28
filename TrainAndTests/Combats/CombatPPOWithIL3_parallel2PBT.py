'''
大改动：PBT (Population Based Training) 版本
1. 引入种群概念 (Population)，包含多个 Agent，每个 Agent 拥有独立的 alpha_il。
2. 采用轮流采样 (Round-Robin) 机制，共享并行 Workers。
3. 增加 PBT 进化步：
   - Exploit: 末位淘汰，复制最优者权重。
   - Momentum Reset: 复制后重置优化器。
   - Explore: 扰动超参数 alpha_il。
4. 保持原有的策略蒸馏和 Elo 匹配机制。
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
import time  # 确保引入 time 模块
from datetime import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import traceback # [新增]
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0_distil2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from prepare_il_datas import run_rules
from VsBaseline_while_training2 import test_worker
from UPolicyWrapper import UnifiedPolicyWrapper

EXEC_COUNT = 0

dt_move = 0.05

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

# ==========================================
# 核心修复：数据结构重组函数
# ==========================================
def restructure_actions(actions_data):
    """
    将 list of dicts [{'fly': 1, 'fire': 0}, ...] 
    转换为 dict of arrays {'cat': array([[1],...]), 'bern': array([[0],...])}
    并确保维度是 (N, 1) 以适配 PPOHybrid2 的索引操作
    """
    # 如果已经是字典格式，直接返回
    if isinstance(actions_data, dict):
        return actions_data
    
    # 如果是列表且包含字典，进行转换
    if isinstance(actions_data, list) and len(actions_data) > 0:
        # print("Restructuring actions from List[Dict] to Dict[Array]...") # 频繁调用可注释掉以减少刷屏
        
        # 初始化容器
        new_actions = {'cat': [], 'bern': []}
        
        for item in actions_data:
            # 兼容处理：item 可能是 dict，也可能是包含 dict 的 numpy array
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item() # 提取 numpy 里的 dict
            
            # 映射 'fly' -> 'cat' (离散机动)
            # 映射 'fire' -> 'bern' (开关开火)
            if isinstance(act, dict):
                val_cat = act.get('fly', act.get('cat'))
                if val_cat is not None:
                    new_actions['cat'].append(val_cat)
                
                # 优先找 'fire'，找不到找 'bern'
                val_bern = act.get('fire', act.get('bern'))
                if val_bern is not None:
                    new_actions['bern'].append(val_bern)

            # 备用：如果数据意外变成了 list/tuple
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                 new_actions['cat'].append(act[0])
                 new_actions['bern'].append(act[1])

        # 转换为 Numpy Array 并调整形状为 (Batch, 1)
        # 这一点至关重要：PPOHybrid2 里的 expert_cat[:, i] 需要 expert_cat 是二维的
        
        # 1. 'cat': 离散动作，转为 int64，Reshape 为 (N, 1)
        cat_arr = np.array(new_actions['cat'], dtype=np.int64)
        if cat_arr.ndim == 1:
            cat_arr = cat_arr.reshape(-1, 1)
        
        # 2. 'bern': 伯努利动作，转为 float32 (BCE Loss需要)，Reshape 为 (N, 1)
        bern_arr = np.array(new_actions['bern'], dtype=np.float32)
        if bern_arr.ndim == 1:
            bern_arr = bern_arr.reshape(-1, 1)

        result = {
            'cat': cat_arr,
            'bern': bern_arr
        }
        
        print(f"Structure fixed: 'cat' shape={cat_arr.shape}, 'bern' shape={bern_arr.shape}")
        return result

    return actions_data

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

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
            
def append_experience(td, obs, state, action, reward, next_state, done, active_mask):
    """
    统一把一次蓝方经验追加到 transition_dict。
    修改：增加 obs 输入，用于存储局部观测
    """
    td['obs'].append(obs) # 新增：存储Actor用的局部观测
    td['states'].append(state) # 修改：这里存储Critic用的全局状态(pomdp=0)
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    td['active_masks'].append(active_mask) # 【新增】active_mask，转入多智能体
    return td



# 加载数据
original_il_transition_dict, _ = load_il_and_transitions(
    os.path.join(cur_dir, "IL"),
    "il_transitions_combat_LR.pkl",
    "transition_dict_combat_LR.pkl"
)
original_il_transition_dict0 = copy.deepcopy(original_il_transition_dict)

# --- 关键步骤：执行数据重构 ---
if original_il_transition_dict is not None:
    # 这里完成 (Batch, Key) -> (Key, Batch) 的转换
    original_il_transition_dict['actions'] = restructure_actions(original_il_transition_dict0['actions'])
    
    # 顺便确保 states 和 returns 也是标准的 float32 numpy array
    original_il_transition_dict['states'] = np.array(original_il_transition_dict0['states'], dtype=np.float32)
    original_il_transition_dict['returns'] = np.array(original_il_transition_dict0['returns'], dtype=np.float32)

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None, 
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta', 
                               rule_rate=0.5, deltaFSP_epsilon=0.5):
    """
    优化后的对手采样逻辑：
    1. 优先判定是否进入“规则复习”分支。
    2. 若未进入，则根据 SP_type 执行具体的采样策略。
    """
    # 【核心修改】在函数内部合并出一个临时的全集字典用于查询分数
    # 这样 keys 里的任何元素都能在这里找到对应的 ELO
    
    if hall_of_fame is not None:
        candidate_pool = hall_of_fame.copy()
        candidate_pool.update(elite_elo_ratings)
    else:
        candidate_pool = elite_elo_ratings
    
    keys = list(candidate_pool.keys())
    if not keys:
        return np.array([]), []

    # --- 第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护) ---
    # 只要 rule_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < rule_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    # --- 第二层判断：进入核心采样逻辑 ---
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = np.max(elos)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
        diffs = elos - actual_target
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
        return probs, keys

    # 2. 处理 FSP (全样本均匀分布)
    elif SP_type == 'FSP':
        probs = np.ones(len(keys)) / len(keys)
        return probs, keys

    # 3. 处理 deltaFSP (新旧池切分)
    elif SP_type == 'deltaFSP':
        n = len(keys)
        new_count = max(1, int(np.ceil(n * 0.2)))
        new_keys = keys[-new_count:]
        old_keys = keys[:-new_count]
        
        # 这里的 deltaFSP_epsilon 建议直接作为参数传入或使用全局变量
        if np.random.rand() < float(deltaFSP_epsilon) or not old_keys:
            target_keys = new_keys
        else:
            target_keys = old_keys
            
        probs = np.ones(len(target_keys)) / len(target_keys)
        return probs, target_keys

    # 4. 处理 SP (最强/最新历史版本)
    elif SP_type == 'SP':
        # rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
        rein_keys = [k for k in keys if re.match(r'^actor_rein\d+(_P\d+)?$', k)]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            try: return int(re.search(r'actor_rein(\d+)', k).group(1))
            except: return -1
            
        best_key = max(rein_keys, key=extract_number)
        return np.array([1.0]), [best_key]

    # 5. 兜底逻辑: Rule 均匀采样 (None)
    else:
        if not rule_keys: return np.array([]), []
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = 9000
    red_height = 9000
    red_psi = -np.pi/2
    blue_psi = np.pi/2
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N = init_North
    red_E = 45e3
    blue_N = init_North
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process(rank, pipe, args, state_dim, hidden_dim, 
                   action_dims_dict, device_worker, dt_maneuver, 
                   seed, opp_greedy_rate, dt_move=0.05, no_crash=1):
    """
    常驻子进程：接收参数 -> 跑完一整场 -> 返回数据 -> 等待
    完整的 Worker 逻辑：包含环境初始化、模型加载、仿真循环、数据回传
    """
    try:  # <--- 【新增】添加此行，并将下方所有代码整体缩进
        # --- 1. 初始化阶段 (只运行一次) ---
        
        # 确保每个进程种子不同，避免所有环境生成完全一样的随机数
        worker_seed = seed + rank * 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        # 初始化环境 (关闭可视化以加速)
        env = ChooseStrategyEnv(args, tacview_show=False)
        env.shielded = no_crash # 假设默认开启防撞
        env.dt_move = dt_move
        env.dt_maneuver = dt_maneuver

        # 初始化本地网络 (CPU)
        local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        # 【修改 1】创建一个 dummy critic，仅为了满足 PPOHybrid 初始化要求
        local_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)
        local_agent = PPOHybrid(
            actor=HybridActorWrapper(local_actor, action_dims_dict, None, device_worker).to(device_worker),
            critic=local_dummy_critic,  # <--- 【修改】传入实体对象，而非 None
            actor_lr=0, critic_lr=0,    # 学习率为0，确保不会更新
            lmbda=0, eps=0, gamma=0, epochs=0, # 补全位置参数
            device=device_worker 
        )
        
        # 初始化对手网络
        adv_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        # 【修改 2】同样为对手创建一个 dummy critic
        adv_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)
        adv_agent = PPOHybrid(
            actor=HybridActorWrapper(adv_actor, action_dims_dict, None, device_worker).to(device_worker),
            critic=adv_dummy_critic,    # <--- 【修改】传入实体对象，而非 None
            actor_lr=0, critic_lr=0, 
            lmbda=0, eps=0, gamma=0, epochs=0, # 补全位置参数
            device=device_worker
        )

        # --- 2. 循环等待阶段 ---
        while True:
            # 阻塞等待指令
            cmd, packet = pipe.recv()
            
            if cmd == 'EXIT':
                env.close()
                break
                
            if cmd == 'RUN_EPISODE':
                # 解包数据
                (actor_weights, opponent_info, settings) = packet
                
                # A. 同步权重 (极快)
                local_agent.actor.load_state_dict(actor_weights)
                
                # B. 配置对手
                opp_name, opp_type, opp_data = opponent_info
                adv_is_rule = (opp_type == 'rule')
                rule_num = 0
                if adv_is_rule:
                    rule_num = opp_data
                else:
                    adv_agent.actor.load_state_dict(opp_data)

                # C. 准备本回合容器
                # Worker 收集完整的 ego_trans (用于 SIL) 和 enm_trans (用于 SIL)
                # local_trans 用于 PPO 更新 (只包含 Blue 视角)
                local_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                ego_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                enm_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}

                # D. 环境重置
                randomized_birth = settings['randomized_birth']
                action_cycle_multiplier = settings['action_cycle_multiplier']
                reward_weight = settings['weight_reward']
                # 在子环境中重新计算出生状态
                # red_birth, blue_birth = create_initial_state_worker(randomized_birth)
                # 使用从master传来的出生状态
                red_birth = settings['red_birth']
                blue_birth = settings['blue_birth']
                env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_ammo=6, blue_init_ammo=6)
                
                # 状态变量初始化
                done = False
                last_decision_obs, last_decision_state = None, None
                last_enm_decision_obs, last_enm_decision_state = None, None
                current_action, current_action_exec, current_enm_action_exec = None, None, None
                
                steps_run = 0
                episode_return = 0 # 仅用于统计显示
                m_fired = 0
                
                dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}
                
                # --- E. 仿真循环 (核心物理逻辑) ---
                # 计算最大步数
                max_counts = int(args.max_episode_len / dt_maneuver)
                
                for count in range(max_counts):
                    if not env.running or done: break
                    
                    # 1. 获取观测
                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                    b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    r_state_global, _ = env.obs_1v1('r', reward_fn=1)

                    # 2. 决策点 (Action Cycle)
                    if steps_run % action_cycle_multiplier == 0:
                        # 2.1 存储【上一个】周期的经验
                        if steps_run > 0:
                            # 注意：这里调用你原文件里的 append_experience 辅助函数
                            # 确保 append_experience 在 这个函数 作用域外是可见的，或者复制进来
                            append_experience(local_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, b_state_global, False, not dead_dict['b'])
                            append_experience(ego_trans, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, b_state_global, False, not dead_dict['b'])
                            append_experience(enm_trans, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, r_state_global, False, not dead_dict['r'])

                        # 2.2 更新上一帧记录
                        last_decision_obs = b_obs
                        last_decision_state = b_state_global
                        last_enm_decision_obs = r_obs
                        last_enm_decision_state = r_state_global
                        
                        # 2.3 产生新动作 (No Grad)
                        with torch.no_grad():
                            # Blue Decision
                            b_action_exec, _, _, _ = local_agent.take_action(b_obs, explore=1)
                            b_action_label = b_action_exec['cat'][0]
                            b_fire = b_action_exec['bern'][0]
                            
                            # Red Decision
                            r_state_check = env.unscale_state(r_check_obs)
                            if adv_is_rule:
                                # 调用规则，假设 basic_rules 已导入
                                r_action_label, r_fire = basic_rules(r_state_check, rule_num, p_random=0.1)
                                r_action_exec = {'cat': np.array([r_action_label]), 'bern': np.array([r_fire], dtype=np.float32)}
                            else:
                                # 随机决定本局对手是否开启探索
                                adv_explore = 1 if np.random.rand() > opp_greedy_rate else 0
                                r_action_exec, _, _, _ = adv_agent.take_action(r_obs, explore={'cont':0, 'cat':adv_explore, 'bern':1})
                                r_action_label = r_action_exec['cat'][0]
                                r_fire = r_action_exec['bern'][0]

                        # 2.4 处理开火
                        b_m_id = launch_missile_immediately(env, 'b') if b_fire else None
                        r_m_id = launch_missile_immediately(env, 'r') if r_fire else None
                        if b_m_id: m_fired += 1
                        
                        # 2.5 记录当前动作供下一帧存储
                        current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                        current_action_exec = {'cat': b_action_exec['cat'], 'bern': np.array([b_m_id is not None])}
                        current_enm_action_exec = {'cat': r_action_exec['cat'], 'bern': np.array([r_m_id is not None])}

                    # 3. 物理步进
                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    steps_run += 1
                    
                    # 4. 奖励计算
                    done, b_rew_event, b_rew_constraint, b_rew_shaping = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier)
                    _, r_rew_event, r_rew_constraint, r_rew_shaping = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier)
                    
                    reward_for_learn = sum(np.array([b_rew_event, b_rew_constraint, b_rew_shaping]) * reward_weight)
                    reward_for_enm = sum(np.array([r_rew_event, r_rew_constraint, r_rew_shaping]) * reward_weight)
                    
                    if steps_run % action_cycle_multiplier == 0 or done:
                        episode_return += (b_rew_event + b_rew_constraint)
                    
                    # 5. 存活更新 (用于 Done 标记)
                    next_b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    next_r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                    dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

                # --- End of Simulation Loop ---
                
                # 6. 存储最后一步经验 (Terminal State)
                # 强制做一次终局判定
                done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False, action_cycle_multiplier)
                
                if last_decision_state is not None:
                    append_experience(local_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(ego_trans, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(enm_trans, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, next_r_state_global, True, not dead_dict['r'])

                # 7. 打包结果
                result_packet = {
                    'trans': local_trans, # 用于 RL Update
                    'ego_trans': ego_trans, # 用于 SIL (win)
                    'enm_trans': enm_trans, # 用于 SIL (lose)
                    'metrics': {
                        'return': episode_return,
                        'steps': steps_run,
                        'win': env.win,
                        'lose': env.lose,
                        'draw': env.draw,
                        'm_fired': m_fired
                    },
                    'opp_name': opp_name
                }
                
                # 8. 发送回 Master
                pipe.send(result_packet)

    except Exception: # [新增] 异常捕获与回传
        print(f"!!! Worker {rank} CRASHED !!!")
        tb = traceback.format_exc()
        print(tb)
        try: pipe.send({'error': tb})
        except: pass

# ==========================================
# PBT Population Helper Class
# ==========================================
class PBTMember:
    def __init__(self, agent_id, agent, elo=1200, alpha_il=0.01):
        self.id = agent_id
        self.agent = agent  # PPOHybrid instance
        self.elo = elo
        self.alpha_il = alpha_il
        self.wins = 0
        self.matches = 0

    def reset_optimizer(self):
        """
        调用 PPOHybrid 内部定义的 reset_optimizer 
        以此来彻底重建 Adam 优化器，清除动量缓存
        """
        self.agent.reset_optimizer()

def run_MLP_simulation(
    num_workers=10, # 并行进程数，根据CPU核数调整，建议 10-20
    mission_name='无名',
    actor_lr=1e-4,
    critic_lr=5e-4,
    actor_lr_init_il = 1e-4,
    critic_lr_init_il = 5e-4,
    IL_epoches=180,
    max_steps=4 * 165e4,
    hidden_dim=None,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=None,
    alpha_il=1.0,
    il_batch_size=128,
    il_batch_size2=128,
    il_buffer_max_size=20000,
    mini_batch_size_mixed=64,
    beta_mixed=1.0,
    label_smoothing=0.3,
    label_smoothing_mixed=0.01,
    action_cycle_multiplier=30,
    trigger0=50e3,
    trigger_delta=50e3,
    weight_reward_0=None,
    IL_rule=2,
    no_crash=1,
    dt_move=0.05,
    max_episode_duration=10*60,
    R_cage = 45e3, # 55e3,
    dt_maneuver=0.2,
    transition_dict_threshold=1000,
    should_kick = True,
    use_init_data = False,
    init_elo_ratings = {
        "Rule_0": 1200,
        "Rule_1": 1200,
        "Rule_2": 1200,
    },
    self_play_type = 'PFSP', # FSP, SP, None 表示非自博弈
    hist_agent_as_opponent = 1, # 是否开始记录历史智能体
    use_sil = True,
    sil_only_maneuver = 1, # 自模仿只包含机动还是也包含开火
    sigma_elo = 400,
    WARM_UP_STEPS = 500e3,
    ADMISSION_THRESHOLD = 0.5,
    MAX_HISTORY_SIZE = 300,  # 100
    deltaFSP_epsilon = 0.8,
    rule_actor_rate = 0.2,
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
    save_interval = 2, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
    opp_greedy_rate = 0.5, # 对手贪婪率
    num_runs = 3, # 测试回合重复次数
    device = torch.device("cpu"),
    max_il_exponent = -2.0,
    k_shape_il = 0.004,
    reverse_kl=0,
    distil_only_maneuver=1,
    # --- PBT Params ---
    pop_size = 4,      # 种群大小
    pbt_interval = 5,  # 多少轮(Generations)进化一次
):

    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 2. 参数与环境配置 (Master 用于获取维度)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = ChooseStrategyEnv(args)
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    # del dummy_env # 不允许删除了，后续还要用
    # ==========================================
    # 新增：teacher_agent类
    # ==========================================
    teacher_agent = UnifiedPolicyWrapper(dummy_env)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Master training device: {device}")

    # 3. 创建基础神经网络 (用于 MARWIL 预训练)
    # 我们先训练一个 base agent，然后复制给 population
    base_actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    base_critic_net = ValueNet(state_dim, hidden_dim).to(device)
    base_actor_wrapper = HybridActorWrapper(base_actor_net, action_dims_dict, None, device).to(device)

    base_agent = PPOHybrid(
        actor=base_actor_wrapper, 
        critic=base_critic_net, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device, 
        k_entropy=k_entropy, 
        max_std=label_smoothing
    )
    
    # 【新增】定义 Teacher Actor 容器 (用于加载历史策略进行蒸馏)
    # 结构必须与 student_agent.actor 完全一致
    teacher_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    teacher_actor = HybridActorWrapper(teacher_net, action_dims_dict, None, device).to(device)
    teacher_actor.eval() # 永远处于验证模式，不更新梯度
    for param in teacher_actor.parameters():
        param.requires_grad = False # 彻底冻结
    teacher_critic = ValueNet(state_dim, hidden_dim).to(device)
    teacher_critic.eval()
    for param in teacher_critic.parameters():
        param.requires_grad = False # 彻底冻结
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    
    save_meta_once(os.path.join(log_dir, "actor.meta.json"), base_agent.actor.state_dict())
    save_meta_once(os.path.join(log_dir, "critic.meta.json"), base_agent.critic.state_dict())
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (只对 base_agent 做一次)
    print("Start MARWIL Training on Base Agent...")
    base_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = base_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, 
            label_smoothing=label_smoothing
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    print("IL Training Finished.")
    

    # ==============================================================================
    # PBT 初始化: 创建种群
    # ==============================================================================
    population = []
    print(f"Initializing Population of size {pop_size}...")
    for i in range(pop_size):
        # 创建新实例
        new_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        new_critic = ValueNet(state_dim, hidden_dim).to(device)
        new_wrapper = HybridActorWrapper(new_actor, action_dims_dict, None, device).to(device)
        
        # 复制预训练权重
        new_wrapper.load_state_dict(base_agent.actor.state_dict())
        new_critic.load_state_dict(base_agent.critic.state_dict())
        
        new_agent_obj = PPOHybrid(
            actor=new_wrapper, critic=new_critic,
            actor_lr=actor_lr, critic_lr=critic_lr,
            lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma, 
            device=device, k_entropy=k_entropy, max_std=label_smoothing
        )
        
        # 初始化 Alpha_IL: 对数均匀分布 [1e-4, 0.1] 之间
        # log10(-4) = -4, log10(0.1) = -1
        rnd_exp = np.random.uniform(-4, -1)
        init_alpha = 10 ** rnd_exp
        
        member = PBTMember(i, new_agent_obj, elo=1200, alpha_il=init_alpha)
        population.append(member)
        print(f"  Agent P{i}: alpha_il={init_alpha:.5f}")

    # 清理 base_agent 节省显存
    del base_agent
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 并行环境初始化 (Worker Setup)
    # ----------------------------------------------------
    
    # 7. 强化学习准备
    
    
    # 进程通信设置
    mp.set_start_method('spawn', force=True)
    
    # --- A. 启动并行测试进程池 (Async Test Pool) ---
    # 这个池子用于 periodic testing，不参与训练数据的生成
    test_pool = mp.Pool(processes=3, maxtasksperchild=10) 
    

    # --- B. 启动并行训练 Worker (Sync Training Workers) ---
    # 这些 Worker 与 Master 同步，负责生成训练数据
    workers = []
    pipes = []
    worker_device = torch.device('cpu') # Worker 使用 CPU 推理
    
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(
            i, child_conn, args, state_dim, hidden_dim, 
            action_dims_dict, worker_device, dt_maneuver, 
            seed, opp_greedy_rate, dt_move, no_crash
        ))
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = copy.deepcopy(elo_ratings)
    hall_of_fame = {}
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # 训练循环变量
    total_steps = 0
    pbt_generation_cnt = 0 # PBT 进化代数
    batch_idx = 0
    trigger = trigger0
    current_max_steps = int(max_steps)
    
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    
    # 历史记录初始化：用 P0 的参数存为初始对手
    init_opponent_name = "actor_rein0"
    init_critic_name = "critic_rein0"
    torch.save(population[0].agent.actor.state_dict(), os.path.join(log_dir, f"{init_opponent_name}.pt"))
    torch.save(population[0].agent.critic.state_dict(), os.path.join(log_dir, f"{init_critic_name}.pt"))
    if self_play_type != 'None': elo_ratings[init_opponent_name] = 1200

    # =========================================================
    # 主循环 (Master Process) - PBT 结构
    # =========================================================
    while True: 
        while total_steps < current_max_steps:
            # --- 1. 并行测试 (Periodic Testing) ---
            # 为了简单起见，我们用当前 Elo 最高的 Agent 进行测试
            if total_steps >= trigger:
                best_member = max(population, key=lambda m: m.elo)
                print(f"\n>>> Triggering Parallel Test (Best Agent P{best_member.id}) at steps {total_steps}...")
                current_weights = {k: v.cpu().clone() for k, v in best_member.agent.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                test_tasks = []
                for r_idx in [0, 1, 2, 3, 4]:
                    obj = test_pool.apply_async(
                        test_worker, 
                        args=(current_weights, r_idx, args, 
                              state_dim, hidden_dim, action_dims_dict, 
                              dt_maneuver, 'cpu', num_runs, action_cycle_multiplier)
                    )
                    test_tasks.append(obj)
                # 等待所有测试进程结束
                test_results = [t.get() for t in test_tasks]

                outcomes = {rule_num: score for rule_num, score, result2 in test_results}
                outcomes_return = {rule_num: result2 for rule_num, score, result2 in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps)
                    logger.add(f"test/agent_vs_rule{r_num}_return", outcomes_return[r_num], total_steps)
                    print(f"  [Test Result] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

                # 名人堂判定：如果全胜则保存并加入池子
                if all(score > 0.5 for score in outcomes.values()):
                    # rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+$', k)]
                    rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+(_P\d+)?$', k)]
                    if rein_keys:
                        # 找到数值最大的编号（即最新的已保存智能体）
                        hof_key = max(rein_keys, key=lambda k: int(k.replace("actor_rein", '')))
                        
                        if hof_key not in hall_of_fame:
                            hall_of_fame[hof_key] = elo_ratings.get(hof_key, best_member.elo)
                            print(f"!!! [Hall of Fame] New Hero Captured: {hof_key}")
                
                trigger += trigger_delta

            # ==================================================================
            # PBT Loop: 每个 Agent 轮流采样和训练
            # ==================================================================
            # 每个 Generation，所有成员都走一遍
            
            for member in population:
                # ---------------------------------------------
                # 2. 准备训练 Batch (针对当前 Member)
                # ---------------------------------------------
                transition_dict = copy.deepcopy(empty_transition_dict)
                
                # A. 获取当前 Agent 权重
                current_actor_weights = {k: v.cpu() for k, v in member.agent.actor.state_dict().items()}
                
                # B. 分发任务
                if not init_elo_ratings:
                    sorted_all_keys = [k for k in sorted(elo_ratings.keys(), 
                                                       key=lambda x: elo_ratings[x] if not x.startswith("__") else -1e9, 
                                                       reverse=True) if not k.startswith("__")]
                    effective_pool = {k: elo_ratings[k] for k in sorted_all_keys[:MAX_HISTORY_SIZE]}
                else:
                    effective_pool = elite_elo_ratings
                
                for rank in range(num_workers):
                    # 对手采样: 种群里的 Agent 也和历史/规则打，产生相对 Elo
                    probs, opponent_keys = get_opponent_probabilities(
                        effective_pool, hall_of_fame, target_elo=member.elo,
                        SP_type=self_play_type, sigma=sigma_elo,
                        rule_rate=rule_actor_rate, deltaFSP_epsilon=deltaFSP_epsilon,
                    )
                    selected_opponent_name = np.random.choice(opponent_keys, p=probs)
                    
                    opp_type = 'rule'
                    opp_data = 0
                    if "Rule" in selected_opponent_name:
                        try: rule_num = int(selected_opponent_name.split('_')[1])
                        except: rule_num = 0
                        opp_data = rule_num
                    else:
                        opp_type = 'nn'
                        adv_path = os.path.join(log_dir, f"{selected_opponent_name}.pt")
                        if os.path.exists(adv_path):
                            opp_data = torch.load(adv_path, map_location='cpu', weights_only=1)
                        else:
                            opp_type = 'rule'; opp_data = 0
                    
                    opp_info = (selected_opponent_name, opp_type, opp_data)
                    rb, bb = create_initial_state_worker(randomized_birth)
                    settings = {
                        'randomized_birth': randomized_birth,
                        'action_cycle_multiplier': action_cycle_multiplier,
                        'weight_reward': weight_reward_0,
                        'red_birth': rb, 'blue_birth': bb
                    }
                    pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

                # C. 收集数据
                batch_results = []
                for rank in range(num_workers):
                    try:
                        res = pipes[rank].recv()
                    except EOFError:
                        for p in workers: p.terminate()
                        raise RuntimeError(f"Worker {rank} crashed.")
                    if isinstance(res, dict) and 'error' in res:
                        for p in workers: p.terminate()
                        raise RuntimeError(f"Worker {rank} crashed: {res['error']}")
                    batch_results.append(res)
                
                # D. 数据处理与 Elo 更新
                batch_total_steps = 0
                batch_return = 0
                batch_wins = 0
                
                for res in batch_results:
                    l_tr = res['trans']
                    metrics = res['metrics']
                    opp_name = res['opp_name']
                    
                    batch_total_steps += metrics['steps']
                    batch_return += metrics['return']
                    if metrics['win']: batch_wins += 1
                    
                    for k in transition_dict: transition_dict[k].extend(l_tr[k])
                    
                    # Update Elo for this member
                    actual_score = 1.0 if metrics['win'] else (0.0 if metrics['lose'] else 0.5)
                    if opp_name in elo_ratings:
                        adv_elo = elo_ratings[opp_name]
                        member.elo = update_elo(member.elo, adv_elo, actual_score, K_FACTOR)
                        # 如果是历史池里的对手，也更新一下它的分
                        new_adv_elo = update_elo(adv_elo, member.elo, 1.0 - actual_score, K_FACTOR)
                        elo_ratings[opp_name] = new_adv_elo
                        if opp_name in elite_elo_ratings:
                             elite_elo_ratings[opp_name] = new_adv_elo
                # [新增] 在 PPO 更新前打印本轮详细战况
                if batch_idx % 1 == 0:
                    print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

                # 更新全局计数
                total_steps += batch_total_steps
                
                # E. 模型更新 (Training)
                if len(transition_dict['dones']) >= transition_dict_threshold:
                    transition_dict['actions'] = restructure_actions(transition_dict['actions'])
                    
                    # Teacher Selection for Distillation
                    # 种群成员依然共用外部的 Teacher Pool
                    teacher_name = None
                    should_distil = False
                    
                    # 简化版 Teacher 选取
                    rein_keys = [k for k in elite_elo_ratings.keys() if k.startswith("actor_rein")]
                    target_rules = ['Rule_3', 'Rule_4']
                    valid_rules = [r for r in target_rules if r in elite_elo_ratings]
                    candidate_keys = rein_keys + valid_rules
                    
                    if candidate_keys:
                         candidate_elos = np.array([elite_elo_ratings[k] for k in candidate_keys], dtype=np.float64)
                         # 简单的概率分布
                         probs = np.exp((candidate_elos - np.min(candidate_elos))/100)
                         probs /= np.sum(probs)
                         teacher_name = np.random.choice(candidate_keys, p=probs)
                    
                    if teacher_name:
                        teacher_agent.agent_info = None; teacher_agent.critic_info = None
                        if teacher_name.startswith("actor_rein"):
                            model_path = os.path.join(log_dir, f"{teacher_name}.pt")
                            if os.path.exists(model_path):
                                teacher_actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=1))
                                teacher_agent.agent_info = ('NN', teacher_actor)
                                should_distil = True
                        elif teacher_name.startswith('Rule'):
                            match = re.search(r'\d+', teacher_name)
                            if match:
                                teacher_agent.agent_info = ('rule', int(match.group()))
                                should_distil = True
                    
                    # 先做基础 PPO
                    member.agent.update(transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed)
                    
                    # 再做蒸馏 (使用该 Member 自己的 alpha_il)
                    if use_sil and should_distil:
                        # PBT 核心: 使用 member.alpha_il
                        member.agent.distil(transition_dict, teacher_agent=teacher_agent,
                                           alpha=member.alpha_il, distil_only_maneuver=distil_only_maneuver,
                                           shuffled=1, mini_batch_size=mini_batch_size_mixed, reverse_kl=reverse_kl)

                    # Member Logging
                    prefix = f"P{member.id}"
                    logger.add(f"{prefix}/return", batch_return/num_workers, total_steps)
                    logger.add(f"{prefix}/win_rate", batch_wins/num_workers, total_steps)
                    logger.add(f"{prefix}/elo", member.elo, total_steps)
                    logger.add(f"{prefix}/actor_loss", member.agent.actor_loss, total_steps)
            
            # End of Population Generation
            pbt_generation_cnt += 1
            batch_idx += 1
            
            print(f"Gen {pbt_generation_cnt}: Total Steps {total_steps}. Best Elo: {max([m.elo for m in population]):.0f}")

            # ==================================================================
            # PBT Evolution Step (Exploit & Explore)
            # ==================================================================
            if pbt_generation_cnt % pbt_interval == 0:
                print(f"\n>>> Executing PBT Evolution at Gen {pbt_generation_cnt}...")
                
                # 1. Sort by Elo
                sorted_pop = sorted(population, key=lambda m: m.elo)
                best_member = sorted_pop[-1]
                worst_member = sorted_pop[0] # 这里只淘汰最差的一个，也可以淘汰后25%
                
                print(f"  Best: P{best_member.id} (Elo {best_member.elo:.0f}, Alpha {best_member.alpha_il:.5f})")
                print(f"  Worst: P{worst_member.id} (Elo {worst_member.elo:.0f}, Alpha {worst_member.alpha_il:.5f})")
                
                # 2. Exploit: Worst copies Best
                worst_member.agent.actor.load_state_dict(best_member.agent.actor.state_dict())
                worst_member.agent.critic.load_state_dict(best_member.agent.critic.state_dict())
                worst_member.elo = best_member.elo # 继承分数
                
                # 3. Momentum Reset: Clear optimizer state
                worst_member.reset_optimizer()
                
                # 4. Explore: Mutate Hyperparams (Alpha_IL)
                # 扰动: x 0.8 or x 1.2
                perturb = np.random.choice([0.8, 1.2])
                new_alpha = best_member.alpha_il * perturb
                # Clip alpha to reasonable bounds
                new_alpha = np.clip(new_alpha, 1e-6, 1.0)
                worst_member.alpha_il = new_alpha
                
                print(f"  -> P{worst_member.id} mutated: New Alpha {worst_member.alpha_il:.5f}")
                
                # Log PBT Stats
                logger.add("PBT/Best_Elo", best_member.elo, total_steps)
                logger.add("PBT/Best_Alpha", best_member.alpha_il, total_steps)

            # ==================================================================
            # 保存与历史池维护 (使用 Best Member)
            # ==================================================================
            if batch_idx % save_interval == 0:
                best_member = max(population, key=lambda m: m.elo)
                
                actor_key = f"actor_rein{batch_idx}_P{best_member.id}"
                critic_key = f"critic_rein{batch_idx}_P{best_member.id}"
                
                torch.save(best_member.agent.actor.state_dict(), os.path.join(log_dir, f"{actor_key}.pt"))
                torch.save(best_member.agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                
                # Elo json update
                main_agent_elo = best_member.elo # 记录当前最强
                elo_ratings[actor_key] = main_agent_elo
                elo_ratings["__CURRENT_MAIN__"] = main_agent_elo
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                
                # Admission to Elite Pool
                if hist_agent_as_opponent and total_steps >= WARM_UP_STEPS:
                     # 简单的 Admission 逻辑，直接看是不是比池子里的一半强
                     valid_elos = [v for k, v in elite_elo_ratings.items() if not k.startswith("__")]
                     if not valid_elos: valid_elos = [1200]
                     avg_elo = np.mean(valid_elos)
                     
                     if best_member.elo > avg_elo * 0.95: # 稍微放宽一点
                         # Cleanup
                         history_keys = [k for k in elite_elo_ratings.keys() if k.startswith("actor_rein")]
                         if len(history_keys) >= MAX_HISTORY_SIZE:
                             weakest = min(history_keys, key=lambda k: elite_elo_ratings[k])
                             del elite_elo_ratings[weakest]
                         
                         elite_elo_ratings[actor_key] = best_member.elo
                         print(f"Saved Checkpoint {actor_key} (Elo {main_agent_elo:.0f}) to Elite Pool.")

                # Save JSONs
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
                
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                        
        # --- [新增] 达到 max_steps 后的交互逻辑 ---
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        print("All simulation workers are now idling safely. System is paused.") 
        
        inp = input(f"Enter new max_steps (current {total_steps}), or press Enter to exit: ")
        
        if not inp.strip():
            print("No input provided. Exiting training.")
            break # 退出外层 while True 循环
        try:
            new_max = int(inp)
            if new_max > total_steps:
                current_max_steps = new_max
                print(f"Continuing training until {current_max_steps} steps.")
            else:
                print("Input steps less than current. Exiting...")
                break # 退出外层 while True 循环
        except ValueError:
            print("Invalid input (not a number). Exiting...")
            break

    # Cleanup
    print("Closing workers...")
    for pipe in pipes:
        try: # 【新增】防止管道已断开导致的报错
            pipe.send(('EXIT', None))
        except:
            pass
            
    for p in workers:
        p.join(timeout=5) # 【修改】给子进程 5 秒优雅退出的时间
        if p.is_alive():
            p.terminate() # 如果没死，强制结束
    
    # 【修改】调整顺序：先尝试关闭测试池
    try:
        test_pool.close()
        test_pool.join()
    except:
        pass
    
    logger.close()
    print("Training Finished.")