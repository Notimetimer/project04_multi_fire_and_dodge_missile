'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

from typing import final
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
import collections
# 必须在任何 sklearn 导入之前执行！
try:
    import threadpoolctl
    # 彻底让信息查询返回空列表
    threadpoolctl.threadpool_info = lambda *args, **kwargs: []
    # 核心修复：直接将 threadpool_limits 变为一个什么都不做的空上下文管理器
    class DummyContextManager:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def _set_threadpool_limits(self): return []
    threadpoolctl.threadpool_limits = DummyContextManager
    # 针对 3.x 版本的控制器拦截
    if hasattr(threadpoolctl, 'ThreadpoolController'):
        threadpoolctl.ThreadpoolController.info = lambda self, *args, **kwargs: []
        threadpoolctl.ThreadpoolController.limit = lambda self, *args, **kwargs: DummyContextManager()
    print("[Patch] threadpoolctl 全版本上下文拦截补丁已成功强行注入。")
except Exception as e:
    print(f"[Patch] 补丁注入失败: {e}，尝试继续运行...")
from sklearn.cluster import KMeans

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new_hierarchical import *
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 奖励函数
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper, infer_mask_cfg_from_actor_meta
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from VsBaseline_while_training_hierarch_plus import test_worker
from RewardWeightController import FireRewardWeightController

dt_move = 0.04

# 测试对战初始弹药（可调）
red_init_ammo = 6
blue_init_ammo = 6

# 周期性测试所对抗的规则对手编号 (仅测试 0, 1, 2, 3)
TEST_RULE_IDS = [0, 1, 2, 3]
# 蒸馏/引导用的教师规则列表（排除 Rule 0，避免其在上一枚导弹中制导时仍开火的激进行为）
TEACHER_RULE_IDS = [1, 2, 3, 4, 5, 6]


class RuleTeacherWrapper:
    """
    基于规则(basic_rules)的教师包装器，用于 RDistill。

    对齐 PolicyNetHybrid.forward 的输出格式：
      - 'cat': list[Tensor]，每个动作头一个软 one-hot 概率分布 (B, n_classes)
      - 'bern': Tensor，开火头 logits (B, 1)
      - 'cont': None
    软 one-hot 构造方式与 PPOHybrid.compute_il_loss 的 pre_training 分支一致：
      正确动作概率 = 1 - label_smoothing，其余均分 label_smoothing/(n-1)。

    通过 env.obs2obs_check + env.unscale_state 把网络局部观测(obs)还原为
    basic_rules 所需的 state_check，再逐样本调用规则得到 (action, fire)。
    """
    is_rule_teacher = True

    def __init__(self, env, rule_num, action_dims_dict, device, label_smoothing=0.1):
        self.env = env
        self.rule_num = rule_num
        self.action_dims_dict = action_dims_dict
        self.device = device
        self.label_smoothing = label_smoothing
        cat_dims = action_dims_dict.get('cat', [])
        # 兼容 int / list 两种写法
        self.cat_dims = list(cat_dims) if isinstance(cat_dims, (list, tuple)) else [cat_dims]

    def eval(self):
        return self

    @torch.no_grad()
    def predict_distributions(self, states):
        obs_np = states.detach().cpu().numpy()
        B = obs_np.shape[0]
        ls = self.label_smoothing
        n_heads = len(self.cat_dims)

        # 初始化为平滑背景概率
        cat_probs = [np.full((B, nc), ls / max(nc - 1, 1), dtype=np.float32) for nc in self.cat_dims]
        bern_logits = np.full((B, 1), -3.0, dtype=np.float32)

        for i in range(B):
            check_obs = self.env.obs2obs_check(obs_np[i])
            state_check = self.env.unscale_state(check_obs)
            action_number, fire = basic_rules(state_check, self.rule_num, p_random=0)

            if isinstance(action_number, (list, tuple, np.ndarray)):
                act_list = list(np.asarray(action_number).reshape(-1))
            else:
                act_list = [action_number]

            for h in range(n_heads):
                nc = self.cat_dims[h]
                idx = int(act_list[h]) if h < len(act_list) else 0
                idx = int(np.clip(idx, 0, nc - 1))
                cat_probs[h][i, :] = ls / max(nc - 1, 1)
                cat_probs[h][i, idx] = 1.0 - ls

            bern_logits[i, 0] = 3.0 if fire else -3.0

        cat_tensors = [torch.tensor(cp, dtype=torch.float, device=self.device) for cp in cat_probs]
        bern_tensor = torch.tensor(bern_logits, dtype=torch.float, device=self.device)
        return {'cont': None, 'cat': cat_tensors, 'bern': bern_tensor}

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
                # 优先找 'fly'，找不到找 'cat'
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
    td['states'].append(state) # 修改：这里存储Critic用的全局状态
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    td['active_masks'].append(active_mask) # 【新增】active_mask，转入多智能体
    return td

# ==========================================
# 新增：混合缓冲区类
# ==========================================
class IL_transition_buffer:
    def __init__(self, init_dict, max_size=10000):
        """
        内部存储全部采用 List，确保状态和动作在最外层长度绝对对等。
        """
        self.max_size = max_size
        self.addon_dict = {}
        
        # 无论 init_dict 是否为空，都显式初始化所有键，确保 'obs' 一定存在
        src = init_dict if init_dict is not None else {}
        
        # 强制转换成 list 存储。如果 src 里没有 'obs'，就用 'states' 代替
        self.addon_dict['obs'] = list(src.get('obs', src.get('states', [])))
        self.addon_dict['states'] = list(src.get('states', []))
        self.addon_dict['returns'] = list(src.get('returns', []))
        self.addon_dict['actions'] = list(src.get('actions', []))
        
        # 特殊处理：如果 returns 是 torch.tensor (如你截图中所示)，转为 list 存储
        if torch.is_tensor(src.get('returns')):
            self.addon_dict['returns'] = src['returns'].tolist()

        # 打印初始长度以供检查
        print(f"[IL_transition_buffer] Initialized. Size: {len(self.addon_dict['states'])}")
        
    def add(self, data):
        """
        data: 包含 'obs', 'states', 'actions', 'returns' 的字典，值应为 List。
        """
        # 1. 提取新数据并确保格式为 list (防止 data 缺失 'obs'，逻辑同 init)
        new_obs = list(data.get('obs', data.get('states', [])))
        new_states = list(data.get('states', []))
        new_returns = list(data.get('returns', []))
        new_actions = list(data.get('actions', []))
        
        # 处理可能传入的 tensor
        if torch.is_tensor(data.get('returns')):
            new_returns = data['returns'].tolist()
        
        # 使用 .extend() 拼接列表元素
        self.addon_dict['obs'].extend(new_obs)
        self.addon_dict['states'].extend(new_states)
        self.addon_dict['returns'].extend(new_returns)
        self.addon_dict['actions'].extend(new_actions)
        
        # 2. 基于添加顺序的剪裁（保留最后/最新的 max_size 条）
        current_len = len(self.addon_dict['states'])
        max_size_int = int(self.max_size)  # 确保 max_size 是整数
        if current_len > max_size_int:
            keep_from = current_len - max_size_int
            self.addon_dict['obs'] = self.addon_dict['obs'][keep_from:]
            self.addon_dict['states'] = self.addon_dict['states'][keep_from:]
            self.addon_dict['returns'] = self.addon_dict['returns'][keep_from:]
            self.addon_dict['actions'] = self.addon_dict['actions'][keep_from:]
    

    
    def read(self, batch_size):
        """
        随机采样并进行格式转换。
        """
        total_len = len(self.addon_dict['states'])
        if total_len == 0:
            raise ValueError("IL_transition_buffer is empty.")
            
        indices = np.random.randint(0, total_len, size=min(int(batch_size), total_len))
        
        # 采样（列表推导式，保持原始元素格式）
        sampled_obs = [self.addon_dict['obs'][i] for i in indices]
        sampled_states = [self.addon_dict['states'][i] for i in indices]
        sampled_returns = [self.addon_dict['returns'][i] for i in indices]
        sampled_actions_raw = [self.addon_dict['actions'][i] for i in indices]
        
        # 此时才将采样的动作列表转换为算法需要的 dict-of-arrays 格式
        return {
            'obs': np.array(sampled_obs, dtype=np.float32),
            'states': np.array(sampled_states, dtype=np.float32),
            'returns': np.array(sampled_returns, dtype=np.float32),
            'actions': restructure_actions(sampled_actions_raw) 
        }

    def clear(self):
        for k in self.addon_dict:
            self.addon_dict[k] = []
        print("[IL_transition_buffer] Buffer cleared.")

    def save(self, path):
        """直接序列化整个 buffer 实例"""
        torch.save(self, path)
        print(f"[IL_transition_buffer] Saved to {path}. Size: {len(self.addon_dict['states'])}")

    @staticmethod # 静态方法调用,不需要实例化
    def load(path):
        """从磁盘加载 buffer 实例"""
        if not os.path.exists(path):
            return None
        buffer = torch.load(path, map_location='cpu')
        print(f"[IL_transition_buffer] Loaded from {path}. Size: {len(buffer.addon_dict['states'])}")
        return buffer

# calculate_expected_score 概率计算
from SelfPlayCalculation import calculate_expected_score, update_elo, get_opponent_probabilities

# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                8000.0 * (1-int(randomized))
    if np.random.uniform(0,1) < 0.2 and int(randomized): # 以很低概率在更大的范围随机初始高度
        blue_height = np.random.uniform(2000.0, 12000.0)
    red_height = blue_height
    # 初始航向随机化
    red_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-pi*2/3, pi*2/3)) # -pi/3, pi/3
    blue_psi = sub_of_radian(np.pi/2 + np.random.uniform(-pi*2/3, pi*2/3))
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N = init_North
    red_E = 55e3 # 45e3
    if np.random.uniform(0,1) < 0.2: # 以低概率随机初始距离
        red_E = np.random.uniform(10e3,55e3)
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
                   seed, opp_greedy_rate, dt_move=0.05, no_crash=1, pomdp=1, vertices=None, mask_cfg=None):
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

        # args.R_cage = np.random.uniform(30e3, 45e3) # 已移除：放在这里会导致同一个Worker的所有episode的环境大小不变

        # 初始化环境 (关闭可视化以加速)
        env = ChooseStrategyEnv(args, tacview_show=False, vertices=vertices)
        env.shielded = no_crash # 假设默认开启防撞
        env.dt_move = dt_move
        env.dt_maneuver = dt_maneuver
        
        env.no_out = 0 # 训练时该出界必须出界

        # 初始化本地网络 (CPU)
        # Worker 仅做推理：直接用 HybridActorWrapper（SAC 与 PPO 共用同一套 actor 接口），无需构建完整 SAC/Q 网络
        local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device_worker)
        local_agent = HybridActorWrapper(local_actor, action_dims_dict, None, device_worker).to(device_worker)
        
        # 初始化对手网络
        adv_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device_worker)
        adv_agent = HybridActorWrapper(adv_actor, action_dims_dict, None, device_worker).to(device_worker)

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
                local_agent.load_state_dict(actor_weights)
                
                # B. 配置对手
                opp_name, opp_type, opp_data, opp_temperature = opponent_info
                adv_is_rule = (opp_type == 'rule')
                rule_num = 0
                if adv_is_rule:
                    rule_num = opp_data
                else:
                    adv_agent.load_state_dict(opp_data)

                # C. 准备本回合容器
                # Worker 收集完整的 ego_trans (用于 SIL) 和 enm_trans (用于 SIL)
                # local_trans 用于 PPO 更新 (只包含 Blue 视角)
                local_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                ego_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                enm_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}

                # 新增: 开火角度参数及导弹存活期 ATA 的回合统计容器
                # 红方（对手）统计
                episode_red_fire_thetas = []      # 开火瞬间俯仰角
                episode_red_ATAs = []             # 开火后30s的ATA
                episode_red_delta_psi_threats = [] # 收到告警后的delta_psi_threat
                episode_red_delta_thetas = []     # 开火后30s内的delta_theta
                episode_red_delta_psis = []       # 开火后30s内的delta_psi
                # 蓝方（本方学习代理）统计
                episode_blue_fire_thetas = []     # 开火瞬间俯仰角
                episode_blue_ATAs = []            # 开火后30s的ATA
                episode_blue_delta_psi_threats = [] # 收到告警后的delta_psi_threat
                episode_blue_delta_thetas = []    # 开火后30s内的delta_theta
                episode_blue_delta_psis = []      # 开火后30s内的delta_psi
                
                # 新增: 开火策略指标统计容器
                # 蓝方（本方）开火策略指标
                episode_blue_fire_intervals = []  # 开火间隔时长
                episode_blue_fire_delta_psis = [] # 开火瞬间的abs(delta_psi)
                episode_blue_fire_distances = []  # 开火距离
                episode_blue_fire_AA_hors = []    # 开火瞬间的abs(AA_hor)
                episode_blue_fire_alts = []           # 开火瞬间高度
                
                # 记录上次开火时间用于计算间隔
                last_blue_fire_time = -1.0
                

                # D. 环境重置
                # randomized_birth = settings['randomized_birth']  # 改在外面随机，里面不需要
                action_cycle_multiplier = settings['action_cycle_multiplier']
                reward_weight = settings['weight_reward']
                # 在子环境中重新计算出生状态
                # red_birth, blue_birth = create_initial_state_worker(randomized_birth)
                # 使用从master传来的出生状态
                red_birth = settings['red_birth']
                blue_birth = settings['blue_birth']
                end_reward_weight = settings['end_reward_weight']
                
                # 每次重新运行对局前，根据Master指定的范围随机化当前环境大小
                # r_min, r_max = settings.get('R_cage_range', (55.00e3, 55.00e3))
                fire_mask = settings.get('fire_mask', 1)
                # env.R_cage = np.random.uniform(r_min, r_max)

                fire_inside_weight = settings.get('fire_inside_weight', None)
                fire_reward_weight = settings.get('fire_reward_weight', None)
                
                # 进场瞬间给全信息
                red_init_ammo=6
                blue_init_ammo=6
                # # 残局训练
                # if np.random.uniform(0,1) < 0.3:
                #     red_init_ammo = int(np.round(np.random.uniform(0,3)))
                #     blue_init_ammo = int(np.round(np.random.uniform(0,3)))
                env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo, pomdp=0)
                
                # 状态变量初始化
                done = False
                last_decision_obs, last_decision_state = None, None
                last_enm_decision_obs, last_enm_decision_state = None, None
                current_action, current_action_exec, current_enm_action_exec = None, None, None
                
                steps_run = 0
                episode_return = 0 # 仅用于统计显示
                episode_return_event = 0
                m_fired = 0
                
                dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}
                
                # 新增: 出界状态及首次出界时间记录（用于Elo惩罚）
                red_out_cage = False
                red_out_cage_time = None
                blue_out_cage = False
                blue_out_cage_time = None
                
                # --- E. 仿真循环 (核心物理逻辑) ---
                # 计算最大步数
                max_counts = int(args.max_episode_len / dt_maneuver)
                
                for count in range(max_counts):
                    if not env.running or done: break
                    
                    # 1. 获取观测
                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=pomdp)
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=pomdp)
                    b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    r_state_global, _ = env.obs_1v1('r', reward_fn=1)

                    # 收到告警后的delta_psi_threat - 红方（对手）
                    r_state_check = env.unscale_state(r_check_obs)
                    delta_psi_threat = np.arccos(r_check_obs["threat"][0])
                    if r_state_check["warning"]:
                        episode_red_delta_psi_threats.append(float(delta_psi_threat))
                    
                    # 收到告警后的delta_psi_threat - 蓝方（本方）
                    b_state_check = env.unscale_state(b_check_obs)
                    blue_delta_psi_threat = np.arccos(b_check_obs["threat"][0])
                    if b_state_check["warning"]:
                        episode_blue_delta_psi_threats.append(float(blue_delta_psi_threat))
                    
                    # 记录开火后30s内的角度数据（与ATA同级别）- 红方（对手）
                    if r_check_obs["weapon"] * 120 <= 30 and not r_state_check["warning"]:  # 有存活导弹（30s内）且没有告警
                        if "target_information" in r_state_check and len(r_state_check["target_information"]) > 0:
                            red_delta_psi = np.arctan2(r_state_check["target_information"][1], r_state_check["target_information"][0])
                            red_delta_theta = r_state_check["target_information"][2]
                            red_ATA = r_state_check["target_information"][4]
                            episode_red_ATAs.append(float(red_ATA))
                            episode_red_delta_thetas.append(float(red_delta_theta))
                            episode_red_delta_psis.append(float(red_delta_psi))
                    
                    # 记录开火后30s内的角度数据 - 蓝方（本方）
                    if b_check_obs["weapon"] * 120 <= 30 and not b_state_check["warning"]:  # 有存活导弹（30s内）且没有告警
                        if "target_information" in b_state_check and len(b_state_check["target_information"]) > 0:
                            blue_delta_psi = np.arctan2(b_state_check["target_information"][1], b_state_check["target_information"][0])
                            blue_delta_theta = b_state_check["target_information"][2]
                            blue_ATA = b_state_check["target_information"][4]
                            episode_blue_ATAs.append(float(blue_ATA))
                            episode_blue_delta_thetas.append(float(blue_delta_theta))
                            episode_blue_delta_psis.append(float(blue_delta_psi))


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
                            b_state_check = env.unscale_state(b_check_obs)
                            b_action_exec, _, _, _ = local_agent.get_action(b_obs, explore=1, mask_on=fire_mask)
                            # b_action_exec, _, _, _ = local_agent.get_action(b_obs, explore=1, check_obs=b_check_obs, mask_on=fire_mask) # 不建议采样也启用mask
                            b_action_label = b_action_exec['cat'] # [0]
                            b_fire = b_action_exec['bern'][0]
                            
                            # Red Decision
                            r_state_check = env.unscale_state(r_check_obs)
                            if adv_is_rule:
                                # 调用规则，假设 basic_rules 已导入
                                r_action_label, r_fire = basic_rules(r_state_check, rule_num, p_random=0.1)
                                r_action_exec = {'cat': r_action_label, 'bern': np.array([r_fire], dtype=np.float32)}
                            else:
                                # 随机决定本局对手是否开启探索
                                adv_explore = 1 if np.random.rand() > opp_greedy_rate else 0
                                r_action_exec, _, _, _ = adv_agent.get_action(r_obs, explore={'cont':0, 'cat':adv_explore, 'bern':1}, 
                                                        mask_on=fire_mask, temperature={'cat':opp_temperature, 'bern':1.0})
                                # r_action_exec, _, _, _ = adv_agent.get_action(r_obs, explore={'cont':0, 'cat':adv_explore, 'bern':1}, check_obs=r_check_obs, mask_on=fire_mask) # 不建议采样也启用mask
                                r_action_label = r_action_exec['cat'] #[0]
                                r_fire = r_action_exec['bern'][0]

                        # 2.4 处理开火 (改为置位标志，由后续物理循环尝试发射)
                        b_is_firing = 0
                        r_is_firing = 0
                        if b_fire: 
                            env.BUAV.about_to_fire = 1
                            b_is_firing = env.has_ammo_to_fire('b')
                        if r_fire: 
                            env.RUAV.about_to_fire = 1
                            r_is_firing = env.has_ammo_to_fire('r')
                        
                        # 2.5 记录当前动作供下一帧存储 (初值设为未发射，若后续周期内发射成功则更新)
                        current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                        current_action_exec = {'cat': b_action_exec['cat'], 'bern': np.array([b_is_firing])}
                        current_enm_action_exec = {'cat': r_action_exec['cat'], 'bern': np.array([r_is_firing])}

                    # 3. 物理步进与尝试发射
                     # 采样的时候如果限制动作次序，会妨碍“试错”，到测试时也必须开启  r_action_label  b_action_label None

                    r_action_label_fire=None
                    b_action_label_fire=None
                    
                    r_m_id = launch_missile_immediately(env, 'r', action_label=r_action_label_fire) if getattr(env.RUAV, 'about_to_fire', 0) else None
                    b_m_id = launch_missile_immediately(env, 'b', action_label=b_action_label_fire) if getattr(env.BUAV, 'about_to_fire', 0) else None
                    
                    if b_m_id: 
                        m_fired += 1
                        # 记录蓝方（本方）开火俯仰角
                        blue_fire_theta = float(env.BUAV.theta)
                        episode_blue_fire_thetas.append(blue_fire_theta)
                        
                        # 记录开火瞬间高度
                        blue_fire_alt = float(env.BUAV.alt)
                        episode_blue_fire_alts.append(blue_fire_alt)
                        
                        # 记录蓝方开火策略指标
                        current_time = steps_run * dt_maneuver
                        if last_blue_fire_time >= 0:
                            fire_interval = current_time - last_blue_fire_time
                            episode_blue_fire_intervals.append(fire_interval)
                        last_blue_fire_time = current_time
                        
                        # 记录开火瞬间的abs(delta_psi)、距离、AA_hor
                        if "target_information" in b_state_check and len(b_state_check["target_information"]) > 0:
                            # delta_psi: target_information[0]
                            fire_delta_psi = np.arccos(b_state_check["target_information"][0])
                            episode_blue_fire_delta_psis.append(float(fire_delta_psi))
                            
                            # 距离: target_information[1]
                            fire_distance = b_state_check["target_information"][3]
                            episode_blue_fire_distances.append(float(fire_distance))
                            
                            # AA_hor: target_information[6]
                            fire_AA_hor = b_state_check["target_information"][6]
                            episode_blue_fire_AA_hors.append(abs(float(fire_AA_hor)))
                        
                    if r_m_id is not None:
                        fire_theta = float(env.RUAV.theta)
                        episode_red_fire_thetas.append(fire_theta)
                    
                    # debug
                    if r_action_label[0] > 4:
                        print("数值超出范围", r_action_label[0], r_action_label[1])

                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    steps_run += 1
                    
                    # 新增: 记录红蓝双方首次出界时刻
                    if not red_out_cage and env.out_cage(env.RUAV):
                        red_out_cage = True
                        red_out_cage_time = env.t
                    if not blue_out_cage and env.out_cage(env.BUAV):
                        blue_out_cage = True
                        blue_out_cage_time = env.t
                    
                    # 4. 奖励计算
                    # 返回值: done, r_event1+r_shaping(训练), r_event1(纯事件), r_event2+r_shaping(固定±100终局)
                    done, b_reward1, b_reward2, b_reward3 = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, 
                                                            action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)
                    _, r_reward1, _, _ = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)

                    reward_for_learn = b_reward1
                    reward_for_enm = r_reward1
                    
                    if steps_run % action_cycle_multiplier == 0 or done:
                        episode_return += b_reward3 # 固定±100结果奖励的归一化回报，用于日志
                        episode_return_event += b_reward2 # 事件+结果奖励(r_event1)
                    
                    # 5. 存活更新 (用于 Done 标记)
                    next_b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    next_r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                    dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

                # --- End of Simulation Loop ---
                
                # 6. 存储最后一步经验 (Terminal State)
                # 强制做一次终局判定
                done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False, action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)
                
                if last_decision_state is not None:
                    append_experience(local_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(ego_trans, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(enm_trans, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, next_r_state_global, True, not dead_dict['r'])

                # --- 序列时空修正 (Credit Assignment Fix) ---死后时间压缩
                # 由于代理死亡后 active_mask 变为 False，回合结束时的同归于尽补偿等延迟奖励
                # 无法回传(会乘以 mask=0)。因此将死亡后产生的所有收益收束到最后一个存活步，并截断死亡后的冗余数据。
                def truncate_and_shift(td):
                    if not td or len(td.get('rewards', [])) == 0:
                        return td
                    
                    active = td['active_masks']
                    last_idx = -1
                    for i in range(len(active)-1, -1, -1):
                        if active[i]:
                            last_idx = i
                            break
                    
                    if last_idx != -1 and last_idx < len(td['rewards']) - 1:
                        # 收束后续所有收益到最后一个决策点
                        td['rewards'][last_idx] += sum(td['rewards'][last_idx+1:])
                        td['dones'][last_idx] = True # 将此步标识为事实上的终止步
                        # 不用传送同归于尽的最后一步next_states到先死瞬间，因为 td_target = rewards + self.gamma * next_vals * (1 - dones)
                        
                        # 截断死亡后的垫充帧
                        for k in td.keys():
                            if isinstance(td[k], list):
                                td[k] = td[k][:last_idx+1]
                    return td
                    
                # 执行死后奖励压缩
                local_trans = truncate_and_shift(local_trans)
                ego_trans = truncate_and_shift(ego_trans)
                enm_trans = truncate_and_shift(enm_trans)
                # ----------------------------------------------

                # 计算本回合红方开火与角度参数指标
                if len(episode_red_fire_thetas) > 0:
                    ep_avg_fire_theta = float(np.mean(episode_red_fire_thetas))
                else:
                    ep_avg_fire_theta = None
                    
                if len(episode_red_ATAs) > 0:
                    ep_avg_ATA = float(np.mean(episode_red_ATAs))
                else:
                    ep_avg_ATA = None
                    
                if len(episode_red_delta_psi_threats) > 0:
                    ep_avg_delta_psi_threat = float(np.mean(episode_red_delta_psi_threats))
                else:
                    ep_avg_delta_psi_threat = None
                    
                if len(episode_red_delta_thetas) > 0:
                    ep_avg_delta_theta = float(np.mean(episode_red_delta_thetas))
                else:
                    ep_avg_delta_theta = None
                    
                if len(episode_red_delta_psis) > 0:
                    ep_avg_delta_psi = float(np.mean(episode_red_delta_psis))
                else:
                    ep_avg_delta_psi = None

                # 计算本回合蓝方（本方）开火与角度参数指标
                if len(episode_blue_fire_thetas) > 0:
                    ep_blue_avg_fire_theta = float(np.mean(episode_blue_fire_thetas))
                else:
                    ep_blue_avg_fire_theta = None
                    
                if len(episode_blue_ATAs) > 0:
                    ep_blue_avg_ATA = float(np.mean(episode_blue_ATAs))
                else:
                    ep_blue_avg_ATA = None
                    
                if len(episode_blue_delta_psi_threats) > 0:
                    ep_blue_avg_delta_psi_threat = float(np.mean(episode_blue_delta_psi_threats))
                else:
                    ep_blue_avg_delta_psi_threat = None
                    
                if len(episode_blue_delta_thetas) > 0:
                    ep_blue_avg_delta_theta = float(np.mean(episode_blue_delta_thetas))
                else:
                    ep_blue_avg_delta_theta = None
                    
                if len(episode_blue_delta_psis) > 0:
                    ep_blue_avg_delta_psi = float(np.mean(episode_blue_delta_psis))
                else:
                    ep_blue_avg_delta_psi = None

                # 计算本回合蓝方（本方）开火策略指标
                if len(episode_blue_fire_intervals) > 0:
                    ep_blue_avg_fire_interval = float(np.mean(episode_blue_fire_intervals))
                else:
                    ep_blue_avg_fire_interval = None
                    
                if len(episode_blue_fire_delta_psis) > 0:
                    ep_blue_avg_fire_delta_psi = float(np.mean(episode_blue_fire_delta_psis))
                else:
                    ep_blue_avg_fire_delta_psi = None
                    
                if len(episode_blue_fire_distances) > 0:
                    ep_blue_avg_fire_distance = float(np.mean(episode_blue_fire_distances))
                else:
                    ep_blue_avg_fire_distance = None
                    
                if len(episode_blue_fire_AA_hors) > 0:
                    ep_blue_avg_fire_AA_hor = float(np.mean(episode_blue_fire_AA_hors))
                else:
                    ep_blue_avg_fire_AA_hor = None
                
                if len(episode_blue_fire_alts) > 0:
                    ep_blue_avg_fire_altitude = float(max(episode_blue_fire_alts)) # 不再记录平均开火高度，改为记录最大开火高度
                else:
                    ep_blue_avg_fire_altitude = None

                WVR = env.close_range_kill()
                BVR_perish_together = (not WVR) and env.draw

                # 7. 打包结果
                result_packet = {
                    'trans': local_trans, # 用于 RL Update
                    'ego_trans': ego_trans, # 用于 SIL (win)
                    'enm_trans': enm_trans, # 用于 SIL (lose)
                    # 命中结果由环境在回合结束时一次性写入，直接传回主进程。
                    'fire_hit_records': copy.deepcopy(getattr(env.BUAV, 'fire_hit_records', []) ), # +
                        # getattr(env.RUAV, 'fire_hit_records', [])),
                    'metrics': {
                        'return': episode_return,
                        'event_return': episode_return_event,
                        'steps': steps_run,
                        'win': env.win,
                        'lose': env.lose,
                        'draw': env.draw,
                        'm_fired': m_fired,
                        'BVR_perish_together': BVR_perish_together
                    },
                    'opp_name': opp_name,
                    # 新增: 本回合红方（对手）开火角度参数统计 [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                    'ep_avg_fire_theta': ep_avg_fire_theta,
                    'ep_avg_ATA': ep_avg_ATA,
                    'ep_avg_delta_psi_threat': ep_avg_delta_psi_threat,
                    'ep_avg_delta_theta': ep_avg_delta_theta,
                    'ep_avg_delta_psi': ep_avg_delta_psi,
                    # 新增: 本回合蓝方（本方）开火角度参数统计
                    'ep_blue_avg_fire_theta': ep_blue_avg_fire_theta,
                    'ep_blue_avg_ATA': ep_blue_avg_ATA,
                    'ep_blue_avg_delta_psi_threat': ep_blue_avg_delta_psi_threat,
                    'ep_blue_avg_delta_theta': ep_blue_avg_delta_theta,
                    'ep_blue_avg_delta_psi': ep_blue_avg_delta_psi,
                    # 新增: 本回合蓝方（本方）开火策略指标统计
                    'ep_blue_avg_fire_interval': ep_blue_avg_fire_interval,
                    'ep_blue_avg_fire_delta_psi': ep_blue_avg_fire_delta_psi,
                    'ep_blue_avg_fire_distance': ep_blue_avg_fire_distance,
                    'ep_blue_avg_fire_AA_hor': ep_blue_avg_fire_AA_hor,
                    'ep_blue_avg_fire_altitude': ep_blue_avg_fire_altitude,
                    # 新增: 回合结束时双方出界状态与首次出界时间
                    'red_out_cage': red_out_cage,
                    'red_out_cage_time': red_out_cage_time,
                    'blue_out_cage': blue_out_cage,
                    'blue_out_cage_time': blue_out_cage_time,
                }
                
                # 8. 发送回 Master
                pipe.send(result_packet)

    except Exception: # [新增] 异常捕获与回传
        print(f"!!! Worker {rank} CRASHED !!!")
        tb = traceback.format_exc()
        print(tb)
        try: pipe.send({'error': tb})
        except: pass
            


def run_MLP_simulation(
    k_nonlinear,
    collape_recover={
        "collapsed": False,
        "best_actor_name": None,
        "actor_frozen_batchs": 5,
    },
    num_workers=10, # 并行进程数，根据CPU核数调整，建议 10-20
    n_clusters=5,
    mission_name='无名',
    actor_lr=1e-4,
    critic_lr=5e-4,
    actor_lr_init_il = 1e-4,
    critic_lr_init_il = 5e-4,
    IL_epoches=180,
    max_steps_of_maneuver=4 * 165e4,
    hidden_dim=None,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=None,
    alpha_il=0.05,
    il_batch_size=128,
    il_batch_size2=None,
    il_buffer_max_size=2e4,
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
    max_episode_duration=15*60,
    R_cage = 62.00e3, # 45e3 # 55e3,
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
    use_sil = 0, # FIXME 再也不做自模仿了
    sil_only_maneuver = 1, # 自模仿只包含机动还是也包含开火
    chosen_quantile = 0.2, 
    DARK_SIDE = 1,  # sil默认找最差
    p_factor = None, # 无效接口
    sigma_elo = 400,
    WARM_UP_STEPS = 500e3,
    ADMISSION_THRESHOLD = 0.5,
    MAX_HISTORY_SIZE = 50, # 300 # 100
    deltaFSP_epsilon = 0.8,
    compete_old_rate = 0.2,
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
    update_interval_of_batch = 1, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
    opp_greedy_rate = 0, # 对手贪婪率
    num_runs = 3, # 测试回合重复次数
    device = torch.device("cpu"),
    max_il_exponent = -2.0,
    k_shape_il = 0.004,
    R_cage_range = (55.00e3, 55.00e3), # 新增：环境随机化范围
    vertices = None,
    resume_dir = None,
    init_il_data = None, # [新增] 从外部传入预拉取的数据集
    init_model_dir = None,
    init_actor_filename = None,
    init_critic_filename = None,
    POMDP = 0, # 0全信息，1部分信息
    should_stir = 0, # 是否搅拌策略参数后存储
    adj_r_w = 0, # 是否允许奖励函数权重浮动
    use_RND = 0, # 好奇心机制
    beta_RND = 0.3,
    use_ADistill = 0, # 温和蒸馏机制
    beta_ADistill = 0.1,
    AFiltered = 0, # 温和蒸馏是否需要优势滤波
    conf_thres = 0.7,
    bern_included = 1, # 开火也一起上
    adistill_anneal_factor = 0.5, # ADistill alpha 退火速度系数
    Bdistill = 0, # 是否启用Bdistill
    Bdistill_alpha = 1.0, # BernDistill alpha
    Bdistill_epochs = 3, # BernDistill epochs
    adistill_teacher_rule = 6, # ADistill 初始固定 teacher 的规则编号
    adistill_teacher_elo_steps = 1e6, # 总步数超过该值后，改用 Elo 最高的 Rule 作为 teacher
    no_bern_distill = 1, # 1: RDistill时不计算bern的KL散度
    distill_learn_type = "dual_prob", # RDistill奖励构造方式: dual_prob(师生KL) 或 single_prob(teacher对真实动作NLL)
    use_supervised_fire = 0, # [新增] 开火命中监督更新开关：仅用 fire_obs->hit_record 监督 fc_bern 头
    supervised_fire_buffer_size = 1000, # [新增] 开火记录滚动缓冲容量（FIFO，像off-policy一样滚动更新）
    supervised_fire_batch_size = 128, # [新增] 每次监督更新从缓冲中采样的批大小
    ppo_with_bern = 1, # [新增] 0 时 PPO 更新跳过开火头 fc_bern（log_prob 不计入 bern 项）
    sup_on_prob_ratio = 10, # 有监督辅助开关
):

    actor_lr0 = actor_lr
    critic_lr0 = critic_lr
    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # --- [修改] 灵活加载模仿学习数据集 ---
    if init_il_data is not None:
        print("Using externally provided IL dataset.")
        original_il_transition_dict = copy.deepcopy(init_il_data)
        original_il_transition_dict0 = copy.deepcopy(init_il_data)
    else:
        # 如果外面没传，则走老路子，从本地文件加载
        original_il_transition_dict, _ = load_il_and_transitions(
            os.path.join(cur_dir, "IL"),
            "il_transitions_combat_LR.pkl",
            "transition_dict_combat_LR.pkl"
        )
        original_il_transition_dict0 = copy.deepcopy(original_il_transition_dict)
    
    # 对加载/传入的数据进行必要的重构
    if original_il_transition_dict is not None:
        original_il_transition_dict['actions'] = restructure_actions(original_il_transition_dict['actions'])
        # 顺便确保 states 和 returns 也是标准的 float32 numpy array
        if 'states' in original_il_transition_dict:
            original_il_transition_dict['states'] = np.array(original_il_transition_dict['states'], dtype=np.float32)
        if 'returns' in original_il_transition_dict:
            original_il_transition_dict['returns'] = np.array(original_il_transition_dict['returns'], dtype=np.float32)
        print(f"IL dataset processed. Samples: {len(original_il_transition_dict['states'] if original_il_transition_dict['states'] is not None else [])}")
    
    # 2. 参数与环境配置 (Master 用于获取维度)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = ChooseStrategyEnv(args)
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    # 保留一个常驻 env 供规则教师(RuleTeacherWrapper)做 obs->check_obs 的还原（仅用其无状态的缩放方法）
    rule_teacher_env = copy.deepcopy(dummy_env)
    # ADistill/BernDistill 用：Rule_1 ~ Rule_6 共识集合（排除 Rule 0）
    adistill_rule_wrappers = [
        RuleTeacherWrapper(rule_teacher_env, rule_num=r,
                           action_dims_dict=action_dims_dict,
                           device=device,
                           label_smoothing=label_smoothing)
        for r in TEACHER_RULE_IDS
    ]
    # 保留单个 teacher_wrapper 仅用于日志 / 兼容其它旧逻辑
    teacher_rule_num_cur = adistill_teacher_rule
    teacher_wrapper = adistill_rule_wrappers[TEACHER_RULE_IDS.index(teacher_rule_num_cur)]
    del dummy_env

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Master training device: {device}")

    # 3. 创建神经网络
    # [新增] 如果续训，根据 actor.meta.json 自动推断当时的 ver/hor 配置，
    # 避免模型结构与 checkpoint 不匹配。
    mask_cfg = None
    if resume_dir is not None and os.path.exists(resume_dir):
        actor_meta_path = os.path.join(resume_dir, "actor.meta.json")
        if os.path.exists(actor_meta_path):
            mask_cfg = infer_mask_cfg_from_actor_meta(actor_meta_path)

    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)

    student_agent = PPOHybrid(
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
        max_std=label_smoothing,
        rnd_state_dim=state_dim,
    )
    
    # ADistill 用：cat 熵系数的基准值（在 agent 类属性上动态缩放）
    base_k_cat = student_agent.k_entropy.get('cat', 0.008)
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    if resume_dir is not None and os.path.exists(resume_dir):
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
        IL_epoches = 0  # 中断续训跳过预训练
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
    
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())

    # [新增] （训练方）蓝方开火策略指标的 EMA 变量（指数=0.2，即 1-0.8）
    ema_fire_interval = None # 50
    ema_fire_delta_psi = None # 30
    ema_fire_distance = None # 50e3
    ema_fire_AA_hor = None # 145
    ema_fire_altitude = None # 3e3
    ema_fire_theta = None # -5
    ema_ATA = None # 37
    ema_delta_psi_threat = None # 135
    ema_delta_theta = None # 4
    EMA_ALPHA = 0.2
    
    fire_inside_weight = None
    fire_reward_weight = None

    RWController = FireRewardWeightController(initial_fire_reward_weight=1.0)

    # 中断续训
    if resume_dir is not None and os.path.exists(resume_dir):
        if collape_recover["collapsed"]:
            best_actor_name = collape_recover["best_actor_name"]
            current_actor_path = os.path.join(log_dir, f"{best_actor_name}.pt")
        else:
            # 优先加载current_actor（确保不加载搅拌后的参数）
            current_actor_path = os.path.join(log_dir, "current_actor.pt")
            
        if os.path.exists(current_actor_path):
            student_agent.actor.load_state_dict(torch.load(current_actor_path, map_location=device))
            print(f"Loaded current actor from: {current_actor_path}")
        else:
            # 如果current_actor不存在，回退到原来的逻辑
            actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
            if len(actor_files) > 0:
                def extract_num(f):
                    m = re.search(r'actor_rein(\d+)\.pt$', f)
                    return int(m.group(1)) if m else -1
                latest_actor = max(actor_files, key=extract_num)
                student_agent.actor.load_state_dict(torch.load(latest_actor, map_location=device))
                print(f"Loaded actor from: {latest_actor}")
        
        critic_path = os.path.join(log_dir, "critic.pt")
        if os.path.exists(critic_path):
            student_agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
            print(f"Loaded critic from: {critic_path}")
        
        opt_path = os.path.join(log_dir, "optimizers_state.pt")
        if os.path.exists(opt_path):
            try:
                opt_states = torch.load(opt_path, map_location=device)
                student_agent.actor_optimizer.load_state_dict(opt_states['actor_optimizer'])
                student_agent.critic_optimizer.load_state_dict(opt_states['critic_optimizer'])
                print("Loaded optimizer states.")
                if student_agent.rnd_target is not None and 'rnd_target' in opt_states:
                    student_agent.rnd_target.load_state_dict(opt_states['rnd_target'])
                    print("Loaded RND target state.")
                if student_agent.rnd_prediction is not None and 'rnd_prediction' in opt_states:
                    student_agent.rnd_prediction.load_state_dict(opt_states['rnd_prediction'])
                    print("Loaded RND prediction state.")
                if student_agent.rnd_optimizer is not None and 'rnd_optimizer' in opt_states:
                    student_agent.rnd_optimizer.load_state_dict(opt_states['rnd_optimizer'])
                    print("Loaded RND optimizer state.")
            except Exception as e:
                print(f"Failed to load optimizers: {e}")
        
        # [新增] 恢复 special EMA 状态和控制器状态
        special_json_path = os.path.join(log_dir, "special.json")
        if os.path.exists(special_json_path):
            with open(special_json_path, "r", encoding="utf-8") as f:
                special_data = json.load(f)
            ema_fire_interval = special_data.get("ema_fire_interval", None)
            ema_fire_delta_psi = special_data.get("ema_fire_delta_psi", None)
            ema_fire_distance = special_data.get("ema_fire_distance", None)
            ema_fire_AA_hor = special_data.get("ema_fire_AA_hor", None)
            ema_fire_altitude = special_data.get("ema_fire_altitude", None)
            ema_fire_theta = special_data.get("ema_fire_theta", None)
            ema_ATA = special_data.get("ema_ATA", None)
            ema_delta_psi_threat = special_data.get("ema_delta_psi_threat", None)
            ema_delta_theta = special_data.get("ema_delta_theta", None)
            # [新增] 恢复控制器状态
            if "controller_state" in special_data:
                RWController.load_state_dict(special_data["controller_state"])
                print(f"Loaded controller state from: {special_json_path}")
            print(f"Loaded special EMA states from: {special_json_path}")
    
    init_model_requested = init_actor_filename is not None or init_critic_filename is not None
    if init_model_requested:
        if resume_dir is not None:
            raise ValueError("init model files cannot be used together with resume_dir")
        if init_model_dir is None or init_actor_filename is None or init_critic_filename is None:
            raise ValueError("init_model_dir, init_actor_filename and init_critic_filename must all be specified")
        init_actor_path = os.path.join(init_model_dir, init_actor_filename)
        init_critic_path = os.path.join(init_model_dir, init_critic_filename)
        if not os.path.isfile(init_actor_path):
            raise FileNotFoundError(f"Initial actor not found: {init_actor_path}")
        if not os.path.isfile(init_critic_path):
            raise FileNotFoundError(f"Initial critic not found: {init_critic_path}")
        student_agent.actor.load_state_dict(torch.load(init_actor_path, map_location=device))
        student_agent.critic.load_state_dict(torch.load(init_critic_path, map_location=device))
        IL_epoches = 0
        print(f"Loaded initial actor from: {init_actor_path}")
        print(f"Loaded initial critic from: {init_critic_path}")
        print("Skipping MARWIL pretraining and starting online training from the selected models.")

    # 保存onnx模型
    # 前提：假设此时 student_agent 已经创建好，且 state_dim 已经定义
    # 构建一个与 state 维度相同的 dummy input (batch_size=1)
    dummy_state = torch.randn(1, state_dim).to(device)
    # ==========================================
    # 1. 导出 Actor 的底层网络（PolicyNetHybrid）
    # ==========================================
    actor_onnx_path = os.path.join(log_dir, "student_actor.onnx")
    # 对于你的 PolicyNetHybrid，它返回的是一个 dict {'cont': ..., 'cat': ..., 'bern': ...}
    # 在高版本的 PyTorch 中，ONNX 对返回 dict 有支持（自动解包），或者你可以写一个简单的 wrapper 解包
    try:
        torch.onnx.export(
            student_agent.actor.net,           # 只导出纯网络结构，避开 Wrapper里的复杂采样操作
            dummy_state,                       # 伪造的输入状态
            actor_onnx_path,                   # 输出的文件名 / 路径
            export_params=True,                # 是否连同参数一起导出（选 True 可以看权重信息）
            opset_version=11,                  # 建议使用 11 或以上的算子集
            do_constant_folding=True,          # 是否执行常量折叠优化
            input_names=['state'],             # 命名的输入节点名称
            output_names=['cat_output', 'bern_output'] # 按照返回顺序手动指定名字
        )
        print(f"Actor ONNX successfully exported to {actor_onnx_path}")
    except Exception as e:
        print(f"Error exporting Actor ONNX: {e}")
    # ==========================================
    # 2. 导出 Critic 的底层网络（ValueNet）
    # ==========================================
    critic_onnx_path = os.path.join(log_dir, "student_critic.onnx")
    try:
        torch.onnx.export(
            student_agent.critic,              
            dummy_state,                       
            critic_onnx_path,                  
            export_params=True,                
            opset_version=11,                 
            do_constant_folding=True,          
            input_names=['state'],             
            output_names=['value_estimate']    # Critic 返回的一般是标量价值
        )
        print(f"Critic ONNX successfully exported to {critic_onnx_path}")
    except Exception as e:
        print(f"Error exporting Critic ONNX: {e}")

    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (Serial Execution on Master)
    if IL_epoches > 0:
        print("Start IL Training...")

    student_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    # === 模仿训练循环 ===
    # 现在 original_il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, # 显存如果够大可以适当调大
            label_smoothing=label_smoothing,
            no_bern = 0, # 0
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            # [新增] MARWIL 监控：每个动作头独立的 NLL 与策略熵 (全采样固定 batch, no_grad)
            for _name, _val in [
                ("il_train/nll_cont", getattr(student_agent, "marwil_nll_cont", None)),
                ("il_train/nll_cat", getattr(student_agent, "marwil_nll_cat", None)),
                # ("il_train/nll_bern", getattr(student_agent, "marwil_nll_bern", None)),
                ("il_train/entropy_cont", getattr(student_agent, "marwil_entropy_cont", None)),
                ("il_train/entropy_cat", getattr(student_agent, "marwil_entropy_cat", None)),
                # ("il_train/entropy_bern", getattr(student_agent, "marwil_entropy_bern", None)),
                ("il_train/accuracy_cont", getattr(student_agent, "marwil_accuracy_cont", None)),
                ("il_train/accuracy_cat", getattr(student_agent, "marwil_accuracy_cat", None)),
                ("il_train/accuracy_bern", getattr(student_agent, "marwil_accuracy_bern", None)),
                ("il_train/weight_mean", getattr(student_agent, "marwil_weight_mean", None)),
                # ("il_train/weight_max", getattr(student_agent, "marwil_weight_max", None)),
                # ("il_train/weight_min", getattr(student_agent, "marwil_weight_min", None)),
                ("il_train/weight_clip_frac", getattr(student_agent, "marwil_weight_clip_frac", None)),
                ("il_train/adv_std", getattr(student_agent, "marwil_adv_std", None)),
                # ("il_train/adv_p95", getattr(student_agent, "marwil_adv_p95", None)),
                ("il_train/adv_max", getattr(student_agent, "marwil_adv_max", None)),
                ("il_train/adv_mean", getattr(student_agent, "marwil_adv_mean", None)),
                ("il_train/adv_positive_frac", getattr(student_agent, "marwil_adv_positive_frac", None)),
            ]:
                if _val is not None:
                    logger.add(_name, _val, epoch)

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")
    
    if IL_epoches > 0:
        print("IL Training Finished.")
    else:
        print("No IL")
    
    # 存储在线训练前的网络参数
    int_agent_name = "actor_rein0"
    torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{int_agent_name}.pt"))
    if init_model_requested:
        torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, "current_actor.pt"))
        torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))


    # ==============================================================================
    # 强化学习 (Self-Play / PFSP) 阶段
    # ==============================================================================
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
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
    
    args.max_episode_len = max_episode_duration
    # args.R_cage = 45e3 # np.random.uniform(30e3, 45e3) # 环境大小随机化
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, 
                       kwargs={
                           'rank': i,
                           'pipe': child_conn,
                           'args': args,
                           'state_dim': state_dim,
                           'hidden_dim': hidden_dim,
                           'action_dims_dict': action_dims_dict,
                           'device_worker': worker_device,
                           'dt_maneuver': dt_maneuver,
                           'seed': seed,
                           'opp_greedy_rate': opp_greedy_rate,
                           'dt_move': dt_move,
                           'no_crash': no_crash,
                           'pomdp': POMDP,
                           'vertices': vertices,
                           'mask_cfg': mask_cfg,
                       })
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = {} # {copy.deepcopy(elo_ratings)}
    hall_of_fame = {}
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")
    # 新增：对手及精英开火与导弹期参数统计文件路径
    Elite_Fire_Stats_path = os.path.join(log_dir, "Elite_Fire_Stats.json")

    GameTimes = {}
    # 初始化开火与导弹期参数统计字典（不区分普通和精英）
    Elite_Fire_Stats = {}

    # 尝试加载历史 # 中断续训
    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f: elo_ratings = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f: hall_of_fame = json.load(f)
    if os.path.exists(Elite_Fire_Stats_path):
        with open(Elite_Fire_Stats_path, 'r', encoding='utf-8') as f: Elite_Fire_Stats = json.load(f)

    main_agent_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)

    
    # 不论如何，记录在线训练前的网络参数
    if (not elite_elo_ratings): # 如果是从零开始训练的
        # 初始对手Elo一律当1200
        for k in init_elo_ratings.keys():
            elo_ratings[k] = main_agent_elo

        # 自博弈开启时的初始分值填充
        if hist_agent_as_opponent:
            elo_ratings[int_agent_name] = main_agent_elo
        
        # 从零开始不论对手有多烂都要加入Elite池
        elite_elo_ratings = copy.deepcopy(elo_ratings)
        
        # 初始化GameTimes表与Elite_Fire_Stats表
        for k in elo_ratings.keys():
            GameTimes[k] = 0
            Elite_Fire_Stats[k] = [0.0, 0.0, 0.0, 0.0, 0.0]  # [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
        # 初始化主代理（蓝方）的fire_stats
        Elite_Fire_Stats["__CURRENT_MAIN__"] = [0.0, 0.0, 0.0, 0.0, 0.0]  # [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]


    # 训练循环变量
    total_steps_of_maneuver = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    batch_idx = elo_ratings.get("__LAST_UPDATE_BATCH__", 0)
    last_il_update_batch_idx = batch_idx
    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
        student_agent.reset_optimizer() # 恢复训练清除动量
    else:
        actor_freeze_until = -1
    trigger = trigger0 + (total_steps_of_maneuver // trigger_delta) * trigger_delta
    
    current_max_steps_of_maneuver = int(max_steps_of_maneuver)
    
    # 全局 Buffer (用于攒够 Batch 训练)
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    transition_dict = copy.deepcopy(empty_transition_dict)

    # [新增] 开火记录滚动缓冲：跨 batch 累积（FIFO 逐条覆盖），不随 transition_dict 清空重记录
    fire_records_buffer = collections.deque(maxlen=int(supervised_fire_buffer_size))

    # [新增] 若中断续训，恢复已保存的开火监督 buffer
    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
    if os.path.exists(fire_buffer_path):
        try:
            with open(fire_buffer_path, "rb") as f:
                fb_data = pickle.load(f)
            if isinstance(fb_data, dict):
                loaded_records = fb_data.get("records", [])
                loaded_maxlen = fb_data.get("maxlen", int(supervised_fire_buffer_size))
            elif isinstance(fb_data, collections.deque):
                loaded_records = list(fb_data)
                loaded_maxlen = fb_data.maxlen
            else:
                loaded_records = list(fb_data)
                loaded_maxlen = int(supervised_fire_buffer_size)
            fire_records_buffer.clear()
            fire_records_buffer.extend(loaded_records[-loaded_maxlen:])
            print(f"Loaded fire records buffer from {fire_buffer_path}. Size: {len(fire_records_buffer)}")
        except Exception as e:
            print(f"Failed to load fire records buffer: {e}")

    # 初始化基于胜率的在线 EMA 变量
    ema_score = 0.5
    ema_step = 0
    target_p1 = 0.65
    ppo_grad_ema = None  # [新增] 初始化 PPO 梯度 EMA 缓存
    rnd_mse = None       # RND 原始 MSE（上一批次的值，首批为 None）

    # =========================================================
    # 主循环 (Master Process)
    # =========================================================
    while True:
        while total_steps_of_maneuver < current_max_steps_of_maneuver:
            if total_steps_of_maneuver < 5e3:
                fire_mask = 0 # 0 # 全程开启开火mask
            else:
                fire_mask = 0
            # --- 【修改】同步并行测试阶段 ---
            # 只有测试跑完并处理完名人堂，才进入下一步的采样和仿真
            # --- 1. 并行测试触发逻辑 (Async) ---
            if total_steps_of_maneuver >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps_of_maneuver}...")
                # 1. 深度拷贝当前 Actor 权重到 CPU 内存
                current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                test_tasks = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        test_worker, 
                        # args=(current_weights, r_idx, args, 
                        #     state_dim, hidden_dim, action_dims_dict, 
                        #     dt_maneuver, 'cpu', num_runs, action_cycle_multiplier)
                        kwds={
                            'model_state_dict': current_weights,
                            'rule_num': r_idx,
                            'env_args': args,
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dims_dict': action_dims_dict,
                            'dt_maneuver_val': dt_maneuver,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'no_out': 0,  # 这里可以根据需要设为 1
                            'deterministic': False,
                            'restrict_fire': True, # False, 和采样保持一致
                            'vertices': vertices,
                            'red_init_ammo': red_init_ammo,
                            'blue_init_ammo': blue_init_ammo,
                        }
                    )
                    test_tasks.append(obj)
                
                # 第二种形式：追加额外测试 (机动动作确定化 + 动作次序限制打开)
                test_tasks_no_random = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        test_worker, 
                        kwds={
                            'model_state_dict': current_weights,
                            'rule_num': r_idx,
                            'env_args': args,
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dims_dict': action_dims_dict,
                            'dt_maneuver_val': dt_maneuver,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'no_out': 0,
                            'deterministic': True,     # 机动动作确定化
                            'restrict_fire': True,      # 动作次序限制打开
                            'vertices': vertices,
                            'red_init_ammo': red_init_ammo,
                            'blue_init_ammo': blue_init_ammo,
                        }
                    )
                    test_tasks_no_random.append(obj)

                # 等待所有测试进程结束
                test_results = [t.get() for t in test_tasks]
                test_results_no_random = [t.get() for t in test_tasks_no_random]

                # 记录第一种测试结果
                outcomes = {rule_num: score for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_return = {rule_num: result2 for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_wins = {rule_num: wins for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_loses = {rule_num: loses for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_draws = {rule_num: draws for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_perish = {rule_num: p_t_ for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps_of_maneuver)
                    logger.add(f"test/agent_vs_rule{r_num}_win_rate", outcomes_wins[r_num], total_steps_of_maneuver)
                    logger.add(f"test/agent_vs_rule{r_num}_lose_rate", outcomes_loses[r_num], total_steps_of_maneuver)
                    logger.add(f"test/agent_vs_rule{r_num}_draw_rate", outcomes_draws[r_num], total_steps_of_maneuver)
                    # logger.add(f"test/agent_vs_rule{r_num}_return", outcomes_return[r_num], total_steps_of_maneuver)
                    print(f"  [Test Result] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

                # 记录第一种测试的平均指标（所有对手的平均Score、胜负平比例和超视距双杀率）
                avg_score = np.mean(list(outcomes.values()))
                avg_win_rate = np.mean(list(outcomes_wins.values()))
                avg_lose_rate = np.mean(list(outcomes_loses.values()))
                avg_draw_rate = np.mean(list(outcomes_draws.values()))
                avg_perish_together = np.mean(list(outcomes_perish.values()))
                logger.add("test/avg_score", avg_score, total_steps_of_maneuver)
                logger.add("test/avg_win_rate", avg_win_rate, total_steps_of_maneuver)
                logger.add("test/avg_lose_rate", avg_lose_rate, total_steps_of_maneuver)
                logger.add("test/avg_draw_rate", avg_draw_rate, total_steps_of_maneuver)
                logger.add("test/BVR perish together", avg_perish_together, total_steps_of_maneuver)

                # 记录第二种测试结果 (test_No_random)
                outcomes_nr = {rule_num: score for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_return_nr = {rule_num: result2 for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_wins_nr = {rule_num: wins for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_loses_nr = {rule_num: loses for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_draws_nr = {rule_num: draws for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_perish_nr = {rule_num: p_t_ for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}

                for r_num, score in outcomes_nr.items():
                    logger.add(f"test_No_random/agent_vs_rule{r_num}", score, total_steps_of_maneuver)
                    logger.add(f"test_No_random/agent_vs_rule{r_num}_win_rate", outcomes_wins_nr[r_num], total_steps_of_maneuver)
                    logger.add(f"test_No_random/agent_vs_rule{r_num}_lose_rate", outcomes_loses_nr[r_num], total_steps_of_maneuver)
                    logger.add(f"test_No_random/agent_vs_rule{r_num}_draw_rate", outcomes_draws_nr[r_num], total_steps_of_maneuver)
                    # logger.add(f"test_No_random/agent_vs_rule{r_num}_return", outcomes_return_nr[r_num], total_steps_of_maneuver)
                    print(f"  [Test No Random] Rule_{r_num}: {score} (return: {outcomes_return_nr[r_num]:.2f})")

                # 记录第二种测试的平均指标（所有对手的平均Score、胜负平比例和超视距双杀率）
                avg_score_nr = np.mean(list(outcomes_nr.values()))
                avg_win_rate_nr = np.mean(list(outcomes_wins_nr.values()))
                avg_lose_rate_nr = np.mean(list(outcomes_loses_nr.values()))
                avg_draw_rate_nr = np.mean(list(outcomes_draws_nr.values()))
                avg_perish_together_nr = np.mean(list(outcomes_perish_nr.values()))
                logger.add("test_No_random/avg_score", avg_score_nr, total_steps_of_maneuver)
                logger.add("test_No_random/avg_win_rate", avg_win_rate_nr, total_steps_of_maneuver)
                logger.add("test_No_random/avg_lose_rate", avg_lose_rate_nr, total_steps_of_maneuver)
                logger.add("test_No_random/avg_draw_rate", avg_draw_rate_nr, total_steps_of_maneuver)
                logger.add("test_No_random/BVR perish together", avg_perish_together_nr, total_steps_of_maneuver)

                # 名人堂判定：如果全胜则保存并加入池子
                if all(score > 0.5 for score in outcomes_nr.values()): # 原先为outcomes.values()
                    # 【核心修改】从全量注册表 elo_ratings 中寻找最新的已保存编号
                    rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+$', k)]
                    
                    if rein_keys:
                        # 找到数值最大的编号（即最新的已保存智能体）
                        hof_key = max(rein_keys, key=lambda k: int(k.replace('actor_rein', '')))
                        
                        if hof_key not in hall_of_fame: # 只能判断key，不能判断value
                            # 存入字典，分数优先取全量表中的记录
                            hall_of_fame[hof_key] = elo_ratings.get(hof_key, main_agent_elo)
                            print(f"!!! [Hall of Fame] New Hero Captured: {hof_key}")
                
                trigger += trigger_delta

            # --- 2. 准备训练 Batch (Synchronous) ---
            # 改变环境奖励权重，超过100轮采样再更新权重，每次权重维持5轮采样
            if adj_r_w and batch_idx > 10:
                fire_inside_weight = None
                fire_reward_weight = None
                # fire_inside_weight, fire_reward_weight = RWController.update({
                #     'ema_fire_interval': ema_fire_interval,
                #     'ema_fire_distance': ema_fire_distance,
                #     'ema_fire_altitude': ema_fire_altitude,
                #     'ema_fire_delta_psi': ema_fire_delta_psi,
                #     'ema_fire_theta': ema_fire_theta,
                #     'ema_ATA': ema_ATA,
                #     'ema_delta_psi_threat': ema_delta_psi_threat,
                #     'ema_delta_theta': ema_delta_theta
                # })
                # logger.add(f"SPECIAL/开火权重", fire_reward_weight, total_steps_of_maneuver)
                # logger.add(f"SPECIAL/0 W_d_fire", fire_inside_weight[0], total_steps_of_maneuver)
                # logger.add(f"SPECIAL/1 W_t_since_fire", fire_inside_weight[1], total_steps_of_maneuver)
                # logger.add(f"SPECIAL/2 W_AA_fire", fire_inside_weight[2], total_steps_of_maneuver)
                # logger.add(f"SPECIAL/3 W_psi_fire", fire_inside_weight[3], total_steps_of_maneuver)
                # logger.add(f"SPECIAL/4 W_v_fire", fire_inside_weight[4], total_steps_of_maneuver)
                # logger.add(f"SPECIAL/5 W_theta_fire", fire_inside_weight[5], total_steps_of_maneuver)
            
            if not adj_r_w:
                fire_inside_weight = None
                fire_reward_weight = None

            # A. 获取当前策略权重 (CPU)
            current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
            
            # B. 分发任务给 Worker
            # 这一步 Master 决定每个 Worker 打谁
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            # [修正] 处理纯自博弈逻辑：当没有初始规则对手时，筛选分数最高的 MAX_HISTORY_SIZE 个对手作为匹配池
            if not init_elo_ratings:
                # 按照 Elo 分数降序排列，排除内部特殊键
                sorted_all_keys = [k for k in sorted(elo_ratings.keys(), 
                                                key=lambda x: elo_ratings[x] if not x.startswith("__") else -1e9, 
                                                reverse=True) if not k.startswith("__")]
                effective_pool = {k: elo_ratings[k] for k in sorted_all_keys[:MAX_HISTORY_SIZE]}
            else:
                effective_pool = elite_elo_ratings
                
            for rank in range(num_workers):
                # 采样对手
                probs, opponent_keys = get_opponent_probabilities(
                    effective_pool,
                    hall_of_fame,
                    target_elo=main_agent_elo,
                    SP_type=self_play_type,
                    sigma=sigma_elo,
                    compete_old_rate=compete_old_rate,
                    deltaFSP_epsilon=deltaFSP_epsilon,
                )
                selected_opponents = np.random.choice(opponent_keys, size=num_workers, p=probs).tolist()
            
            for rank in range(num_workers):
                selected_opponent_name = selected_opponents[rank]
                
                # 更新GameTimes表
                if selected_opponent_name in GameTimes:
                    GameTimes[selected_opponent_name] += 1
                else:
                    GameTimes[selected_opponent_name] = 1
                
                # 准备对手数据
                opp_type = 'rule'
                opp_data = 0
                if "Rule" in selected_opponent_name:
                    try:
                        rule_num = int(selected_opponent_name.split('_')[1])
                    except:
                        rule_num = 0
                    opp_data = rule_num
                else:
                    opp_type = 'nn'
                    adv_path = os.path.join(log_dir, f"{selected_opponent_name}.pt")
                    if os.path.exists(adv_path):
                        opp_data = torch.load(adv_path, map_location='cpu', weights_only=1) # 传给 Worker 必须是 CPU Tensor
                    else:
                        # Fallback
                        opp_type = 'rule'
                        opp_data = 0
                
                opp_temperature = np.random.uniform(0.99, 1.0) # 0.8, 1.0
                opp_info = (selected_opponent_name, opp_type, opp_data, opp_temperature)
                
                # 初始位置配置
                rb, bb = create_initial_state_worker(randomized_birth)
                settings = {
                    'randomized_birth': randomized_birth,
                    'action_cycle_multiplier': action_cycle_multiplier,
                    'weight_reward': weight_reward_0,
                    'red_birth': rb,
                    'blue_birth': bb,
                    # 'R_cage_range': R_cage_range, # 将范围传给Worker
                    'fire_mask': fire_mask,
                    'end_reward_weight': 0.3 + np.clip(total_steps_of_maneuver/18e3, 0, 1)*(0.4-0.3), # 1.0
                    'fire_inside_weight': fire_inside_weight,
                    'fire_reward_weight': fire_reward_weight,
                }
                
                # 发送指令 pipe.send
                pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # C. 等待所有 Worker 完成 (Barrier)
            batch_results = []
            for rank in range(num_workers):
                try: # <--- 【新增】
                    res = pipes[rank].recv() # 阻塞等待  # pipe.send
                except EOFError: # <--- 【新增】捕获管道断开错误
                    print(f"[Error] Worker {rank} crashed silently.")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed.")
                    
                # [新增] 检查 Worker 是否传回了奔溃信息
                if isinstance(res, dict) and 'error' in res:
                    print(f"--- Master received error from Worker {rank}, aborting. ---")
                    # 关闭所有子进程防止残留
                    for p in workers: p.terminate()
                    # 抛出具体的运行时错误
                    raise RuntimeError(f"Worker {rank} crashed with error:\n{res['error']}")
                    
                batch_results.append(res)
            
            # --- 3. 数据聚合与处理 ---
            batch_total_steps_of_maneuver = 0
            batch_wins = 0
            batch_loss_cnt = 0
            batch_draw_cnt = 0        # 新增统计
            batch_bvr_perish_together_cnt = 0 # 新增统计
            batch_total_return = 0    # 新增统计
            batch_total_event_return = 0
            batch_total_m_fired = 0   # 新增统计
            batch_fire_supervised = 0 # [新增] 本 batch 收集到的已结算开火记录数
            
            # 新增: 批次开火策略指标统计
            batch_blue_fire_intervals = []
            batch_blue_fire_delta_psis = []
            batch_blue_fire_distances = []
            batch_blue_fire_AA_hors = []
            batch_blue_fire_alts = []
            batch_blue_fire_thetas = []
            batch_blue_ATAs = []
            batch_blue_delta_psi_threats = []
            batch_blue_delta_thetas = []
            
            for res in batch_results:
                # res 结构: {'trans':..., 'ego_tr':..., 'enm_tr':..., 'metrics':..., 'opp_name':...}
                l_tr = res['trans'] # PPO 训练数据 (含探索)
                ego_tr = res['ego_trans'] # SIL 蓝方数据
                enm_tr = res['enm_trans'] # SIL 红方数据
                metrics = res['metrics']
                opp_name = res['opp_name']
                
                # --- 新增: 更新红方（对手）开火角度参数的 EMA ---
                ep_avg_fire_theta = res.get('ep_avg_fire_theta')
                ep_avg_ATA = res.get('ep_avg_ATA')
                ep_avg_delta_psi_threat = res.get('ep_avg_delta_psi_threat')
                ep_avg_delta_theta = res.get('ep_avg_delta_theta')
                ep_avg_delta_psi = res.get('ep_avg_delta_psi')
                
                # 更新 EMA [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                alpha_ema = 0.1
                if opp_name not in Elite_Fire_Stats:
                    Elite_Fire_Stats[opp_name] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    
                old_stats = Elite_Fire_Stats[opp_name]
                new_stats = list(old_stats)
                
                if ep_avg_fire_theta is not None:
                    new_stats[0] = alpha_ema * ep_avg_fire_theta + (1 - alpha_ema) * old_stats[0]
                if ep_avg_ATA is not None:
                    new_stats[1] = alpha_ema * ep_avg_ATA + (1 - alpha_ema) * old_stats[1]
                if ep_avg_delta_psi_threat is not None:
                    new_stats[2] = alpha_ema * ep_avg_delta_psi_threat + (1 - alpha_ema) * old_stats[2]
                if ep_avg_delta_theta is not None:
                    new_stats[3] = alpha_ema * ep_avg_delta_theta + (1 - alpha_ema) * old_stats[3]
                if ep_avg_delta_psi is not None:
                    new_stats[4] = alpha_ema * ep_avg_delta_psi + (1 - alpha_ema) * old_stats[4]
                    
                Elite_Fire_Stats[opp_name] = new_stats
                
                # --- 新增: 更新蓝方（本方/主代理）开火角度参数的 EMA ---
                ep_blue_avg_fire_theta = res.get('ep_blue_avg_fire_theta')
                ep_blue_avg_ATA = res.get('ep_blue_avg_ATA')
                ep_blue_avg_delta_psi_threat = res.get('ep_blue_avg_delta_psi_threat')
                ep_blue_avg_delta_theta = res.get('ep_blue_avg_delta_theta')
                ep_blue_avg_delta_psi = res.get('ep_blue_avg_delta_psi')
                
                # 使用当前主代理的key来记录本方行为
                main_agent_key = "__CURRENT_MAIN__"
                if main_agent_key not in Elite_Fire_Stats:
                    Elite_Fire_Stats[main_agent_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                old_main_stats = Elite_Fire_Stats[main_agent_key]
                new_main_stats = list(old_main_stats)
                
                if ep_blue_avg_fire_theta is not None:
                    new_main_stats[0] = alpha_ema * ep_blue_avg_fire_theta + (1 - alpha_ema) * old_main_stats[0]
                if ep_blue_avg_ATA is not None:
                    new_main_stats[1] = alpha_ema * ep_blue_avg_ATA + (1 - alpha_ema) * old_main_stats[1]
                if ep_blue_avg_delta_psi_threat is not None:
                    new_main_stats[2] = alpha_ema * ep_blue_avg_delta_psi_threat + (1 - alpha_ema) * old_main_stats[2]
                if ep_blue_avg_delta_theta is not None:
                    new_main_stats[3] = alpha_ema * ep_blue_avg_delta_theta + (1 - alpha_ema) * old_main_stats[3]
                if ep_blue_avg_delta_psi is not None:
                    new_main_stats[4] = alpha_ema * ep_blue_avg_delta_psi + (1 - alpha_ema) * old_main_stats[4]
                    
                Elite_Fire_Stats[main_agent_key] = new_main_stats
                
                # [新增] 填充 buffer 用户打印详情
                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")
                
                batch_total_steps_of_maneuver += metrics['steps']
                batch_total_return += metrics['return'] # 已是固定±100终局的归一化回报
                batch_total_event_return += metrics['event_return']
                batch_total_m_fired += metrics['m_fired']

                # [新增] 收集本回合已结算的开火命中记录，直接拼接入滚动缓冲
                fire_records = [r for r in res.get('fire_hit_records', []) if r.get('hit_record') is not None]
                fire_records_buffer.extend(fire_records)
                batch_fire_supervised += len(fire_records)
                if metrics.get('BVR_perish_together', False):
                    batch_bvr_perish_together_cnt += 1

                # 收集蓝方开火策略指标
                ep_blue_avg_fire_interval = res.get('ep_blue_avg_fire_interval')
                ep_blue_avg_fire_delta_psi = res.get('ep_blue_avg_fire_delta_psi')
                ep_blue_avg_fire_distance = res.get('ep_blue_avg_fire_distance')
                ep_blue_avg_fire_AA_hor = res.get('ep_blue_avg_fire_AA_hor')
                ep_blue_avg_fire_altitude = res.get('ep_blue_avg_fire_altitude')
                ep_blue_avg_fire_theta = res.get('ep_blue_avg_fire_theta')
                
                if ep_blue_avg_fire_interval is not None:
                    batch_blue_fire_intervals.append(ep_blue_avg_fire_interval)
                if ep_blue_avg_fire_delta_psi is not None:
                    batch_blue_fire_delta_psis.append(ep_blue_avg_fire_delta_psi)
                if ep_blue_avg_fire_distance is not None:
                    batch_blue_fire_distances.append(ep_blue_avg_fire_distance)
                if ep_blue_avg_fire_AA_hor is not None:
                    batch_blue_fire_AA_hors.append(ep_blue_avg_fire_AA_hor)
                if ep_blue_avg_fire_altitude is not None:
                    batch_blue_fire_alts.append(ep_blue_avg_fire_altitude)
                if ep_blue_avg_fire_theta is not None:
                    batch_blue_fire_thetas.append(ep_blue_avg_fire_theta)
                
                if ep_blue_avg_ATA is not None:
                    batch_blue_ATAs.append(ep_blue_avg_ATA)
                if ep_blue_avg_delta_psi_threat is not None:
                    batch_blue_delta_psi_threats.append(ep_blue_avg_delta_psi_threat)
                if ep_blue_avg_delta_theta is not None:
                    batch_blue_delta_thetas.append(ep_blue_avg_delta_theta)

                if metrics['win']: batch_wins += 1
                elif metrics['lose']: batch_loss_cnt += 1
                else: 
                    batch_draw_cnt += 1
                
                # 3.1 聚合 PPO 数据到全局 Buffer
                for k in transition_dict:
                    transition_dict[k].extend(l_tr[k])
                
                # 3.3 ELO 更新 (实时更新)
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                
                
                if opp_name in elo_ratings:
                    prev_main_elo = main_agent_elo
                    adv_elo = elo_ratings[opp_name]
                    
                    # 更新主智能体Elo分
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                    # 更新对手Elo分
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                    elo_ratings["__CURRENT_MAIN__"] = main_agent_elo
                else:
                    # 新对手：初始Elo与主智能体相同
                    new_adv_elo = main_agent_elo
                
                # # --- 新增: 出界惩罚 ---
                # # 若对手在3分钟内出界，对手Elo额外扣除200分；当前主智能体Elo不额外修改
                # red_out_cage = res.get('red_out_cage', False)
                # red_out_cage_time = res.get('red_out_cage_time')
                # if red_out_cage and red_out_cage_time is not None and red_out_cage_time <= 3 * 60:
                #     new_adv_elo -= 200
                
                elo_ratings[opp_name] = new_adv_elo
                # 同步更新 Elite 池中已有的对手Elo分值
                if opp_name in elite_elo_ratings:
                    elite_elo_ratings[opp_name] = new_adv_elo
                # 同步更新 Hall of Fame 中已有的对手Elo分值
                if opp_name in hall_of_fame:
                    hall_of_fame[opp_name] = new_adv_elo
            
            # 计算蓝方开火策略指标的批次平均值
            batch_blue_avg_fire_interval = float(np.mean(batch_blue_fire_intervals)) if batch_blue_fire_intervals else None
            batch_blue_avg_fire_delta_psi = float(np.mean(batch_blue_fire_delta_psis)) if batch_blue_fire_delta_psis else None
            batch_blue_avg_fire_distance = float(max(batch_blue_fire_distances)) if batch_blue_fire_distances else None
            batch_blue_avg_fire_AA_hor = float(np.mean(batch_blue_fire_AA_hors)) if batch_blue_fire_AA_hors else None
            batch_blue_avg_fire_altitude = float(np.mean(batch_blue_fire_alts)) if batch_blue_fire_alts else None
            batch_blue_avg_fire_theta = float(np.mean(batch_blue_fire_thetas)) if batch_blue_fire_thetas else None
            batch_blue_avg_ATA = float(np.mean(batch_blue_ATAs)) if batch_blue_ATAs else None
            batch_blue_avg_delta_psi_threat = float(np.mean(batch_blue_delta_psi_threats)) if batch_blue_delta_psi_threats else None
            batch_blue_avg_delta_theta = float(np.mean(batch_blue_delta_thetas)) if batch_blue_delta_thetas else None

            # [新增] 用 EMA(指数=0.2) 平滑批次均值
            def _ema_update(ema_val, batch_val, alpha=EMA_ALPHA):
                if batch_val is None:
                    return ema_val
                return batch_val if ema_val is None else (1 - alpha) * ema_val + alpha * batch_val
            ema_fire_interval = _ema_update(ema_fire_interval, batch_blue_avg_fire_interval if batch_blue_avg_fire_interval is not None else None)
            ema_fire_delta_psi = _ema_update(ema_fire_delta_psi, batch_blue_avg_fire_delta_psi*180/pi if batch_blue_avg_fire_delta_psi is not None else None)
            ema_fire_distance = _ema_update(ema_fire_distance, batch_blue_avg_fire_distance if batch_blue_avg_fire_distance is not None else None)
            ema_fire_AA_hor = _ema_update(ema_fire_AA_hor, batch_blue_avg_fire_AA_hor*180/pi if batch_blue_avg_fire_AA_hor is not None else None)
            ema_fire_altitude = _ema_update(ema_fire_altitude, batch_blue_avg_fire_altitude if batch_blue_avg_fire_altitude is not None else None)
            ema_fire_theta = _ema_update(ema_fire_theta, batch_blue_avg_fire_theta*180/pi if batch_blue_avg_fire_theta is not None else None)
            ema_ATA = _ema_update(ema_ATA, batch_blue_avg_ATA*180/pi if batch_blue_avg_ATA is not None else None)
            ema_delta_psi_threat = _ema_update(ema_delta_psi_threat, batch_blue_avg_delta_psi_threat*180/pi if batch_blue_avg_delta_psi_threat is not None else None)
            ema_delta_theta = _ema_update(ema_delta_theta, batch_blue_avg_delta_theta*180/pi if batch_blue_avg_delta_theta is not None else None)

            # [新增] 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps_of_maneuver += batch_total_steps_of_maneuver
            batch_idx += 1
            
            # --- 3.5 记录批次聚合指标 ---
            # 更新在线胜率 EMA 并且加入偏差修正（既稳又抗滞后）
            batch_score = (batch_wins + batch_draw_cnt * 0.5) / num_workers
            ema_step += 1
            alpha_ema = 0.1  # 平滑系数，较大可防止滞后，和tensorboard的smoothing是补数的关系
            # 若第一步则直接初始化，避免前期偏差
            if ema_step == 1:
                ema_score = batch_score
            else:
                ema_score = (1 - alpha_ema) * ema_score + alpha_ema * batch_score
                

            # 使用带有偏差修正的滤波值
            filtered_score = ema_score
            if use_RND and rnd_mse is not None:
                logger.add("train_plus/RND_mse", rnd_mse, total_steps_of_maneuver)
            logger.add("train_plus/batch_score", batch_score, total_steps_of_maneuver)
            logger.add("train_plus/filtered_score", filtered_score, total_steps_of_maneuver)
            logger.add("train_plus/target_p1", target_p1, total_steps_of_maneuver)

            # 记录开火策略指标 - 蓝方（本方），使用原始批次均值
            if batch_blue_avg_fire_interval is not None:
                logger.add("special/1 开火间隔时长", batch_blue_avg_fire_interval, total_steps_of_maneuver)
            if batch_blue_avg_fire_delta_psi is not None:
                logger.add("special/2 开火abs(delta_psi)", batch_blue_avg_fire_delta_psi*180/pi, total_steps_of_maneuver)
            if batch_blue_avg_fire_distance is not None:
                logger.add("special/3 开火距离", batch_blue_avg_fire_distance, total_steps_of_maneuver)
            if batch_blue_avg_fire_AA_hor is not None:
                logger.add("special/4 开火abs(AA_hor)", batch_blue_avg_fire_AA_hor*180/pi, total_steps_of_maneuver)
            if batch_blue_avg_fire_altitude is not None:
                logger.add("special/0 开火高度", batch_blue_avg_fire_altitude, total_steps_of_maneuver)
            if batch_blue_avg_fire_theta is not None:
                logger.add("special/5 fire_theta", batch_blue_avg_fire_theta*180/pi, total_steps_of_maneuver)
            if batch_blue_avg_ATA is not None:
                logger.add("special/6 ATA30", batch_blue_avg_ATA*180/pi, total_steps_of_maneuver)
            if batch_blue_avg_delta_psi_threat is not None:
                logger.add("special/7 delta_psi_threat", batch_blue_avg_delta_psi_threat*180/pi, total_steps_of_maneuver)
            if batch_blue_avg_delta_theta is not None:
                logger.add("special/8 delta_theta30", batch_blue_avg_delta_theta*180/pi, total_steps_of_maneuver)
            
            # [新增] 保存 EMA 状态和控制器状态到 special.json
            special_data = {
                "ema_fire_interval": ema_fire_interval,
                "ema_fire_delta_psi": ema_fire_delta_psi,
                "ema_fire_distance": ema_fire_distance,
                "ema_fire_AA_hor": ema_fire_AA_hor,
                "ema_fire_altitude": ema_fire_altitude,
                "ema_fire_theta": ema_fire_theta,
                "ema_ATA": ema_ATA,
                "ema_delta_psi_threat": ema_delta_psi_threat,
                "ema_delta_theta": ema_delta_theta,
                "controller_state": RWController.state_dict(),  # [新增] 保存控制器状态
            }
            with open(os.path.join(log_dir, "special.json"), "w", encoding="utf-8") as f:
                json.dump(special_data, f, ensure_ascii=False, indent=2)
            
            # 记录平均回报与胜率
            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps_of_maneuver)
            logger.add("train_plus/Avg event return", batch_total_event_return / num_workers, total_steps_of_maneuver)
            logger.add("train/2 win", batch_wins / num_workers, total_steps_of_maneuver)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps_of_maneuver)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps_of_maneuver)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, total_steps_of_maneuver)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, total_steps_of_maneuver)
            # 找最好的智能体
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps_of_maneuver)
            logger.add("agent/ batch_step", batch_idx, total_steps_of_maneuver)

            # --- 5. 更新，保存与维护 (Checkpoint & Pool) ---
            if batch_idx % update_interval_of_batch == 0 and \
                len(transition_dict['dones']) >= transition_dict_threshold:
                # # --- 4. 执行训练 (PPO Update) ---
                # # 当收集的数据量超过 capacity 时更新
                # if len(transition_dict['dones']) >= transition_dict_threshold:
                # 重构 Action 结构 (List[Dict] -> Dict[Array])
                transition_dict['actions'] = restructure_actions(transition_dict['actions'])
                
                '记录ELo相对位置'
                # [新增] 调节alpha_il
                # --- [新增] 动态计算 alpha_il ---
                # 1. 筛选对手池：Rule开头的所有Key + actor_rein开头的最后300个Key
                all_keys = list(elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                # 取最后（最新插入）的300个
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                
                target_pool_keys = rule_keys + latest_rein_keys
                # 计算池子滑动平均分
                avg_pool_elo = np.mean([elo_ratings[k] for k in target_pool_keys])
                # 计算 Elo 差值 x (当前主分 - 池子均分)
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, total_steps_of_maneuver)
                
                # 学习率warm_up
                actor_lr = min(actor_lr0, actor_lr0 * total_steps_of_maneuver/1e6)
                critic_lr = min(critic_lr0, critic_lr0 * total_steps_of_maneuver/1e6)
                student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

                if batch_idx <= actor_freeze_until:
                    freeze_actor = 1
                else:
                    freeze_actor = 0
                # # critic先收敛
                # if total_steps_of_maneuver < 5e3:
                #     freeze_actor = 1
                
                max_fire_logits = 4.0

                alpha_distill = 0
                teacher_wrapper = None
                k_entropy_cat_scale = 1.0

                student_agent.k_entropy['cat'] = base_k_cat * k_entropy_cat_scale

                student_agent.update(transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed, target_p1=target_p1,
                                     k_nonlinear=k_nonlinear, mask_on=fire_mask, actor_frozen=freeze_actor, bern_max_logits=max_fire_logits,
                                     alpha_distill=alpha_distill, teacher_actor=adistill_rule_wrappers,
                                     AFiltered=AFiltered, conf_thres=conf_thres, bern_included=bern_included,
                                     ppo_with_bern=ppo_with_bern)

                # [新增] 开火命中监督更新：从滚动缓冲采样，仅更新 actor.net.fc_bern，不影响机动策略
                if use_supervised_fire and len(fire_records_buffer) > 100: #  and\
                    # (student_agent.max_fire_prob / student_agent.min_fire_prob <= sup_on_prob_ratio  or  ppo_with_bern==0): # PPO不更新，就全程带上，否则只作为开火分布保护器

                    fire_sl_batch = random.sample(list(fire_records_buffer), min(int(supervised_fire_batch_size), len(fire_records_buffer)))
                    student_agent.supervised_fire_update(fire_sl_batch, epochs=1, batch_size=int(supervised_fire_batch_size), fire_lr_ratio=1)
                    logger.add("train_plus/supervised_fire_buffer_size", len(fire_records_buffer), total_steps_of_maneuver)
                    logger.add("train_plus/supervised_fire_samples", batch_fire_supervised, total_steps_of_maneuver)
                    logger.add("train/7b supervised_fire_loss", getattr(student_agent, 'last_supervised_fire_loss', 0.0), total_steps_of_maneuver)


                # # 开火概率保护，如果策略向满开火/不开一发坍缩，直接用有监督暴力修正开火概率
                # if batch_idx % 10 == 0:
                #     student_agent.fire_prob_protection(transition_dict, protect_epochs=4)
                
                # # 机动概率保护，未被调好，无法区分告警状态的有无，加上进攻引导就不会躲，加上防御引导又不会进攻
                # if  \
                #     (batch_blue_avg_fire_delta_psi*180/pi > 60 or
                #         batch_blue_avg_delta_psi_threat*180/pi < 90):
                #     student_agent.maneuver_il_protection(transition_dict, alpha=3, epochs=12)

                # 计算/更新 PPO actor pre-clip 梯度的 EMA 值
                current_ppo_grad = student_agent.pre_clip_actor_grad
                if current_ppo_grad is not None and not np.isnan(current_ppo_grad):
                    if ppo_grad_ema is None:
                        ppo_grad_ema = current_ppo_grad
                    else:
                        ppo_grad_ema = 0.95 * ppo_grad_ema + 0.05 * current_ppo_grad

                alpha_il_real = alpha_il #  * np.clip(1 - total_steps_of_maneuver/5e6, 0.1, 1)

                # [Modification] 保留原有梯度监控代码
                actor_pre_clip_grad = student_agent.pre_clip_actor_grad
                critic_pre_clip_grad = student_agent.pre_clip_critic_grad

                # 梯度监控
                # logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps_of_maneuver)
                # logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps_of_maneuver)
                # 损失函数监控
                logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps_of_maneuver)
                logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps_of_maneuver)
                # 强化学习actor特殊项监控
                logger.add("train/9 entropy", student_agent.entropy_mean, total_steps_of_maneuver)
                logger.add("train/9 entropy_cat", student_agent.entropy_cat, total_steps_of_maneuver)
                logger.add("train/9 entropy_bern", student_agent.entropy_bern, total_steps_of_maneuver)
                logger.add("train_plus/max_fire_prob", student_agent.max_fire_prob, total_steps_of_maneuver)
                logger.add("train_plus/min_fire_prob", student_agent.min_fire_prob, total_steps_of_maneuver)

                # logger.add("train/10 advantage", student_agent.advantage, total_steps_of_maneuver) 
                # 强化学习
                # logger.add("train/10 explained_var", student_agent.explained_var, total_steps_of_maneuver)
                # logger.add("train/10 approx_kl", student_agent.approx_kl, total_steps_of_maneuver)
                # logger.add("train/10 clip_frac", student_agent.clip_frac, total_steps_of_maneuver)
                
                # [新增] 诊断监控
                logger.add("train_plus/td_error_var", student_agent.td_error_var, total_steps_of_maneuver)

                print(f"Step {total_steps_of_maneuver}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # 原本是在这里清空Buffer的，但是现在要在搅拌之后清空，所以移到了后面
                
                # A. 保存模型
                actor_key = f"actor_rein{batch_idx}"
                
                # 正常保存模型
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{actor_key}.pt"))
                torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                # 额外保存当前训练用的actor参数（覆盖式保存，用于续训）
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, "current_actor.pt"))
                print(f"Saved Checkpoint: {actor_key}")
                print(f"Saved Current Actor: current_actor.pt")
                
                # 清空 Buffer（在搅拌之后）
                transition_dict = copy.deepcopy(empty_transition_dict)

                # B. 经典胜率精英池维护
                # 只有自博弈能够更新精英Elo和胜率表，否则只能更新普通胜率和Elo表
                if total_steps_of_maneuver >= WARM_UP_STEPS:
                    # 复制当前主代理的蓝方行为统计到该历史版本，记录它"作为蓝方时"的行为特征
                    if "__CURRENT_MAIN__" in Elite_Fire_Stats:
                        Elite_Fire_Stats[actor_key] = copy.deepcopy(Elite_Fire_Stats["__CURRENT_MAIN__"])
                    else:
                        Elite_Fire_Stats[actor_key] = [0.0, 0.0, 0.0, 0.0, 0.0]  # [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                    elo_ratings[opp_name] = main_agent_elo

                    
                # -----------------------------------------------------------
                # 逻辑分支 B: 维护“全量历史记录” (Full JSON)
                # -----------------------------------------------------------
                # 目标：记录所有产生过的 Agent 的最后一次已知 Elo，无论它是否在精英池里
                # 无论是否进入精英池，全量表都要记录
                elo_ratings[actor_key] = main_agent_elo
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                
                # -----------------------------------------------------------
                # 逻辑分支 B: 定期重刷精英池 (每 1M step)
                # -----------------------------------------------------------
                if total_steps_of_maneuver > 0 and total_steps_of_maneuver % 1e6 == 0:
                    print(f"\n>>> [Elite Pool Refresh] Refreshing elite_elo_ratings at {total_steps_of_maneuver} steps...")
                    # 从 elo_ratings 中选出 Elo 最高的 min(当前非Rule数量, MAX_HISTORY_SIZE) 个非 Rule 智能体
                    non_rule_elo = {k: v for k, v in elo_ratings.items() if not k.startswith("Rule") and not k.startswith("__")}
                    current_non_rule_count = len([k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")])
                    refresh_size = min(current_non_rule_count, MAX_HISTORY_SIZE)
                    sorted_non_rule = sorted(non_rule_elo.items(), key=lambda x: x[1], reverse=True)
                    top_non_rule = dict(sorted_non_rule[:refresh_size])
                    # 加上所有 Rule
                    rule_elo = {k: v for k, v in elo_ratings.items() if k.startswith("Rule")}
                    # 重置 elite_elo_ratings
                    elite_elo_ratings = {**rule_elo, **top_non_rule}
                    print(f"  [Elite Pool Refresh] Refreshed: {len(rule_elo)} Rules + {len(top_non_rule)} top agents (size={refresh_size})")

                # 5. 保存全量日志
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

                # -----------------------------------------------------------
                # 逻辑分支 C: 保存“精英池快照” (Elite JSON)
                # -----------------------------------------------------------
                # 这才是下次训练 resume 时应该读取的文件
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                
                # -----------------------------------------------------------
                # 逻辑分支 D: 保存名人堂 (hall_of_fame.json)
                # -----------------------------------------------------------
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                # -----------------------------------------------------------
                # 逻辑分支 E: 保存胜率表 (用于核心更新与选择)
                # -----------------------------------------------------------
                with open(Elite_Fire_Stats_path, "w", encoding="utf-8") as f:
                    json.dump(Elite_Fire_Stats, f, ensure_ascii=False, indent=2)

                # --- 日志记录 (Logging) - 保持不变，展示的是精英池状态 ---
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps_of_maneuver)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    
                    # 升序：Elo 低（弱对手）在前。threshold=0 取最弱 Rule Elo（最低），=1 取最强 Rule Elo（最高）
                    rule_elos = sorted([v for k, v in elite_elo_ratings.items() if k.startswith("Rule")], reverse=False) # 升序排列
                    if not rule_elos:
                        rule_elo_thres = main_agent_elo
                    elif ADMISSION_THRESHOLD < 0:
                        rule_elo_thres = main_agent_elo  # 无门槛，差值仅供参考
                    else:
                        t = np.clip(ADMISSION_THRESHOLD, 0, 1)
                        idx = np.clip(int(t * (len(rule_elos) - 1) + 0.5), 0, len(rule_elos) - 1)
                        rule_elo_thres = rule_elos[idx]  # =0 取最低 Rule Elo，=0.5 取中位数，=1 取最高 Rule Elo

                    # 记录与 Rule 阈值的差值 (维持旧指标名)
                    elo_diff_to_thres = main_agent_elo - rule_elo_thres
                    logger.add("Elo_Centered/EloDiffToMean", elo_diff_to_thres, total_steps_of_maneuver)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps_of_maneuver)

                    curr_rate = 0.5 if elo_spread == 0 else (main_agent_elo - min_elo) / elo_spread

                    # Elo分数没有稳定不许记录
                    if total_steps_of_maneuver >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rate_normed %", curr_rate * 100, total_steps_of_maneuver)
                    
                    
                    # # 仅对高门槛使用
                    # rule_elo_thres = ADMISSION_THRESHOLD * elo_ratings['Rule_2'] +\
                    #     (1-ADMISSION_THRESHOLD) * min_rule_elo

                    if main_agent_elo >= rule_elo_thres:
                        # 满员清理
                        history_keys = [k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")]
                        while len(history_keys) >= MAX_HISTORY_SIZE:
                            # 每次找到当前池子中最弱的一个
                            weakest_history_key = min(history_keys, key=lambda k: elite_elo_ratings[k])
                            old_elo = elite_elo_ratings[weakest_history_key]
                            # 从 ELO 字典和局部列表中同步删除
                            del elite_elo_ratings[weakest_history_key]
                            history_keys.remove(weakest_history_key)
                            print(f"[Pool Cleanup] Kicked weakest: {weakest_history_key} (Elo: {old_elo:.0f}), Current Pool: {len(history_keys)}")
                        
                        # --- 正式入池 ---
                        if hist_agent_as_opponent:
                            elite_elo_ratings[actor_key] = main_agent_elo
                            print(f"Accepted {actor_key} into Elite Pool.")

                    # # 动态学习率调节
                    # actor_lr = 1e-4 + np.clip(curr_rate, 0, 1) * (1e-5 - 1e-4)
                    # critic_lr = actor_lr * 5
                    # student_agent.set_learning_rate(actor_lr, critic_lr)

                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps_of_maneuver)

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, total_steps_of_maneuver)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps_of_maneuver)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, total_steps_of_maneuver)

                # --- 例行保存优化器状态和步数 ---
                torch.save({
                    'actor_optimizer': student_agent.actor_optimizer.state_dict(),
                    'critic_optimizer': student_agent.critic_optimizer.state_dict(),
                    'rnd_target': student_agent.rnd_target.state_dict() if student_agent.rnd_target is not None else None,
                    'rnd_prediction': student_agent.rnd_prediction.state_dict() if student_agent.rnd_prediction is not None else None,
                    'rnd_optimizer': student_agent.rnd_optimizer.state_dict() if student_agent.rnd_optimizer is not None else None,
                }, os.path.join(log_dir, "optimizers_state.pt"))

                
                # 保存开火监督数据 buffer
                if use_supervised_fire:
                    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
                    with open(fire_buffer_path, "wb") as f:
                        pickle.dump({"records": list(fire_records_buffer), "maxlen": fire_records_buffer.maxlen}, f)
                    print(f"Saved fire records buffer to {fire_buffer_path}. Size: {len(fire_records_buffer)}")
                    
                # print(f"Optimizers routinely saved to optimizers_state.pt")
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps_of_maneuver
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

        # --- [新增] 达到 max_steps_of_maneuver 后的交互逻辑 ---
        print(f"\n--- Target steps reached: {total_steps_of_maneuver} / {current_max_steps_of_maneuver} ---")
        # 【新增】明确告知用户 Worker 当前状态
        print("All simulation workers are now idling safely. System is paused.") 
        
        inp = input(f"Enter new max_steps_of_maneuver (current {total_steps_of_maneuver}), or press Enter to exit: ")
        
        if not inp.strip():
            print("No input provided. Exiting training.")
            break # 退出外层 while True 循环
        try:
            new_max = int(inp)
            if new_max > total_steps_of_maneuver:
                current_max_steps_of_maneuver = new_max
                print(f"Continuing training until {current_max_steps_of_maneuver} steps.")
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