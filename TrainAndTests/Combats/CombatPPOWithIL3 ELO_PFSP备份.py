'''
更改目标：
1、实现混合规则对手开关可调, 对手池中的规则对手都有谁, 需可调开关      √ init_elo_ratings -> pure_self_play
2、实现历史对手淘汰机制开关可调                                     √ should_kick
3、自模仿仅包含机动/包含机动与开火, 需可调开关                       √ sil_only_maneuver
4、是否能够向对手池中添加历史策略, 需可调开关                        √ self_play_type -> hist_agent_as_opponent
5、指定自博弈方式，以string传入，最后作用到 get_opponent_probabilities 上去      √ self_play_type
6、0.4和0.7的逻辑梳理   √ 直接砍掉

elo_ratings.json 仍然是所有agent的elo表格，但是变量 elo_ratings 仅包含规则和精英历史智能体
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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from prepare_il_datas import run_rules
from VsBaseline_while_training import test_worker



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
    td['states'].append(state) # 修改：这里存储Critic用的全局状态(pomdp=0)
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
        if current_len > self.max_size:
            keep_from = current_len - self.max_size
            self.addon_dict['obs'] = self.addon_dict['obs'][keep_from:]
            self.addon_dict['states'] = self.addon_dict['states'][keep_from:]
            self.addon_dict['returns'] = self.addon_dict['returns'][keep_from:]
            self.addon_dict['actions'] = self.addon_dict['actions'][keep_from:]
    
        # # 2. 基于 Return 的原子化排序与剪裁
        # # 通过 zip 绑定每一行数据，确保 S, A, R 在排序和删除时永远同步
        # combined = list(zip(
        #     self.addon_dict['obs'],
        #     self.addon_dict['states'],
        #     self.addon_dict['returns'],
        #     self.addon_dict['actions']
        # ))
        # # 按 returns (索引为 2) 降序排列（从高质量到低质量）
        # combined.sort(key=lambda x: x[2], reverse=True)
        
        # # 保留前 max_size 条高质量经验
        # top_data = combined[:self.max_size]
        
        # # 解包回列表
        # unzipped = list(zip(*top_data)) if top_data else ([], [], [], [])
        # self.addon_dict['obs'] = list(unzipped[0])
        # self.addon_dict['states'] = list(unzipped[1])
        # self.addon_dict['returns'] = list(unzipped[2])
        # self.addon_dict['actions'] = list(unzipped[3])
    
    def read(self, batch_size):
        """
        随机采样并进行格式转换。
        """
        total_len = len(self.addon_dict['states'])
        if total_len == 0:
            raise ValueError("IL_transition_buffer is empty.")
            
        indices = np.random.randint(0, total_len, size=batch_size)
        
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



def run_MLP_simulation(
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
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
):

    
    # 设置随机数种子
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # # 尽量开启确定性模式（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass
    
    # ------------------------------------------------------------------
    # 参数与环境配置
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    env = ChooseStrategyEnv(args)
    state_dim = env.obs_dim

    # 动作空间定义 (需要与 BasicRules 产生的数据对应)
    # cat: 离散机动动作头 (env.fly_act_dim 通常是一个列表 [n_actions])
    # bern: 攻击动作头 (env.fire_dim 通常是 1)
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    action_bound = None

    # 1. 创建神经网络
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)

    # 2. Wrapper
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

    # 3. student_agent
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
        max_std=label_smoothing
    )

    adv_agent = copy.deepcopy(student_agent)

    
    mp.set_start_method('spawn', force=True) # 重要：CUDA 环境下推荐使用 spawn
    
    # [修改] 使用 maxtasksperchild 防止内存泄漏 (如 JSBSim 未完全释放)
    test_pool = mp.Pool(processes=3, maxtasksperchild=10) 
    
    pending_tests = [] # 用于存放正在运行的测试任务
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    print("Start MARWIL Training...")

    student_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    # === 模仿训练循环 ===
    # 现在 original_il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, # 显存如果够大可以适当调大
            label_smoothing=label_smoothing
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    print("IL Training Finished.")
    
    # --- 新增：实例化混合缓冲区 ---
    il_transition_buffer = None
    if IL_epoches > 0:
        print("Initializing IL Transition Buffer...")
        # addon_dict 大小限制，可根据显存调整，例如 20000
        il_transition_buffer = IL_transition_buffer(original_il_transition_dict0, max_size=il_buffer_max_size)
    
    # ==============================================================================
    # 强化学习 (Self-Play / PFSP) 阶段
    # ==============================================================================
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    # # 根据参数数量缩放学习率
    # from Math_calculates.ScaleLearningRate import scale_learning_rate
    # actor_lr = scale_learning_rate(actor_lr, student_agent.actor)
    # critic_lr = scale_learning_rate(critic_lr, student_agent.critic)
    # student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    
    # tacview_input = input("Enable tacview visualization? (0=no, 1=yes) [default 0]: ").strip()
    tacview_input = 0
    
    if tacview_input == "":
        tacview_show = 0
    else:
        try:
            tacview_show = 1 if int(tacview_input) != 0 else 0
        except Exception:
            tacview_show = 0
    print(f"tacview_show={tacview_show}")
    env = ChooseStrategyEnv(args, tacview_show=tacview_show)
    env.shielded = no_crash # 不得不全程带上，否则对手也会撞地
    env.dt_move = dt_move # 仿真跑得快点
    
    t_bias = 0
    
    
    def create_initial_state(randomized=0):
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
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


    # --- [Fix] 初始化 ELO 变量与辅助函数 ---
    # 初始化 ELO 字典，包含基础规则智能体
    elo_ratings = init_elo_ratings
    # 定义路径
    # 注意：这里提前定义路径，确保后面逻辑一致
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    
    # [修改] 优先加载 Elite Pool，如果没有则尝试加载旧的 full pool，都没有则用默认
    if os.path.exists(elite_json_path):
        print(f"Loading Elite Elo Pool from: {elite_json_path}")
        with open(elite_json_path, 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
    elif os.path.exists(full_json_path):
        print(f"Warning: Elite Pool not found. Loading Full History from: {full_json_path}")
        with open(full_json_path, 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
            
    # 主智能体当前的 ELO
    main_agent_elo = 1200

    # [新增] 如果没有历史对手（例如第一次运行且屏蔽了规则），保存当前初始策略作为 actor_rein0
    if (not elo_ratings) or IL_epoches>0:
        init_opponent_name = "actor_rein0"
        init_opponent_path = os.path.join(log_dir, f"{init_opponent_name}.pt")
        torch.save(student_agent.actor.state_dict(), init_opponent_path)
        
        if not self_play_type=='None':
            elo_ratings[init_opponent_name] = 1200

        print(f"Initialized {init_opponent_name} as the first opponent.")

    def calculate_expected_score(player_elo, opponent_elo):
        """计算期望得分"""
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

    def update_elo(player_elo, opponent_elo, score):
        """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
        expected = calculate_expected_score(player_elo, opponent_elo)
        return player_elo + K_FACTOR * (score - expected)

    
    def get_opponent_probabilities(elo_ratings, target_elo=None, sigma=400, SP_type='PFSP_balanced'):
        """
        返回与 keys 顺序对应的概率数组。
        
        参数:
        - elo_ratings: dict, 格式为 {name: elo_value}
        - target_elo: float, PFSP 模式下优先靠近的 ELO。
        - sigma: float, PFSP 模式下的高斯核标准差。
        - SP_type: str, 采样类型:
            - 'PFSP_balanced': 匹配实力相近的对手（target = target_elo）。
            - 'PFSP_challenge': 匹配最强的对手（target = max_elo）。
            - 'FSP': 虚构自博弈，全样本均匀分布。
            - 'SP': 自博弈，仅选择 actor_rein 编号最大的那一个。
            - 'None': 仅在以 "Rule" 开头的 key 中均匀分布。
        """
        keys = list(elo_ratings.keys())
        if len(keys) == 0:
            return np.array([]), []

        elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

        # --- 1. 确定高斯采样的参数 (Target & Sigma) ---
        if SP_type.startswith('PFSP'):
            # [新增] 如果是通用的 'PFSP'，则在 balanced 和 challenge 之间随机选择
            if SP_type == 'PFSP':
                actual_target = 0.5*float(target_elo) if target_elo is not None else np.mean(elos) + 0.5*np.max(elos)
                
            actual_sigma = float(sigma)
            if SP_type == 'PFSP_balanced':
                actual_target = float(target_elo) if target_elo is not None else np.mean(elos)

            # elif SP_type == 'PFSP_balanced_biased':
            #     actual_target = (float(target_elo) if target_elo is not None else np.mean(elos)) + 200

            elif SP_type == 'PFSP_challenge':
                actual_target = np.max(elos)

            # elif SP_type == 'PFSP':  # 默认是平衡
            #     actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
            
            # 统一计算 PFSP 概率
            diffs = elos - actual_target
            scores = np.exp(-0.5 * (diffs / actual_sigma)**2)
            probs = scores / (scores.sum() + 1e-12)
            return probs, keys

            
        # # --- 逻辑 1: PFSP (保持原样) ---
        # if SP_type == 'PFSP':
        #     # 以高斯核度量相似度（基于差的平方）
        #     diffs = elos - float(target_elo)
        #     # 高斯核计算
        #     scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        #     probs = scores / (scores.sum() + 1e-12)
        #     return probs, keys

        # --- 逻辑 2: FSP (全样本均匀分布) ---
        elif SP_type == 'FSP':
            probs = np.ones(len(keys)) / len(keys)
            return probs, keys

        elif SP_type == 'deltaFSP':
            # 保持 keys 的插入顺序（json.load / dict 保留插入顺序）
            n = len(keys)
            if n == 0:
                return np.array([]), []
            # 将后 20% 视为“新”，其余为“旧”
            new_count = max(1, int(np.ceil(n * 0.2)))
            new_keys = keys[-new_count:]
            old_keys = keys[:-new_count] if len(keys) - new_count > 0 else []
            # 以 deltaFSP_epsilon 概率从“新”池均匀采样，否则从“旧”池均匀采样
            if np.random.rand() < float(deltaFSP_epsilon):
                probs = np.ones(len(new_keys)) / len(new_keys)
                return probs, new_keys
            else:
                # 若旧池为空则回退到新池
                if not old_keys:
                    probs = np.ones(len(new_keys)) / len(new_keys)
                    return probs, new_keys
                probs = np.ones(len(old_keys)) / len(old_keys)
                return probs, old_keys
        
        # --- 逻辑 3: SP (找到 actor_rein 编号最大者) ---
        elif SP_type == 'SP':
            # 过滤出所有以 actor_rein 开头的 key
            rein_keys = [k for k in keys if k.startswith('actor_rein')]
            if not rein_keys:
                return np.array([]), []
            
            # 提取编号并找到最大值对应的 key
            # 假设格式是 actor_rein + 数字，例如 actor_rein25
            def extract_number(key_str):
                num_part = key_str.replace('actor_rein', '')
                try:
                    return int(num_part)
                except ValueError:
                    return -1

            best_key = max(rein_keys, key=extract_number)
            
            # 只返回这一个元素，概率为 1.0
            return np.array([1.0]), [best_key]

        # --- 逻辑 4: None (Rule开头均匀分布) ---
        elif SP_type == 'None' or SP_type is None:
            rule_keys = [k for k in keys if k.startswith('Rule')]
            if not rule_keys:
                return np.array([]), []
            
            probs = np.ones(len(rule_keys)) / len(rule_keys)
            return probs, rule_keys

        else:
            raise ValueError(f"Unknown SP_type: {SP_type}")
    
    # 新增：将完整路径或名字缩短为 actor_rein<数字> 形式的 key（优先匹配 actor_*）
    def shorten_actor_key(path_or_name):
        base = os.path.basename(path_or_name)
        name, _ = os.path.splitext(base)
        m = re.search(r'(actor_rein\d+)', name) # [Fix] regex 修正匹配 rein
        if m: return m.group(1)
        return name

    # 循环变量初始化
    i_episode = 0 
    total_steps = 0
    launch_time_count = 0
    t_bias = 0
    decide_steps_after_update = 0
    return_list = []
    
    r_action_list = []
    b_action_list = []
        
    # 修改：初始化增加 'obs' 键
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    transition_dict = copy.deepcopy(empty_transition_dict)
    
    # --- 新增：测试回合控制变量 ---
    trigger = trigger0
    test_run = 0
    
    # 使用一个可变的变量来跟踪当前的 max_steps
    current_max_steps = int(max_steps)

    while True: # 外层循环，用于在达到 max_steps 后暂停并请求新目标
        
        while total_steps < current_max_steps:
            i_episode += 1
            
            # --- 新增：测试模式判断与设置 ---
            is_testing = False
            # -- 并行测试回合 --
            if total_steps >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
                # 1. 深度拷贝当前 Actor 权重到 CPU 内存
                current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}
                
                # 2. 异步启动 3 个对战
                for r_idx in [0, 1, 2, 3, 4]:
                    res_obj = test_pool.apply_async(
                        test_worker, 
                        args=(current_weights, r_idx, args, state_dim, hidden_dim, action_dims_dict, dt_maneuver, 'cpu') # [修改] 显式传入 dt_maneuver
                    )
                    pending_tests.append((res_obj, total_steps)) # 记录任务对象和触发时的步数
                
                trigger += trigger_delta # 更新下一次触发阈值

            # --- 结果轮询记录 (非阻塞) ---
            if len(pending_tests) > 0:
                finished_tasks = []
                for task in pending_tests:
                    res_obj, recorded_step = task
                    if res_obj.ready(): # 检查进程是否跑完
                        # [修改] 直接获取结果，不再包裹 try-except
                        rule_num, outcome = res_obj.get()
                        # 在主进程的 Logger 中记录结果
                        logger.add(f"test/agent_vs_rule{rule_num}", outcome, recorded_step)
                        print(f"  [Async Test Result] Rule_{rule_num}: {outcome} (Triggered at {recorded_step})")
                        finished_tasks.append(task)
                
                # 清理已完成的任务
                for task in finished_tasks:
                    pending_tests.remove(task)
                

            # --- [Modified] 对手选择逻辑 (Bypass & PFSP) ---
            ego_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
            enm_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
            
            # 1. 计算当前排位 rank_pos
            valid_elo_values = [v for k, v in elo_ratings.items() if not k.startswith("__")]
            if not valid_elo_values:
                rank_pos = 0.5
                min_elo, max_elo = main_agent_elo, main_agent_elo
            else:
                min_elo = np.min(valid_elo_values)
                max_elo = np.max(valid_elo_values)
                denom = float(max_elo - min_elo)
                # 如果分母为0，视为0.5中位
                if denom == 0.0:
                    rank_pos = 0.5
                else:
                    rank_pos = float((main_agent_elo - min_elo) / denom)

            # else:
            # 策略 B & C: 使用 Elo 概率，但可能对 Rule_0 进行加权
            probs, opponent_keys = get_opponent_probabilities(elo_ratings, target_elo=main_agent_elo, SP_type=self_play_type, sigma=sigma_elo)


            # 最终根据概率采样
            selected_opponent_name = np.random.choice(opponent_keys, p=probs)
            
            
            # 判断对手类型并加载
            if "Rule" in selected_opponent_name:
                adv_is_rule = True
                # 解析 "Rule_1" -> 1
                try:
                    rule_num = int(selected_opponent_name.split('_')[1])
                except:
                    rule_num = 0
                # 在测试模式下 Rule_1/Rule_2 可能不在 elo_ratings 中，避免直接索引导致 KeyError
                if selected_opponent_name in elo_ratings:
                    print(f"Eps {i_episode}: Opponent is {selected_opponent_name} (ELO: {elo_ratings[selected_opponent_name]:.0f})")
                else:
                    print(f"Eps {i_episode}: Opponent is {selected_opponent_name} (test-only, no ELO entry)")
            else:
                adv_is_rule = False
                # 尝试找到对应的权重文件
                # 假设 selected_opponent_name 格式为 "actor_rein10"
                adv_path = os.path.join(log_dir, f"{selected_opponent_name}.pt")
                if os.path.exists(adv_path):
                    adv_agent.actor.load_state_dict(torch.load(adv_path, map_location=device, weights_only=1), strict=False)
                    print(f"Eps {i_episode}: Opponent Loaded from {selected_opponent_name} (ELO: {elo_ratings[selected_opponent_name]:.0f})")
                else:
                    print(f"Warning: Opponent file {adv_path} not found. Fallback to Rule_0.")
                    adv_is_rule = True
                    rule_num = 0
            
            episode_return = 0

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = create_initial_state(randomized=randomized_birth)

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            
            r_action_label = 0
            b_action_label = 0
            
            # 修改：分别记录决策时的 Obs (Actor用) 和 State (Critic用)
            last_decision_obs = None 
            last_decision_state = None
            last_enm_decision_obs = None
            last_enm_decision_state = None
            
            current_action = None

            current_action_exec = None
            current_enm_action_exec = None
            
            b_rew_event, b_rew_constraint, b_rew_shaping = 0,0,0
            
            # 新增：每回合的死亡查询表（0 表示存活，1 表示已记录死亡瞬间）
            dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()
            
            last_r_action_label = 0
            last_b_action_label = 0
            
            steps_of_this_eps = -1
            
            m_fired = 0
            enemy_m_fired = 0
            fired_at_bad_condition = 0
            
            # --- Episode Loop ---
            for count in range(round(args.max_episode_len / dt_maneuver)):
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done:
                    break
                # 修改：同时获取局部观测(pomdp=1)和全局状态(pomdp=0)
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1) # Actor Input
                b_state_global, _ = env.obs_1v1('b', reward_fn=1)  # Critic Input
                r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    if steps_of_this_eps > 0: # and not dead_dict['b']:
                        # 修改：传入 last_decision_obs 和 last_decision_state
                        transition_dict = append_experience(transition_dict, last_decision_obs, last_decision_state, current_action, reward_for_learn, b_state_global, False, not dead_dict['b'])
                        # 保存当前对局中的状态转移
                        ego_transition_dict = append_experience(ego_transition_dict, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, b_state_global, False, not dead_dict['b'])
                        enm_transition_dict = append_experience(enm_transition_dict, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, r_state_global, False, not dead_dict['r'])

                        '''todo 引入active_mask'''
                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    # 修改：更新 obs 和 state 两个变量
                    last_decision_obs = b_obs
                    last_decision_state = b_state_global
                    
                    last_enm_decision_obs = r_obs
                    last_enm_decision_state = r_state_global
                    # 2. student_agent 产生一个动作
                    
                    # 红方(对手决策)
                    r_state_check = env.unscale_state(r_check_obs)
                    if adv_is_rule:
                        # [Fix] 传入选定的 rule_num
                        r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=last_r_action_label)
                        r_action_exec = {'cat': None, 'bern': None}
                        r_action_exec['cat'] = np.array([r_action_label])
                        r_action_exec['bern'] = np.array([r_fire], dtype=np.float32)
                    else:
                        # [Fix] NN 对手决策
                        r_action_exec, r_action_raw, _, r_action_check = adv_agent.take_action(r_obs, explore=1)
                        r_action_label = r_action_exec['cat'][0]
                        r_fire = r_action_exec['bern'][0] # 网络控制开火
                    last_r_action_label = r_action_label
                    r_m_id = None
                    if r_fire:
                        r_m_id = launch_missile_immediately(env, 'r')
                        
                    r_missile_fired = r_m_id is not None
                    
                    if r_missile_fired:
                        enemy_m_fired += 1
                        r_ATA = r_check_obs['target_information'][4]
                        if r_ATA > pi/2:
                            fired_at_bad_condition += 1
                    
                    # --- 蓝方 (训练对象) 决策 ---
                    b_state_check = env.unscale_state(b_check_obs)
                    # 修改：Actor 依然使用 b_obs (局部观测) 进行决策
                    
                    # --- 新增：测试模式下使用确定性动作 ---
                    explore_rate = 1

                    b_action_exec, b_action_raw, _, b_action_check = student_agent.take_action(b_obs, explore=explore_rate)
                    b_action_label = b_action_exec['cat'][0]
                    b_fire = b_action_exec['bern'][0]

                    b_m_id = None
                    if b_fire:
                        b_m_id = launch_missile_immediately(env, 'b')
                        # print(b_m_id)
                        
                    b_missile_fired = b_m_id is not None
                    if b_m_id is not None:
                        m_fired += 1

                    # if i_episode % 2 == 0:
                    #     b_action_label = 12 # debug
                    
                    # print("机动概率分布", b_action_check['cat'])
                    # print("开火概率", b_action_check['bern'][0])
                    
                    decide_steps_after_update += 1
                    
                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    # PPO需要的是开火意图
                    current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                    
                    # IL需要的是实际的开火执行情况
                    current_action_exec = {'cat': b_action_exec['cat'], 'bern': np.array([b_missile_fired])}
                    current_enm_action_exec = {'cat': r_action_exec['cat'], 'bern': np.array([r_missile_fired])}
                    
                if adv_is_rule:
                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label) # 同步动作空间，现在都是区分左右
                else:
                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)

                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)

                env.step(r_maneuver, b_maneuver)
                done, b_rew_event, b_rew_constraint, b_rew_shaping = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier)
                _, r_rew_event, r_rew_constraint, r_rew_shaping = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier)
                
                reward_for_show = b_rew_event + b_rew_constraint
                
                weight_reward = weight_reward_0
                # weight_reward[2] = 0.0
                
                reward_for_learn = sum(np.array([b_rew_event, b_rew_constraint, b_rew_shaping]) * weight_reward)
                reward_for_enm = sum(np.array([r_rew_event, r_rew_constraint, r_rew_shaping]) * weight_reward)
                
                # Accumulate rewards between student_agent decisions
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    episode_return += reward_for_show
                    # print(b_rew_event, b_rew_constraint, b_rew_shaping)
                    # print()
                
                # if dead_dict['b'] == 0:
                # 修改：在死亡检测时，如果存活，也需要同时更新 next_b_obs 和 next_b_state_global 用于回合结束时的存储
                # next_b_obs, next_b_check_obs = env.obs_1v1('b', pomdp=1)
                next_b_state_global, _ = env.obs_1v1('b', reward_fn=1) # 获取全局Next State
                next_r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

                
                total_steps += 1
                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            # # --- 回合结束处理 ---
            # [修复] 胜负漏记问题 强制在回合结束后调用一次终止判定，以获取最终的胜负状态
            # 无论循环是正常结束(超时)还是break(done=True)，都重新获取一次最终结果
            done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False, action_cycle_multiplier)
            
            # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
            # 循环结束，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
            if last_decision_state is not None:
                # --- 新增：测试模式下不存储经验 ---
                # # 若在回合结束前未曾在死亡瞬间计算 next_b_obs（例如超时终止或其他非击毁终止），做一次后备计算
                # 修改：传入最后时刻的 next_b_state_global 作为 Next State
                transition_dict = append_experience(transition_dict, last_decision_obs, last_decision_state, current_action, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                
                ego_transition_dict = append_experience(ego_transition_dict, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                enm_transition_dict = append_experience(enm_transition_dict, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, next_r_state_global, True, not dead_dict['r'])
                episode_return += reward_for_show
                
            print('r 剩余导弹数量:', env.RUAV.ammo)
            print('r 发射时间:', env.RUAV.missile_launch_time)
            
            print('b 剩余导弹数量:', env.BUAV.ammo)
            print('b 发射时间:', env.BUAV.missile_launch_time)
            
            if env.crash(env.RUAV):
                print('r 撞地')
            if env.crash(env.BUAV):
                print('b 撞地')
            
            # --- [Optimized] ELO 更新与策略池清洗逻辑 (只在训练回合执行) ---
            
            if env.win: actual_score = 1.0
            elif env.lose: actual_score = 0.0
            else: 
                env.draw=1
                actual_score = 0.5

            prev_main_elo = main_agent_elo
            
            # 1. 判定对手是否为“待踢出”类型 (只针对非规则智能体)
            is_rule_agent = "Rule" in selected_opponent_name
            is_kicked_opponent = False
            
            # # 踢出判定条件：
            # if not (is_rule_agent or selected_opponent_name==init_opponent_name):
            #     '''
            #     简单粗暴的对手筛选策略：
            #     1、撞地
            #     2、零开火且未获胜
            #     3、所有导弹均背对目标开火
            #     '''
            #     cond_crash = env.crash(env.RUAV) # 撞地
            #     r_fired_count = 6 - env.RUAV.ammo
            #     cond_coward = (r_fired_count == 0 and env.win) # 0弹且未取胜 (蓝方赢)
            #     blind_shot = (fired_at_bad_condition == enemy_m_fired)
                
            #     if cond_crash or cond_coward or blind_shot:
            #         is_kicked_opponent = True
            #         print(f"\n[Pool Filter] Found opponent for KICKING: {selected_opponent_name}")


            # 2. ELO 更新与记录逻辑
            if selected_opponent_name in elo_ratings and not is_kicked_opponent:
                # A. 正常更新 ELO：对手合格，进行 ELO 结算
                adv_elo = elo_ratings[selected_opponent_name]
                main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score)
                
                # 更新对手 ELO
                new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score)
                elo_ratings[selected_opponent_name] = new_adv_elo
                
                print(f"  Result: Score={actual_score}, Main ELO: {prev_main_elo:.0f}->{main_agent_elo:.0f}, Adv ELO: {adv_elo:.0f}->{new_adv_elo:.0f}")
                
            elif is_kicked_opponent:
                # B. 忽略 ELO 更新：对手不合格，主智能体的 ELO 保持不变
                # 保持 main_agent_elo == prev_main_elo
                print(f"  Result: Score={actual_score}, Main ELO: {prev_main_elo:.0f} (No Change due to Opponent Filter).")
                
            else:
                # C. 极端情况
                print(f"Warning: Opponent {selected_opponent_name} not found in ELO dict. ELO not updated.")


            # 3. 执行踢出操作
            if is_kicked_opponent and should_kick:
                if selected_opponent_name in elo_ratings:
                    del elo_ratings[selected_opponent_name]
                    print(f"  Opponent {selected_opponent_name} has been removed from the ELO pool.")


            # 有没有试图发射过导弹
            logger.add("special/0 发射的导弹数量", m_fired, total_steps)
            # 每一场胜负变化
            logger.add("train/1 episode_return", episode_return, total_steps)
            logger.add("train/2 win", env.win, total_steps)
            logger.add("train/2 lose", env.lose, total_steps)
            logger.add("train/2 draw", env.draw, total_steps)
            logger.add("debug/胜负统计", env.win+env.lose+env.draw, total_steps)  # debug 和不为1
            logger.add("train/11 episode/step", i_episode, total_steps)
            
            if is_testing == False and use_sil:
                # 添加当前回合回放信息和对手回放信息
                # 赢了学自己，输了学对手
                if not env.lose:
                    # 自己
                    new_il_transition_dict = {'obs':[], 'states':[], 'actions': [], 'returns': []}
                    new_il_transition_dict['obs'] = ego_transition_dict['obs']
                    new_il_transition_dict['states'] = ego_transition_dict['states']
                    new_il_transition_dict['actions'] = ego_transition_dict['actions']
                    # 将状态转移处理成蒙特卡洛回报形式
                    new_il_transition_dict['returns'] = compute_monte_carlo_returns(gamma, \
                                                                                ego_transition_dict['rewards'], \
                                                                                ego_transition_dict['dones'])
                    il_transition_buffer.add(new_il_transition_dict)
                
                # 对手
                if not env.win:
                    new_il_transition_dict = {'obs':[], 'states':[], 'actions': [], 'returns': []}
                    new_il_transition_dict['obs'] = enm_transition_dict['obs']
                    new_il_transition_dict['states'] = enm_transition_dict['states']
                    new_il_transition_dict['actions'] = enm_transition_dict['actions']
                    # 将状态转移处理成蒙特卡洛回报形式
                    new_il_transition_dict['returns'] = compute_monte_carlo_returns(gamma, \
                                                                                enm_transition_dict['rewards'], \
                                                                                enm_transition_dict['dones'])
                    il_transition_buffer.add(new_il_transition_dict)


            # --- RL Update ---
            if len(transition_dict['dones'])>=transition_dict_threshold: 
                if use_sil:
                    #===========================================
                    # 混合强化学习与模仿学习
                    il_transition_dict = il_transition_buffer.read(il_batch_size2)
                    # 1. 定义 IL 衰减的最大轮次
                    # 使用浮点数以确保计算精度
                    # MAX_IL_EPISODE = 500 # 100.0 
                    # 2. 计算当前 IL 权重 alpha_il (线性衰减，确保不小于 0)
                    # 当 i_episode = 0 时，alpha_il = 1.0
                    # 当 i_episode = 100 时，alpha_il = 0.0
                    # alpha_il = 1.0 # max(0.0, 1.0 - i_episode / MAX_IL_EPISODE)
                    # 3. 调用混合更新函数，传入计算出的 alpha
                    student_agent.mixed_update(
                        transition_dict,          # RL 数据
                        il_transition_dict,       # IL 数据
                        init_il_transition_dict = original_il_transition_dict0 if use_init_data else None,
                        eta = np.clip(1-total_steps/3e6, 0, 1),
                        adv_normed=True,          # 沿用 RL 实例中的优势归一化
                        il_batch_size=None,        # 沿用 IL 实例中的 Batch Size 128
                        label_smoothing=label_smoothing_mixed,      # 沿用 IL 实例中的标签平滑
                        alpha=alpha_il,           # 核心：传入随时间衰减的权重
                        beta=beta_mixed,                  # 沿用 IL 实例中的 beta
                        sil_only_maneuver = sil_only_maneuver,
                        mini_batch_size = mini_batch_size_mixed
                    )
                else:
                    #====================
                    # 原有强化学习部分
                    student_agent.update(transition_dict, adv_normed=1, mini_batch_size=64)  # 优势归一化 debug
                    #====================
                decide_steps_after_update = 0

                # [Modification] 保留原有梯度监控代码
                actor_pre_clip_grad = student_agent.pre_clip_actor_grad
                critic_pre_clip_grad = student_agent.pre_clip_critic_grad

                # 梯度监控
                logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
                logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
                # 损失函数监控
                logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
                logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
                # 强化学习actor特殊项监控
                logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
                logger.add("train/9 entropy_cat", student_agent.entropy_cat, total_steps)
                logger.add("train/9 entropy_bern", student_agent.entropy_bern, total_steps)
                
                logger.add("train/10 advantage", student_agent.advantage, total_steps) 
                # 强化学习
                logger.add("train/10 explained_var", student_agent.explained_var, total_steps)
                logger.add("train/10 approx_kl", student_agent.approx_kl, total_steps)
                logger.add("train/10 clip_frac", student_agent.clip_frac, total_steps)
                
                # IL-PPO信号强度对比
                if use_sil:
                    logger.add("train_plus/原始信号强度对比IL-PPO", student_agent.IL_samples/student_agent.PPO_samples*alpha_il, total_steps)
                    logger.add("train_plus/滤波后信号强度对比IL-PPO", student_agent.IL_valid_samples/student_agent.PPO_valid_samples*alpha_il, total_steps)
                
                # 修改：重置 transition_dict 时保留 obs 键
                transition_dict = copy.deepcopy(empty_transition_dict)
                
                
            return_list.append(episode_return)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t

            print(f"Episode {i_episode}, Progress: {total_steps/current_max_steps:.3f}, Curr Return: {episode_return}")
            print()

            
            
            # --- 保存模型与 ELO 维护 ---
            if i_episode % 5 == 0:
                torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                
                actor_name = f"actor_rein{i_episode}.pt" # 物理文件名
                actor_key = f"actor_rein{i_episode}"     # ELO 字典 Key
                
                # 保存权重
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, actor_name))
                print(f"Saved critic.pt and {actor_name} at episode {i_episode}")

                # # 定义文件路径
                # full_json_path = os.path.join(log_dir, "elo_ratings.json")       # 全量历史（用于分析）
                # elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json") # 精英池（用于采样）
                
                # -----------------------------------------------------------
                # 逻辑分支 A: 维护“内存中的精英池” (elo_ratings)
                # -----------------------------------------------------------
                # 只有开启了自博弈开关，且步数超过冷启动阈值 (500k)，才允许扩充池子
                admission_status = "Skipped" # 用于日志打印状态

                # 将当前主智能体作为一个新的历史快照加入 ELO 列表
                # 新快照继承当前主智能体的 ELO
                if hist_agent_as_opponent and total_steps >= WARM_UP_STEPS:
                    # elo_ratings[actor_key] = main_agent_elo
                    rule_elos = [v for k, v in elo_ratings.items() if k.startswith("Rule")]
                    
                    # 防御性逻辑：防止还没有规则数据
                    if not rule_elos:
                        rule_min, rule_max = 0, 0
                        rank_pos = 1.0 
                    else:
                        rule_min = np.min(rule_elos)
                        rule_max = np.max(rule_elos)
                        denom = float(rule_max - rule_min)
                        rank_pos = 0.5 if denom == 0 else (main_agent_elo - rule_min) / denom

                    # 2. 准入判断 (50% 位次) 
                    if rank_pos >= ADMISSION_THRESHOLD:
                        # --- 容量控制 (淘汰机制) ---
                        
                        # 找出所有现存的历史智能体 (排除 Rules 和 特殊key)
                        history_keys = [k for k in elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")]
                        
                        # if len(history_keys) >= MAX_HISTORY_SIZE:
                        #     # 淘汰最弱的历史智能体
                        #     weakest_history_key = min(history_keys, key=lambda k: elo_ratings[k])
                        #     del elo_ratings[weakest_history_key]
                        #     print(f"[Pool Cleanup] Kicked weakest: {weakest_history_key} (Elo: {elo_ratings.get(weakest_history_key, 'N/A')})")
                        # 使用 while 循环确保：即使积压了多个过剩智能体，也能一次性清理到位
                        while len(history_keys) >= MAX_HISTORY_SIZE:
                            # 每次找到当前池子中最弱的一个
                            weakest_history_key = min(history_keys, key=lambda k: elo_ratings[k])
                            old_elo = elo_ratings[weakest_history_key]
                            # 从 ELO 字典和局部列表中同步删除
                            del elo_ratings[weakest_history_key]
                            history_keys.remove(weakest_history_key)
                            print(f"[Pool Cleanup] Kicked weakest: {weakest_history_key} (Elo: {old_elo:.0f}), Current Pool: {len(history_keys)}")
                        
                        # --- 正式入池 ---
                        elo_ratings[actor_key] = main_agent_elo
                        admission_status = f"Accepted (Rank {rank_pos:.2f})"
                    else:
                        admission_status = f"Rejected (Rank {rank_pos:.2f} < {ADMISSION_THRESHOLD})"
                
                elif total_steps < WARM_UP_STEPS:
                    admission_status = f"Locked (Warm-up {total_steps}/{int(WARM_UP_STEPS)})"

                print(f"[Pool Status] Agent {actor_key}: {admission_status}")

                # -----------------------------------------------------------
                # 逻辑分支 B: 维护“全量历史记录” (Full JSON)
                # -----------------------------------------------------------
                # 目标：记录所有产生过的 Agent 的最后一次已知 Elo，无论它是否在精英池里
                try:
                    full_history_data = {}
                    # 1. 尝试读取旧的全量数据
                    if os.path.exists(full_json_path):
                        with open(full_json_path, 'r', encoding='utf-8') as f:
                            try:
                                full_history_data = json.load(f)
                            except json.JSONDecodeError:
                                full_history_data = {} # 文件损坏则重置
                    
                    # 2. 用当前内存中的精英池更新全量数据 (因为规则和幸存的精英 Elo 变了)
                    for k, v in elo_ratings.items():
                        if not k.startswith("__"):
                            full_history_data[k] = v
                    
                    # 3. 强制把【当前】Agent 写入全量数据 (即使它被拒绝入池，也要记录它来过)
                    full_history_data[actor_key] = main_agent_elo
                    
                    # 4. 记录当前元数据
                    full_history_data["__CURRENT_MAIN__"] = main_agent_elo
                    full_history_data["__LAST_UPDATE_STEP__"] = total_steps
                    
                    # 5. 保存全量日志
                    with open(full_json_path, "w", encoding="utf-8") as f:
                        json.dump(full_history_data, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"Warning: failed to save FULL elo json: {e}")

                # -----------------------------------------------------------
                # 逻辑分支 C: 保存“精英池快照” (Elite JSON)
                # -----------------------------------------------------------
                # 这才是下次训练 resume 时应该读取的文件
                try:
                    save_data = copy.deepcopy(elo_ratings)
                    save_data["__CURRENT_MAIN__"] = main_agent_elo 
                    
                    # 为了原子性写入，先写临时文件再 rename
                    tmp_path = elite_json_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, elite_json_path)
                except Exception as e:
                    print(f"Warning: failed to save ELITE elo json: {e}")

                # --- 日志记录 (Logging) - 保持不变，展示的是精英池状态 ---
                valid_elos = {k: v for k, v in elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    denom = float(max_elo - min_elo)
                    if denom == 0.0:
                        rank_pos = 0.5
                    else:
                        rank_pos = float((main_agent_elo - min_elo) / denom)
                    # 现有日志
                    logger.add("Elo_Centered/Current_Rank %", rank_pos*100, total_steps)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps)
                    
                    rule_vals = [v for k, v in valid_elos.items() if k.startswith("Rule")]
                    if rule_vals:
                        r_min, r_max = np.min(rule_vals), np.max(rule_vals)
                        denom = float(r_max - r_min)
                        curr_rank = 0.5 if denom == 0 else (main_agent_elo - r_min) / denom
                        logger.add("Elo_Centered/Current_Rank %", curr_rank * 100, total_steps)
                    
                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps)

                    # 记录详细分数
                    keys_to_log = [k for k in sorted(valid_elos.keys()) if k.startswith("Rule_")]
                    if actor_key in valid_elos and actor_key not in keys_to_log:
                        keys_to_log.append(actor_key)
                    
                    for k in keys_to_log:
                        tag_suffix = k if k.startswith("Rule_") else "Latest_Saved"
                        logger.add(f"Elo_Raw/{tag_suffix}", valid_elos[k], total_steps)
                    
                    # 只记录所有规则智能体和最新保存的智能体（actor_key）
                    rule_keys = [k for k in sorted_keys if k.startswith("Rule_")]
                    keys_to_log = list(sorted(rule_keys))
                    # actor_key 在本代码块上方已定义为当前保存的快照名
                    if 'actor_key' in locals() and actor_key in valid_elos and actor_key not in keys_to_log:
                        keys_to_log.append(actor_key)
                        
                    for k in keys_to_log:
                        # 如果是规则智能体，使用其自身名字
                        if k.startswith("Rule_"):
                            raw_tag = f"Elo_Raw/{k}"
                            centered_tag = f"Elo_Centered/{k}"
                        # 否则，认为是最新智能体，使用固定标签 "Latest"
                        else:
                            raw_tag = "Elo_Raw/Latest"
                            centered_tag = "Elo_Centered/Latest"
                        
                        logger.add(raw_tag, valid_elos[k], total_steps)
                        logger.add(centered_tag, valid_elos[k] - mean_elo, total_steps)

                    # --- 插入: 记录 Latest(当前保存的 actor_key) 相对于所有存在的 Rule_* 的 ELO 差值 ---
                    if 'actor_key' in locals() and actor_key in valid_elos:
                        latest_elo = float(valid_elos[actor_key])
                        rule_keys_present = [rk for rk in valid_elos.keys() if rk.startswith("Rule_")]
                        # diffs = []
                        for rk in rule_keys_present:
                            diff = latest_elo - float(valid_elos[rk])
                            # diffs.append(diff)
                            logger.add(f"Elo_Diff/Latest_vs_{rk}", diff, total_steps)
                            
                # ############# 旧逻辑 #####################################     
                # try:
                #     tmp_path = elo_json_path + ".tmp"
                #     with open(tmp_path, "w", encoding="utf-8") as f:
                #         # 同时保存 current main agent elo 以便断点续传
                #         save_data = copy.deepcopy(elo_ratings)
                #         save_data["__CURRENT_MAIN__"] = main_agent_elo 
                #         json.dump(save_data, f, ensure_ascii=False, indent=2)
                #     os.replace(tmp_path, elo_json_path)
                # except Exception as e:
                #     print(f"Warning: failed to save elo json: {e}")

                # # --- [Fix] Logging ELOs (Sorted) ---
                # # 过滤掉特殊 Key
                # valid_elos = {k: v for k, v in elo_ratings.items() if not k.startswith("__")}
                # if valid_elos:
                #     mean_elo = np.mean(list(valid_elos.values()))
                #     # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                #     # 这里为了简单，直接遍历
                #     sorted_keys = sorted(valid_elos.keys())
                    
                #     # 记录主智能体
                #     logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps)

                #     # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                #     # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                #     min_elo = np.min(list(valid_elos.values()))
                #     max_elo = np.max(list(valid_elos.values()))
                #     denom = float(max_elo - min_elo)
                #     if denom == 0.0:
                #         rank_pos = 0.5
                #     else:
                #         rank_pos = float((main_agent_elo - min_elo) / denom)
                #     # 现有日志
                #     logger.add("Elo_Centered/Current_Rank %", rank_pos*100, total_steps)
                    
                #     # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                #     elo_spread = float(max_elo - min_elo)
                #     print('elo分极差：', elo_spread)
                #     logger.add("Elo/Spread", elo_spread, total_steps)
                    
                #     # 只记录所有规则智能体和最新保存的智能体（actor_key）
                #     rule_keys = [k for k in sorted_keys if k.startswith("Rule_")]
                #     keys_to_log = list(sorted(rule_keys))
                #     # actor_key 在本代码块上方已定义为当前保存的快照名
                #     if 'actor_key' in locals() and actor_key in valid_elos and actor_key not in keys_to_log:
                #         keys_to_log.append(actor_key)
                    
                #     for k in keys_to_log:
                #         # 如果是规则智能体，使用其自身名字
                #         if k.startswith("Rule_"):
                #             raw_tag = f"Elo_Raw/{k}"
                #             centered_tag = f"Elo_Centered/{k}"
                #         # 否则，认为是最新智能体，使用固定标签 "Latest"
                #         else:
                #             raw_tag = "Elo_Raw/Latest"
                #             centered_tag = "Elo_Centered/Latest"
                        
                #         logger.add(raw_tag, valid_elos[k], total_steps)
                #         logger.add(centered_tag, valid_elos[k] - mean_elo, total_steps)

                #     # --- 插入: 记录 Latest(当前保存的 actor_key) 相对于所有存在的 Rule_* 的 ELO 差值 ---
                #     if 'actor_key' in locals() and actor_key in valid_elos:
                #         latest_elo = float(valid_elos[actor_key])
                #         rule_keys_present = [rk for rk in valid_elos.keys() if rk.startswith("Rule_")]
                #         # diffs = []
                #         for rk in rule_keys_present:
                #             diff = latest_elo - float(valid_elos[rk])
                #             # diffs.append(diff)
                #             logger.add(f"Elo_Diff/Latest_vs_{rk}", diff, total_steps)
                        
        # --- [新增] 达到 max_steps 后的交互逻辑 ---
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        try:
            new_max_steps_input = input(f"Enter new max_steps to continue training, or press Enter to exit (current total steps: {total_steps}): ")
            
            if not new_max_steps_input.strip():
                print("No input provided. Exiting training.")
                break # 退出外层 while True 循环

            new_max_steps = int(new_max_steps_input)

            if new_max_steps > total_steps:
                current_max_steps = new_max_steps
                print(f"Continuing training until {current_max_steps} steps.")
            else:
                print(f"Input ({new_max_steps}) is not greater than current steps ({total_steps}). Exiting training.")
                break # 退出外层 while True 循环

        except ValueError:
            print("Invalid input. Please enter an integer. Exiting training.")
            break # 退出外层 while True 循环
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted by user. Exiting training.")
            break
    
    # End Training
    training_end_time = time.time()
    env.end_render()
    print("Total Steps Reached. Training Finished.")
    
    # [新增] 训练结束后，如果还有未完成的测试，等待它们跑完（可选）
    if len(pending_tests) > 0:
        print(f"Waiting for {len(pending_tests)} pending tests to finish...")
        for res_obj, recorded_step in pending_tests:
            rule_num, outcome = res_obj.get() # 阻塞直至完成
            logger.add(f"test/agent_vs_rule{rule_num}", outcome, recorded_step)
            print(f"  [Finalizing Test] Rule_{rule_num}: {outcome}")

    logger.close()
    
    # [修改] 关闭进程池
    test_pool.close()
    test_pool.join()