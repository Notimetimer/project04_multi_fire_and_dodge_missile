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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Algorithms.Wasted.PPOHybrid24 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from BasicRules import *
from Envs.Tasks.ChooseStrategyEnv3 import *

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
        print("Restructuring actions from List[Dict] to Dict[Array]...")
        
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
                if 'fly' in act:
                    new_actions['cat'].append(act['fly'])
                if 'fire' in act:
                    new_actions['bern'].append(act['fire'])
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
            
def append_b_experience(td, obs, state, action, reward, next_state, done):
    """
    统一把一次蓝方经验追加到 transition_dict。
    修改：增加 obs 输入，用于存储局部观测
    """
    td['obs'].append(obs) # 新增：存储Actor用的局部观测
    td['states'].append(state) # 修改：这里存储Critic用的全局状态(pomdp=0)
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state) # 修改：这里存储Critic用的全局Next状态(pomdp=0)
    td['dones'].append(done)
    return td

# 加载数据
il_transition_dict, transition_dict = load_il_and_transitions(
    os.path.join(cur_dir, "IL"),
    "il_transitions_combat.pkl",
    "transition_dict_combat.pkl"
)

# --- 关键步骤：执行数据重构 ---
if il_transition_dict is not None:
    # 这里完成 (Batch, Key) -> (Key, Batch) 的转换
    il_transition_dict['actions'] = restructure_actions(il_transition_dict['actions'])
    
    # 顺便确保 states 和 returns 也是标准的 float32 numpy array
    il_transition_dict['states'] = np.array(il_transition_dict['states'], dtype=np.float32)
    il_transition_dict['returns'] = np.array(il_transition_dict['returns'], dtype=np.float32)

# ------------------------------------------------------------------
# 参数与环境配置
# ------------------------------------------------------------------
parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=10*60, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=55e3, help="")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4 # 4 1e-3
critic_lr = actor_lr * 5 # * 5
IL_epoches= 0  # 80 检查一下，这个模仿学习可能有问题!!!
max_steps = 4 * 165e4
hidden_dim = [128, 128, 128]
gamma = 0.95
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy={'cont':0.01, 'cat':0.01, 'bern':0.08} # 1 # 0.05 # 给MSE用，这个项需要大一些来把熵压在目标熵附近

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
    max_std=0.3
)

adv_agent = copy.deepcopy(student_agent)



if __name__ == "__main__":
    
    if il_transition_dict is None:
        print("No il_transitions_combat file found.")
        sys.exit(1)

    summarize(il_transition_dict)

    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    mission_name = 'RL_combat_PFSP_熵裁剪'
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

    # # 模仿学习训练循环
    # # 现在 il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # # 能够被 MARWIL_update 里的 items() 正常遍历
    # for epoch in range(IL_epoches): 
    #     avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
    #         il_transition_dict, 
    #         beta=1.0, 
    #         batch_size=128, # 显存如果够大可以适当调大
    #         label_smoothing=0.3
    #     )
        
    #     # 记录
    #     if epoch % 1 == 0:
    #         logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
    #         logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
    #         # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

    #         print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    # print("Training Finished.")
    
    # 加载打靶预训练的智能体
    from Utilities.LocateDirAndAgents2 import *
    pre_train_logs_root_dir = os.path.join(project_root, "logs/combat")
    pre_train_latest_log_dir = get_latest_log_dir(pre_train_logs_root_dir, "ILRL_combat_打rule0带导弹")
    pre_train_agent_path = find_latest_agent_path(pre_train_latest_log_dir)
    
    if pre_train_agent_path:
        print(f"Loading Actor from: {pre_train_agent_path}")
        # [Fix] 添加 strict=False 以兼容旧权重文件（忽略缺失的 log_temp 参数）
        actor_wrapper.load_state_dict(torch.load(pre_train_agent_path, map_location=device, weights_only=1), strict=False)
    
    # 加载 critic
    critic_path = os.path.join(pre_train_latest_log_dir, "critic.pt")
    if os.path.exists(critic_path):
        print(f"Loading Critic from: {critic_path}")
        student_agent.critic.load_state_dict(torch.load(critic_path, map_location=device, weights_only=1), strict=False)
    else:
        print(f"Warning: Critic file {critic_path} not found.")

    # ==============================================================================
    # 强化学习 (Self-Play / PFSP) 阶段
    # ==============================================================================
    
    # # 根据参数数量缩放学习率
    # from Math_calculates.ScaleLearningRate import scale_learning_rate
    # actor_lr = scale_learning_rate(actor_lr, student_agent.actor)
    # critic_lr = scale_learning_rate(critic_lr, student_agent.critic)
    # student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    
    tacview_input = input("Enable tacview visualization? (0=no, 1=yes) [default 0]: ").strip()
    if tacview_input == "":
        tacview_show = 0
    else:
        try:
            tacview_show = 1 if int(tacview_input) != 0 else 0
        except Exception:
            tacview_show = 0
    print(f"tacview_show={tacview_show}")
    env = ChooseStrategyEnv(args, tacview_show=tacview_show)
    env.shielded = 1 # 不得不全程带上，否则对手也会撞地
    env.dt_move = 0.05 # 仿真跑得快点
    
    t_bias = 0
    
    
    def create_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 45e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier
    transition_dict_threshold = 5 * env.args.max_episode_len//dt_action_cycle + 1 

    # --- [Fix] 初始化 ELO 变量与辅助函数 ---
    K_FACTOR = 32
    INITIAL_ELO = 1200
    
    # 初始化 ELO 字典，包含基础规则智能体
    elo_ratings = {
        "Rule_0": INITIAL_ELO,
        # "Rule_1": INITIAL_ELO,
        # "Rule_2": INITIAL_ELO
    }
    elo_json_path = os.path.join(log_dir, "elo_ratings.json")
    
    # 尝试加载已有的 ELO 记录
    if os.path.exists(elo_json_path):
        with open(elo_json_path, 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
            
    # 主智能体当前的 ELO
    main_agent_elo = INITIAL_ELO

    # [新增] 如果没有历史对手（例如第一次运行且屏蔽了规则），保存当前初始策略作为 actor_rein0
    if (not elo_ratings) or IL_epoches>0:
        init_opponent_name = "actor_rein0"
        init_opponent_path = os.path.join(log_dir, f"{init_opponent_name}.pt")
        torch.save(student_agent.actor.state_dict(), init_opponent_path)
        elo_ratings[init_opponent_name] = INITIAL_ELO
        print(f"Initialized {init_opponent_name} as the first opponent.")

    def calculate_expected_score(player_elo, opponent_elo):
        """计算期望得分"""
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

    def update_elo(player_elo, opponent_elo, score):
        """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
        expected = calculate_expected_score(player_elo, opponent_elo)
        return player_elo + K_FACTOR * (score - expected)

    def get_opponent_probabilities(elo_ratings, target_elo=None, sigma=400):
        """返回与 elo_ratings.keys() 顺序对应的概率数组。
        - target_elo: 要优先靠近的 ELO（传入 main_agent_elo）。
        - sigma: 高斯核标准差，越小越只选接近 target_elo 的对手。
        """
        keys = list(elo_ratings.keys())
        if len(keys) == 0:
            return np.array([]), keys # 返回 keys 以便索引
        elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

        if target_elo is None:
            target_elo = np.mean(elos)

        # 以高斯核度量相似度（基于差的平方）
        diffs = elos - float(target_elo)
        # 数值稳定性：将 exponent 的常数项减去 max
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
        return probs, keys # 修改为返回 probs 和 keys
    
    '''
    通过高斯核与elo分确定对手概率
    Δ = 200 (1σ = 200)：≈ 0.7597 → 75.97% ： 相似度 ≈ 0.6065，区域内概率0.683
    Δ = 400 (2σ = 400)：≈ 0.9091 → 90.91% ： 相似度 ≈ 0.1353，区域内概率0.955
    Δ = 600 (3σ = 600)：≈ 0.9693 → 96.93% ： 相似度 ≈ 0.0111，区域内概率0.997
    '''

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
    training_start_time = time.time()
    launch_time_count = 0
    t_bias = 0
    decide_steps_after_update = 0
    return_list = []
    
    r_action_list = []
    b_action_list = []
    
    # 修改：初始化增加 'obs' 键
    transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    
    while total_steps < int(max_steps):
        i_episode += 1
        
        # --- [Fix] 对手选择逻辑 (PFSP) ---
        probs, opponent_keys = get_opponent_probabilities(elo_ratings, target_elo=main_agent_elo)
        selected_opponent_name = np.random.choice(opponent_keys, p=probs)
        
        adv_is_rule = False
        rule_num = 0
        
        # 判断对手类型并加载
        if "Rule" in selected_opponent_name:
            adv_is_rule = True
            # 解析 "Rule_1" -> 1
            try:
                rule_num = int(selected_opponent_name.split('_')[1])
            except:
                rule_num = 0
            print(f"Eps {i_episode}: Opponent is {selected_opponent_name} (ELO: {elo_ratings[selected_opponent_name]:.0f})")
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

        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = create_initial_state()

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        
        r_action_label = 0
        b_action_label = 0
        
        # 修改：分别记录决策时的 Obs (Actor用) 和 State (Critic用)
        last_decision_obs = None 
        last_decision_state = None
        
        current_action = None
        b_reward = None
        
        # 新增：每回合的死亡查询表（0 表示存活，1 表示已记录死亡瞬间）
        dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

        done = False

        env.dt_maneuver = dt_maneuver
        
        episode_start_time = time.time()
        
        last_r_action_label = 0
        last_b_action_label = 0
        
        steps_of_this_eps = -1
        
        m_fired = 0
        
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
            

            # --- 智能体决策 ---
            # 判断是否到达了决策点（每 10 步）
            if steps_of_this_eps % action_cycle_multiplier == 0:
                # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                if steps_of_this_eps > 0 and not dead_dict['b']: # 临时接替一下 active mask
                    # 修改：传入 last_decision_obs 和 last_decision_state
                    transition_dict = append_b_experience(transition_dict, last_decision_obs, last_decision_state, current_action, b_reward, b_state_global, False)
                    
                    '''需要引入 active_mask以应对“死后还在做决策”的极端情况'''

                # **关键点 2: 开始【新的】一个动作周期**
                # 1. 记录新周期的起始状态
                # 修改：更新 obs 和 state 两个变量
                last_decision_obs = b_obs
                last_decision_state = b_state_global
                # 2. student_agent 产生一个动作
                
                # 红方(对手决策)
                r_state_check = env.unscale_state(r_check_obs)
                if adv_is_rule:
                    # [Fix] 传入选定的 rule_num
                    r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=last_r_action_label)
                else:
                    # [Fix] NN 对手决策
                    r_action_exec, r_action_raw, _, r_action_check = adv_agent.take_action(r_obs, explore=1)
                    r_action_label = r_action_exec['cat'][0]
                    r_fire = r_action_exec['bern'][0] # 网络控制开火
                last_r_action_label = r_action_label
                r_m_id = None
                if r_fire:
                    r_m_id = launch_missile_immediately(env, 'r')

                # --- 蓝方 (训练对象) 决策 ---
                b_state_check = env.unscale_state(b_check_obs)
                # 修改：Actor 依然使用 b_obs (局部观测) 进行决策
                b_action_exec, b_action_raw, _, b_action_check = student_agent.take_action(b_obs, explore=1)
                b_action_label = b_action_exec['cat'][0]
                b_fire = b_action_exec['bern'][0]

                b_m_id = None
                if b_fire:
                    b_m_id = launch_missile_immediately(env, 'b')
                    # print(b_m_id)
                if b_m_id is not None:
                    m_fired += 1

                # if i_episode % 2 == 0:
                #     b_action_label = 12 # debug
                
                # print("机动概率分布", b_action_check['cat'])
                # print("开火概率", b_action_check['bern'][0])
                
                decide_steps_after_update += 1
                
                b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                
                

            r_maneuver = env.maneuver14(env.RUAV, r_action_label)
            b_maneuver = env.maneuver14(env.BUAV, b_action_label)

            env.step(r_maneuver, b_maneuver)
            done, b_reward, b_reward_assisted = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None)
            done = done

            # Accumulate rewards between student_agent decisions
            if steps_of_this_eps % action_cycle_multiplier == 0:
                episode_return += b_reward
                # print(b_reward-b_reward_assisted, b_reward_assisted)
                # print()
            
            if dead_dict['b'] == 0:
                # 修改：在死亡检测时，如果存活，也需要同时更新 next_b_obs 和 next_b_state_global 用于回合结束时的存储
                next_b_obs, next_b_check_obs = env.obs_1v1('b', pomdp=1)
                next_b_state_global, _ = env.obs_1v1('b', reward_fn=1) # 获取全局Next State
                if env.BUAV.dead:
                    dead_dict['b'] = 1

            total_steps += 1
            '''显示运行轨迹'''
            # 可视化
            env.render(t_bias=t_bias)
        
        # # --- 回合结束处理 ---
        # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
        # 循环结束，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
        if last_decision_state is not None:
            # # 若在回合结束前未曾在死亡瞬间计算 next_b_obs（例如超时终止或其他非击毁终止），做一次后备计算
            # 修改：传入最后时刻的 next_b_state_global 作为 Next State
            transition_dict = append_b_experience(transition_dict, last_decision_obs, last_decision_state, current_action, b_reward, next_b_state_global, True)
            episode_return += b_reward
            
        print('r 剩余导弹数量:', env.RUAV.ammo)
        print('r 发射时间:', env.RUAV.missile_launch_time)
        
        print('b 剩余导弹数量:', env.BUAV.ammo)
        print('b 发射时间:', env.BUAV.missile_launch_time)
        
        if env.crash(env.RUAV):
            print('r 撞地')
        if env.crash(env.BUAV):
            print('b 撞地')
            
        # --- [Fix] ELO 更新逻辑 ---
        actual_score = 0.5 # Default draw
        if env.win: actual_score = 1.0
        elif env.lose: actual_score = 0.0
        
        # 更新主智能体ELO (临时变量)
        prev_main_elo = main_agent_elo
        adv_elo = elo_ratings[selected_opponent_name]
        
        main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score)
        # 更新对手ELO (写入字典)
        new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score)
        elo_ratings[selected_opponent_name] = new_adv_elo
        
        print(f"  Result: Score={actual_score}, Main ELO: {prev_main_elo:.0f}->{main_agent_elo:.0f}, Adv ELO: {adv_elo:.0f}->{new_adv_elo:.0f}")
        
        # 有没有试图发射过导弹
        logger.add("special/0 发射的导弹数量", m_fired, total_steps)
        # 每一场胜负变化
        logger.add("train/1 episode_return", episode_return, total_steps)
        logger.add("train/2 win", env.win, total_steps)
        logger.add("train/2 lose", env.lose, total_steps)
        logger.add("train/2 draw", env.draw, total_steps)
        logger.add("train/11 episode/step", i_episode, total_steps)
        
        # --- RL Update ---
        if len(transition_dict['dones'])>=transition_dict_threshold: 
            student_agent.update(transition_dict, adv_normed=1, mini_batch_size=128)  # 优势归一化 debug
            decide_steps_after_update = 0

            # [Modification] 保留原有梯度监控代码
            actor_grad_norm = student_agent.actor_grad
            actor_pre_clip_grad = student_agent.pre_clip_actor_grad
            critic_grad_norm = student_agent.critic_grad
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
            
            # 修改：重置 transition_dict 时保留 obs 键
            transition_dict = {'obs': [], 'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
            
            
        return_list.append(episode_return)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

        print(f"Episode {i_episode}, Progress: {total_steps/max_steps:.3f}, Curr Return: {episode_return}")
        print()

        
        
        # --- 保存模型与 ELO 维护 ---
        if i_episode % 10 == 0:
            torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
            
            actor_name = f"actor_rein{i_episode}.pt" # 物理文件名
            actor_key = f"actor_rein{i_episode}"     # ELO 字典 Key
            
            # 保存权重
            torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, actor_name))
            print(f"Saved critic.pt and {actor_name} at episode {i_episode}")

            # 将当前主智能体作为一个新的历史快照加入 ELO 列表
            # 新快照继承当前主智能体的 ELO
            elo_ratings[actor_key] = main_agent_elo
            
            # 保存 ELO JSON
            try:
                tmp_path = elo_json_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    # 同时保存 current main agent elo 以便断点续传
                    save_data = copy.deepcopy(elo_ratings)
                    save_data["__CURRENT_MAIN__"] = main_agent_elo 
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, elo_json_path)
            except Exception as e:
                print(f"Warning: failed to save elo json: {e}")

            # --- [Fix] Logging ELOs (Sorted) ---
            # 过滤掉特殊 Key
            valid_elos = {k: v for k, v in elo_ratings.items() if not k.startswith("__")}
            if valid_elos:
                mean_elo = np.mean(list(valid_elos.values()))
                # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                # 这里为了简单，直接遍历
                sorted_keys = sorted(valid_elos.keys())
                
                # 记录主智能体
                logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps)
                logger.add("Elo/Main_Agent_Centered", main_agent_elo - mean_elo, total_steps)

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

        
    # End Training
    training_end_time = time.time()
    env.end_render()
    print("Total Steps Reached. Training Finished.")
    logger.close()