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
from BasicRules import *
from Envs.Tasks.ChooseStrategyEnv2_0 import * # 1217-205333前
# from Envs.Tasks.ChooseStrategyEnv2_1 import *  # 1224
from Algorithms.HybridBuffer_rnn_expandable import HybridReplayBuffer
from Algorithms.PPOHybrid23_2 import PPOHybrid, ValueNet, PolicyNetHybrid, HybridActorWrapper
from Visualize.tensorboard_visualize import TensorBoardLogger


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
            

# # 加载数据
# il_transition_dict, transition_dict = load_il_and_transitions(
#     os.path.join(cur_dir, "IL"),
#     "il_transitions_combat.pkl",
#     "transition_dict_combat.pkl"
# )

# # --- 关键步骤：执行数据重构 ---
# if il_transition_dict is not None:
#     # 这里完成 (Batch, Key) -> (Key, Batch) 的转换
#     il_transition_dict['actions'] = restructure_actions(il_transition_dict['actions'])
    
#     # 顺便确保 states 和 returns 也是标准的 float32 numpy array
#     il_transition_dict['states'] = np.array(il_transition_dict['states'], dtype=np.float32)
#     il_transition_dict['returns'] = np.array(il_transition_dict['returns'], dtype=np.float32)

# ------------------------------------------------------------------
# 参数与环境配置
# ------------------------------------------------------------------
parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=10*60, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=55e3, help="")
args = parser.parse_args()

# 超参数
actor_lr = 1e-3 # 4 1e-3
critic_lr = actor_lr * 5 # * 5
# IL_epoches= 0  # 80 检查一下，这个模仿学习可能有问题!!!
max_steps = 165e4
hidden_dim = [128, 128, 128]
gamma = 0.995
lmbda = 0.995
epochs = 4  # 可能需要更小一些 10 4
eps = 0.2
k_entropy = 0.05 # 1 # 
seqlen = 1 # GRU 序列长度 3 最小是1了，再小buffer就不可用了
n_envs = 1
use_attention = 0 # 是否使用通道注意力 1

env = ChooseStrategyEnv(args)
state_dim = env.obs_dim
# --- 修改：定义隐藏状态维度并初始化 ReplayBuffer ---
a_h_dim = (1, 128) # (num_layers, hidden_size)
c_h_dim = (1, 128)

# 动作空间定义 (需要与 BasicRules 产生的数据对应)
# cat: 离散机动动作头 (env.fly_act_dim 通常是一个列表 [n_actions])
# bern: 攻击动作头 (env.fire_dim 通常是 1)
action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
# 实例化新的 Buffer，取代原有的 transition_dict 列表

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
action_bound = None

# 1. 创建神经网络
# critic 不使用GRU
actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, 
                            reduction_ratio=16, num_layers=a_h_dim[0],
                            use_channel_attention=use_attention, use_gru=0).to(device)
critic_net = ValueNet(state_dim, hidden_dim, reduction_ratio=16, 
                      num_layers=c_h_dim[0],
                      use_channel_attention=use_attention, use_gru=0).to(device)

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
    
    # if il_transition_dict is None:
    #     print("No il_transitions_combat file found.")
    #     sys.exit(1)

    # summarize(il_transition_dict)

    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    mission_name = '打莽夫_左右_noCA_4epochs'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    # print("Start MARWIL Training...")

    # # 训练循环
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

    # ==============================================================================
    # 强化学习 (Self-Play / PFSP) 阶段
    # ==============================================================================
    
    # # 根据参数数量缩放学习率
    # from Math_calculates.ScaleLearningRate import scale_learning_rate
    # actor_lr = scale_learning_rate(actor_lr, student_agent.actor)
    # critic_lr = scale_learning_rate(critic_lr, student_agent.critic)
    # student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    
    # tacview_input = input("Enable tacview visualization? (0=no, 1=yes) [default 0]: ").strip()
    tacview_input= 0
    
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

    action_cycle_multiplier = 40 # 8s 决策一次 40
    dt_action_cycle = dt_maneuver * action_cycle_multiplier
    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 

    # 初始化经验池
    replay_buffer = HybridReplayBuffer(
        n_envs=n_envs, 
        buffer_size=int(transition_dict_threshold), 
        obs_dim=state_dim, 
        state_dim=state_dim, 
        action_dims_dict=action_dims_dict,
        actor_hidden_dim=a_h_dim, 
        critic_hidden_dim=c_h_dim,
        use_active_masks=True,
        device=device
    )
    # --- [Modification] 移除 ELO 相关的所有逻辑 ---
    # K_FACTOR = 32
    # INITIAL_ELO = 1200
    # elo_ratings = {}
    # elo_json_path = os.path.join(log_dir, "elo_ratings.json")
    # main_agent_elo = INITIAL_ELO
    # ... (所有 ELO 函数和加载逻辑均被移除)

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
    
    
    while total_steps < int(max_steps):
        i_episode += 1
        # --- 修改：回合开始，重置 GRU 隐藏状态 ---
        # 显式初始化为全零，形状：(Layers, Batch=1, Dim)
        h_actor = torch.zeros(a_h_dim[0], 1, a_h_dim[1]).to(device)
        h_critic = torch.zeros(c_h_dim[0], 1, c_h_dim[1]).to(device)
        
        # 也要为存储变量预留位置
        last_h_actor = h_actor.detach().cpu().numpy()
        last_h_critic = h_critic.detach().cpu().numpy()
        # [Modification] 移除了 actor_key 的提前定义，只在保存时定义
        
        # --- [Modification] 对手选择逻辑 (PFSP) ---
        # 强制对手为 Rule 0
        adv_is_rule = True
        rule_num = 0
        print(f"Eps {i_episode}: Opponent is Rule {rule_num}")
        
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
        
        ego = env.BUAV
        enm = env.RUAV
        # 新增：每回合的死亡查询表（0 表示存活，1 表示已记录死亡瞬间）
        dead_dict = {'r': int(bool(enm.dead)), 'b': int(bool(ego.dead))}

        done = False

        env.dt_maneuver = dt_maneuver
        
        episode_start_time = time.time()
        
        last_r_action_label = 0
        last_b_action_label = 0
        
        steps_of_this_eps = -1
        
        m_fired = 0
        
        last_decision_time = 0.0 # 初始化
        
        # --- Episode Loop ---
        for count in range(round(args.max_episode_len / dt_maneuver)):
            current_time = env.t
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
                # 1. 计算当前时刻的 Value 和隐状态 注意：即便在采样阶段，也需要用 Critic 获取 next_value 以便存入 Buffer
                current_v_tensor, h_critic_next = student_agent.critic(
                    torch.tensor(b_state_global, dtype=torch.float).to(device), 
                    h_in=h_critic
                )
                current_v = current_v_tensor.item()
                
                # 2. 存储上一个动作周期的经验
                # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                if steps_of_this_eps > 0 and not dead_dict['b']: # 临时接替一下 active mask
                    # 修改：传入 last_decision_obs 和 last_decision_state
                    # 使用 HybridBuffer 的 add 方法
                    # 存储：(旧观测, 旧状态, 动作, 奖励, done, 当前Value, Actor旧隐状态, Critic旧隐状态)
                    replay_buffer.add(
                        obs=last_decision_obs, 
                        state=last_decision_state, 
                        action_dict=current_action, 
                        reward=b_reward, 
                        done=False, 
                        next_value=current_v, 
                        active_mask=not ego.dead,
                        actor_h=last_h_actor, 
                        critic_h=last_h_critic,
                    )
                    # 时间戳检查
                    replay_buffer.timestamps_dict['states'].append(last_decision_time)
                    replay_buffer.timestamps_dict['next_values'].append(current_time)
                    last_decision_time = current_time # 为下一周期更新起点

                # 3. 记录本次决策前的状态，作为新周期的起始记忆
                # 更新 obs 和 state 两个变量
                last_decision_obs = b_obs.copy()
                last_decision_state = b_state_global.copy()
                # 记录产生此动作前的隐藏状态，以便存入 Buffer 保持时序对齐
                last_h_actor = h_actor.detach().cpu().numpy()
                last_h_critic = h_critic.detach().cpu().numpy()
                
                # 2. student_agent 产生一个动作
                # 红方(对手决策)
                r_state_check = env.unscale_state(r_check_obs)
                if adv_is_rule:
                    # [Fix] 传入选定的 rule_num
                    r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=last_r_action_label)
                else:
                    pass
                    # # [Fix] NN 对手决策
                    # r_action_exec, r_action_raw, _, r_action_check = adv_agent.take_action(r_obs, explore=1)
                    # r_action_label = r_action_exec['cat'][0]
                    # r_fire = r_action_exec['bern'][0] # 网络控制开火
                last_r_action_label = r_action_label
                r_m_id = None
                if r_fire:
                    r_m_id = launch_missile_immediately(env, 'r')

                # --- 蓝方 (训练对象) 决策 ---
                b_state_check = env.unscale_state(b_check_obs)
                # 修改：Actor 依然使用 b_obs (局部观测) 进行决策
                b_action_exec, b_action_raw, h_actor_next, b_action_check = student_agent.take_action(
                    b_obs, h0=h_actor, explore=1
                )
                # 更新隐藏状态
                h_actor = h_actor_next
                h_critic = h_critic_next
                
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
                
                

            r_maneuver = env.maneuver14(enm, r_action_label) # 规则智能体不分左右
            b_maneuver = env.maneuver14LR(ego, b_action_label) # 强化学习智能体区分左右

            env.step(r_maneuver, b_maneuver)
            done, b_reward, b_reward_assisted = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier)
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
                with torch.no_grad():
                    # 使用最后时刻更新后的 next_b_state_global 和 h_critic
                    # 注意：h_critic 在循环最后已经更新为 h_critic_next 了，正好对应下一时刻
                    val_tensor, _ = student_agent.critic(
                        torch.tensor(next_b_state_global, dtype=torch.float).to(device),
                        h_in=h_critic 
                    )
                    current_v = val_tensor.item()
                if ego.dead:
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
            replay_buffer.add(
                        obs=last_decision_obs, 
                        state=last_decision_state, 
                        action_dict=current_action, 
                        reward=b_reward, 
                        done=True, 
                        next_value=current_v, 
                        actor_h=last_h_actor, 
                        critic_h=last_h_critic,
                    )
            episode_return += b_reward
            
            # 时间戳检查
            replay_buffer.timestamps_dict['states'].append(last_decision_time)
            replay_buffer.timestamps_dict['next_values'].append(current_time)
            last_decision_time = current_time # 为下一周期更新起点
            
        print('r 剩余导弹数量:', enm.ammo)
        print('r 发射时间:', enm.missile_launch_time)
        
        print('b 剩余导弹数量:', ego.ammo)
        print('b 发射时间:', ego.missile_launch_time)
        
        if env.crash(enm):
            print('r 撞地')
        if env.crash(ego):
            print('b 撞地')
            
        
        
        # --- [Modification] 移除 ELO 更新逻辑 ---
        # actual_score = 0.5 # Default draw
        # if env.win: actual_score = 1.0
        # elif env.lose: actual_score = 0.0
        # ... (所有 ELO 更新代码均被移除)
        
        # 有没有试图发射过导弹
        logger.add("special/0 发射的导弹数量", m_fired, total_steps)
        # 每一场胜负变化
        logger.add("train/1 episode_return", episode_return, total_steps)
        logger.add("train/2 win", env.win, total_steps)
        logger.add("train/2 lose", env.lose, total_steps)
        logger.add("train/2 draw", env.draw, total_steps)
        logger.add("train/11 episode/step", i_episode, total_steps)
        
        # --- RL Update ---
        
        # if replay_buffer.full:  # 仅在并行无扩容的情况下使用
        if replay_buffer.ptr >= transition_dict_threshold:
            # 准备序列数据
            transition_data = replay_buffer.get_recurrent_data(
                student_agent.critic, 
                seq_len=seqlen, 
                gamma=gamma, 
                lmbda=lmbda
            )
            student_agent.update(transition_data, adv_normed=1)  # 优势归一化 debug
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
            
            # 重置 Buffer
            replay_buffer.clear()
            decide_steps_after_update = 0
            
            
        return_list.append(episode_return)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

        print(f"Episode {i_episode}, Progress: {total_steps/max_steps:.3f}, Curr Return: {episode_return}")
        print()

        
        # --- [Modification] 保存模型，actor不覆盖 ---
        if i_episode % 10 == 0:
            # 1. 保存 Critic (可以覆盖)
            torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
            
            # 2. 保存带回合数的 Actor 文件，不覆盖
            actor_name = f"actor_rein{i_episode}.pt"
            torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, actor_name))
            print(f"Saved critic.pt and {actor_name} at episode {i_episode}")

        # --- [Modification] 移除 ELO 日志记录 ---

    # End Training
    training_end_time = time.time()
    env.end_render()
    print("Total Steps Reached. Training Finished.")
    logger.close()