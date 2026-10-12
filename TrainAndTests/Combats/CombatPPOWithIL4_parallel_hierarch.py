'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

import typing
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
import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import traceback # [新增]
import random
import collections

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
import BasicRules_Homo
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
import Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward # 奖励函数
import Algorithms.MAPPOHybrid
import Algorithms.ActorMLPHeads

import Algorithms.MLP_heads
import Visualize.tensorboard_visualize
import Algorithms.Utils
import VsBaseline_while_training_hierarch_plus

    
dt_move = 0.04

# 测试对战初始弹药（可调）
test_red_init_ammo = 6
test_blue_init_ammo = 6

# 周期性测试所对抗的规则对手编号 (仅测试 0, 1, 2, 3)
TEST_RULE_IDS = [0, 1, 2, 3]
# 蒸馏/引导用的教师规则列表（排除 Rule 0，避免其在上一枚导弹中制导时仍开火的激进行为）
TEACHER_RULE_IDS = [1, 2, 3, 4, 5, 6]


def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()

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


# calculate_expected_score 概率计算

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None,
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta',
                               compete_old_rate=0.5, deltaFSP_epsilon=0.5,):
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
    
    if not keys: return np.array([]), []

    #  第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护)
    # 只要 compete_old_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < compete_old_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    #  第二层判断：进入核心采样逻辑
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = min(np.max(elos), float(target_elo) + 300)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
            # # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            # actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
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

    # 4. 处理 SP (最新历史版本)
    elif SP_type == 'SP':
        # rein_keys = [k for k in keys if k.startswith('actor_rein') and '_step_' not in k]
        # 严格匹配 actor_rein + 数字
        rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            # try: return int(k.replace('actor_rein', ''))
            # except: return -1
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
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                8000.0 * (1-int(randomized))
    if np.random.uniform(0,1) < 0.2 and int(randomized): # 以很低概率在更大的范围随机初始高度
        blue_height = np.random.uniform(2000.0, 12000.0)
    red_height = blue_height
    # 初始航向随机化
    red_psi = BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.sub_of_radian(-np.pi/2 + np.random.uniform(-BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.pi*2/3, BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.pi*2/3)) # -pi/3, pi/3
    blue_psi = BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.sub_of_radian(np.pi/2 + np.random.uniform(-BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.pi*2/3, BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.pi*2/3))
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


def worker_process(rank, pipe, args, state_dim, hidden_dim, action_dims_dict, device_worker, dt_maneuver, seed, opp_greedy_rate, dt_move=0.05, no_crash=1, pomdp=1, vertices=None, mask_cfg=None):
    import torch
    import numpy as np
    from Algorithms.ActorMLPHeads import ActorMLPWrapper
    from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv

    env = ChooseStrategyEnv(args, tacview_show=False)
    
    blue_agent = Algorithms.ActorMLPHeads.ActorMLPWrapper().to(device_worker)
    red_agent = Algorithms.ActorMLPHeads.ActorMLPWrapper().to(device_worker)

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RUN_EPISODE':
                blue_agent.load_state_dict(packet[0])
                opp_name, opp_type, opp_data, opp_temperature = packet[1]
                if opp_type == 'nn' and opp_data != 0:
                    try:
                        red_agent.load_state_dict(opp_data)
                    except:
                        pass
                
                env.reset(red_init_num=2, blue_init_num=2)
                
                blue_trans = {'states': [], 'obs': [], 'actions': [], 'rewards': [], 'dones': [],  'active_masks': []}
                
                ep_return = 0
                event_return = 0

                for count in range(args.max_episode_len):
                    blue_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'b' and not uav.dead]
                    red_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'r' and not uav.dead]
                    
                    if not blue_ids and not red_ids:
                        break
                        
                    id_to_idx = {uav.id: i for i, uav in enumerate(env.UAVs)}
                    flight_cmds = [None] * len(env.UAVs)
                    
                    # Forward for blue
                    if blue_ids:
                        blue_obs_all = [env.get_obs(id) for id in blue_ids]
                        blue_agent_in = [o[0] for o in blue_obs_all]
                        blue_mask_in = [o[1] for o in blue_obs_all]
                        blue_rule_in = [o[2] for o in blue_obs_all]
                        
                        batch_blue_agent = {
                            'fixed': torch.tensor(np.array([inp['fixed'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                            'enm': torch.tensor(np.array([inp['enm'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                            'ally': torch.tensor(np.array([inp['ally'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker)
                        }
                        env_actions, raw_actions, _ = blue_agent.get_action(batch_blue_agent, blue_rule_in, blue_mask_in, explore=True)
                        
                        for i, id in enumerate(blue_ids):
                            act = env_actions[i]
                            env.radar_action(id, act['radar_action'])
                            if act['fire_action']['do']:
                                env.fire_120(id, act['fire_action']['at'])
                            flight_cmds[id_to_idx[id]] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], blue_rule_in[i])
                            
                            # Store transition, 保存过程中顺序会改变，顺序更换为
                            # TODO 蓝方已经有了，把红方也加上
                            obs_tuple = (blue_agent_in[i], blue_rule_in[i], blue_mask_in[i])
                            blue_trans['obs'].append(obs_tuple)
                            blue_trans['actions'].append({k: v[i].detach().cpu().numpy() for k,v in raw_actions.items()})
                            
                            # FIXME CRITIC 不能这么获取数据 Get flat state for ValueNet
                            flat_state, _, _ = env.get_obs(id, reward_fn=1)
                            state_t = torch.tensor([flat_state['fixed']]).to(device_worker)
                            flat_s = blue_agent.flatten_agent_inputs({'fixed': state_t, 'enm': torch.zeros((1, 2, 10)).to(device_worker), 'ally': torch.zeros((1, 1, 9)).to(device_worker)})
                            blue_trans['states'].append(flat_s[0].cpu().numpy())
                    
                    # Forward for red
                    if red_ids:
                        red_obs_all = [env.get_obs(id) for id in red_ids]
                        red_agent_in = [o[0] for o in red_obs_all]
                        red_mask_in = [o[1] for o in red_obs_all]
                        red_rule_in = [o[2] for o in red_obs_all]
                        
                        batch_red_agent = {
                            'fixed': torch.tensor(np.array([inp['fixed'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                            'enm': torch.tensor(np.array([inp['enm'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                            'ally': torch.tensor(np.array([inp['ally'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker)
                        }
                        env_actions, _, _ = red_agent.get_action(batch_red_agent, red_rule_in, red_mask_in, explore=True)
                        
                        for i, id in enumerate(red_ids):
                            act = env_actions[i]
                            env.radar_action(id, act['radar_action'])
                            if act['fire_action']['do']:
                                env.fire_120(id, act['fire_action']['at'])
                            flight_cmds[id_to_idx[id]] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], red_rule_in[i])

                    env.step(flight_cmds)

                    # Get rewards for blue
                    if blue_ids:
                        for i, id in enumerate(blue_ids):
                            is_done, reward_dict = env.combat_terminate_and_reward(id)
                            rew = sum(reward_dict.values())
                            blue_trans['rewards'].append(rew)
                            blue_trans['dones'].append(is_done)
                            ep_return += rew

                pipe.send({
                    'cmd': 'EPISODE_DONE',
                    'blue_trans': blue_trans,
                    'ep_return': ep_return,
                    'event_return': event_return,
                    'wins': 1 if len(blue_ids) > 0 and len(red_ids) == 0 else 0,
                    'loss_cnt': 1 if len(blue_ids) == 0 and len(red_ids) > 0 else 0,
                    'draw_cnt': 1 if (len(blue_ids) > 0 and len(red_ids) > 0) or (len(blue_ids)==0 and len(red_ids)==0) else 0,
                    'bvr_perish_together_cnt': 0
                })
            elif cmd == 'EXIT':
                break
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            pipe.send({'error': tb})
            break


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
    max_steps=4 * 165e4,
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
    use_sil = True,
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
    save_interval = 1, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
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
    
    # 2. 参数与环境配置 (Master 用于获取维度)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.ChooseStrategyEnv(args)
    state_dim = getattr(dummy_env, 'obs_dim', 57)
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}

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
            mask_cfg = Algorithms.MAPPOHybrid.infer_mask_cfg_from_actor_meta(actor_meta_path)

    critic_net = BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.Algorithms.MLP_heads.ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = Algorithms.ActorMLPHeads.ActorMLPWrapper().to(device)
    student_agent = Algorithms.MAPPOHybrid.MAPPOHybrid(
        actor=actor_wrapper, 
        critic=critic_net, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device, 
        beta_entropy=0.01, 
        max_std=label_smoothing
    )

    # [--- 1.训练初始化
    # ADistill 用：cat 熵系数的基准值（在 agent 类属性上动态缩放）
    base_k_cat = student_agent.k_entropy.get('cat', 0.008)
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    if resume_dir is not None and os.path.exists(resume_dir):
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
        IL_epoches = 0  # 中断续训跳过预训练
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
    
    #  仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    fire_inside_weight = None
    fire_reward_weight = None

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
                
            except Exception as e:
                print(f"Failed to load optimizers: {e}")
        
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
    
    # 导出 Actor 的底层网络（PolicyNetHybrid）
    
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
    
    # 导出 Critic 的底层网络（ValueNet）
    
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
    logger = BasicRules_Homo.Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward.Visualize.tensorboard_visualize.TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    # 存储在线训练前的网络参数
    int_agent_name = "actor_rein0"
    torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{int_agent_name}.pt"))
    if init_model_requested:
        torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, "current_actor.pt"))
        torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))

    
    # 强化学习 (Self-Play / PFSP) 阶段
    
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    # 并行环境初始化 (Worker Setup)
    # 进程通信设置
    mp.set_start_method('spawn', force=True)
    
    #  启动并行测试进程池 (Async Test Pool)
    # 这个池子用于 periodic testing，不参与训练数据的生成
    test_pool = mp.Pool(processes=3, maxtasksperchild=10) 
    

    # ] 1.训练初始化
    # [--- 2.交互采样
    # 启动并行训练 Worker (Sync Training Workers)
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
    total_steps = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    batch_idx = elo_ratings.get("__LAST_UPDATE_BATCH__", 0)
    last_il_update_batch_idx = batch_idx
    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
        student_agent.reset_optimizer() # 恢复训练清除动量
    else:
        actor_freeze_until = -1
    trigger = trigger0 + (total_steps // trigger_delta) * trigger_delta
    
    current_max_steps = int(max_steps)
    
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

    
    # 主循环 (Master Process)
    
    while True:
        while total_steps < current_max_steps:
            fire_mask = 0
            #  【修改】同步并行测试阶段
            # 只有测试跑完并处理完名人堂，才进入下一步的采样和仿真
            #  1. 并行测试触发逻辑 (Async)
            if total_steps >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
                # 1. 深度拷贝当前 Actor 权重到 CPU 内存
                current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                test_tasks = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        VsBaseline_while_training_hierarch_plus.test_worker, 
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
                            'red_init_ammo': test_red_init_ammo,
                            'blue_init_ammo': test_blue_init_ammo,
                        }
                    )
                    test_tasks.append(obj)
                
                # 第二种形式：追加额外测试 (机动动作确定化 + 动作次序限制打开)
                test_tasks_no_random = []
                for r_idx in TEST_RULE_IDS:
                    obj = test_pool.apply_async(
                        VsBaseline_while_training_hierarch_plus.test_worker, 
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
                            'red_init_ammo': test_red_init_ammo,
                            'blue_init_ammo': test_blue_init_ammo,
                        }
                    )
                    test_tasks_no_random.append(obj)

                # 等待所有测试进程结束
                test_results = [t.get() for t in test_tasks]
                test_results_no_random = [t.get() for t in test_tasks_no_random]

                # 记录第一种测试结果
                outcomes = {rule_num: score for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_return = {rule_num: result2 for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
                outcomes_perish = {rule_num: p_t_ for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps)
                    # logger.add(f"test/agent_vs_rule{r_num}_return", outcomes_return[r_num], total_steps)
                    print(f"  [Test Result] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

                # 记录第一种测试的平均指标（所有对手的平均Score和超视距双杀率）
                avg_score = np.mean(list(outcomes.values()))
                avg_perish_together = np.mean(list(outcomes_perish.values()))
                logger.add("test/avg_score", avg_score, total_steps)
                logger.add("test/BVR perish together", avg_perish_together, total_steps)

                # 记录第二种测试结果 (test_No_random)
                outcomes_nr = {rule_num: score for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_return_nr = {rule_num: result2 for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}
                outcomes_perish_nr = {rule_num: p_t_ for rule_num, score, result2, wins, loses, draws, p_t_ in test_results_no_random}

                for r_num, score in outcomes_nr.items():
                    logger.add(f"test_No_random/agent_vs_rule{r_num}", score, total_steps)
                    # logger.add(f"test_No_random/agent_vs_rule{r_num}_return", outcomes_return_nr[r_num], total_steps)
                    print(f"  [Test No Random] Rule_{r_num}: {score} (return: {outcomes_return_nr[r_num]:.2f})")

                # 记录第二种测试的平均指标（所有对手的平均Score和超视距双杀率）
                avg_score_nr = np.mean(list(outcomes_nr.values()))
                avg_perish_together_nr = np.mean(list(outcomes_perish_nr.values()))
                logger.add("test_No_random/avg_score", avg_score_nr, total_steps)
                logger.add("test_No_random/BVR perish together", avg_perish_together_nr, total_steps)

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

            #  准备训练 Batch (Synchronous)
            # 改变环境奖励权重，超过100轮采样再更新权重，每次权重维持5轮采样
            if adj_r_w and batch_idx > 10:
                fire_inside_weight = None
                fire_reward_weight = None

            if not adj_r_w:
                fire_inside_weight = None
                fire_reward_weight = None

            # 获取当前策略权重 (CPU)
            current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
            
            # 分发任务给 Worker
            # 这一步 Master 决定每个 Worker 打谁
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            # 处理纯自博弈逻辑：当没有初始规则对手时，筛选分数最高的 MAX_HISTORY_SIZE 个对手作为匹配池
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
                    'end_reward_weight': 0.3 + np.clip(total_steps/18e3, 0, 1)*(0.4-0.3), # 1.0
                    'fire_inside_weight': fire_inside_weight,
                    'fire_reward_weight': fire_reward_weight,
                }
                
                # 发送指令 pipe.send
                pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # 等待所有 Worker 完成 (Barrier)
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
            
            #  数据聚合与处理
            batch_total_steps = 0
            batch_wins = 0
            batch_loss_cnt = 0
            batch_draw_cnt = 0        # 新增统计
            batch_bvr_perish_together_cnt = 0 # 新增统计
            batch_total_return = 0    # 新增统计
            batch_total_event_return = 0
            batch_total_m_fired = 0   # 新增统计
            batch_fire_supervised = 0 # [新增] 本 batch 收集到的已结算开火记录数
                        
            for res in batch_results:
                # res 结构: {'trans':..., 'ego_tr':..., 'enm_tr':..., 'metrics':..., 'opp_name':...}
                l_tr = res['trans'] # PPO 训练数据 (含探索)
                ego_tr = res['ego_trans'] # SIL 蓝方数据
                enm_tr = res['enm_trans'] # SIL 红方数据
                metrics = res['metrics']
                opp_name = res['opp_name']
                
                # 更新 EMA [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                alpha_ema = 0.1
                if opp_name not in Elite_Fire_Stats:
                    Elite_Fire_Stats[opp_name] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    
                old_stats = Elite_Fire_Stats[opp_name]
                new_stats = list(old_stats)
                    
                Elite_Fire_Stats[opp_name] = new_stats
                
                
                # 使用当前主代理的key来记录本方行为
                main_agent_key = "__CURRENT_MAIN__"
                if main_agent_key not in Elite_Fire_Stats:
                    Elite_Fire_Stats[main_agent_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                old_main_stats = Elite_Fire_Stats[main_agent_key]
                new_main_stats = list(old_main_stats)
                    
                Elite_Fire_Stats[main_agent_key] = new_main_stats
                
                # [新增] 填充 buffer 用户打印详情
                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")
                
                batch_total_steps += metrics['steps']
                batch_total_return += metrics['return'] # 已是固定±100终局的归一化回报
                batch_total_event_return += metrics['event_return']
                batch_total_m_fired += metrics['m_fired']

                # [新增] 收集本回合已结算的开火命中记录，直接拼接入滚动缓冲
                fire_records = [r for r in res.get('fire_hit_records', []) if r.get('hit_record') is not None]
                fire_records_buffer.extend(fire_records)
                batch_fire_supervised += len(fire_records)
                if metrics.get('BVR_perish_together', False):
                    batch_bvr_perish_together_cnt += 1
                

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
                
                elo_ratings[opp_name] = new_adv_elo
                # 同步更新 Elite 池中已有的对手Elo分值
                if opp_name in elite_elo_ratings:
                    elite_elo_ratings[opp_name] = new_adv_elo
                # 同步更新 Hall of Fame 中已有的对手Elo分值
                if opp_name in hall_of_fame:
                    hall_of_fame[opp_name] = new_adv_elo
            
        
            # [新增] 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps += batch_total_steps
            batch_idx += 1
            
            #  3.5 记录批次聚合指标
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
            logger.add("train_plus/batch_score", batch_score, total_steps)
            logger.add("train_plus/filtered_score", filtered_score, total_steps)
            logger.add("train_plus/target_p1", target_p1, total_steps)
            
            # 记录平均回报与胜率
            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps)
            logger.add("train_plus/Avg event return", batch_total_event_return / num_workers, total_steps)
            logger.add("train/2 win", batch_wins / num_workers, total_steps)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, total_steps)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, total_steps)
            # 找最好的智能体
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps)
            logger.add("agent/ batch_step", batch_idx, total_steps)

            #  更新，保存与维护 (Checkpoint & Pool)
            if batch_idx % save_interval == 0 and \
                len(transition_dict['dones']) >= transition_dict_threshold:
                # #  执行训练 (PPO Update)
                
                '记录ELo相对位置'
                # [新增] 调节alpha_il
                #  [新增] 动态计算 alpha_il
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
                logger.add("train_plus/elo_diff_x", x_elo_diff, total_steps)
                
                # 学习率warm_up
                actor_lr = min(actor_lr0, actor_lr0 * total_steps/1e6)
                critic_lr = min(critic_lr0, critic_lr0 * total_steps/1e6)
                student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

                if batch_idx <= actor_freeze_until:
                    freeze_actor = 1
                else:
                    freeze_actor = 0
                # # critic先收敛
                # if total_steps < 5e3:
                #     freeze_actor = 1
                
                max_fire_logits = 4.0

                alpha_distill = 0
                k_entropy_cat_scale = 1.0

                student_agent.k_entropy['cat'] = base_k_cat * k_entropy_cat_scale

                student_agent.update(transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed, target_p1=target_p1,
                                     k_nonlinear=k_nonlinear, mask_on=fire_mask, actor_frozen=freeze_actor, bern_max_logits=max_fire_logits,
                                     alpha_distill=alpha_distill, teacher_actor=None,
                                     AFiltered=AFiltered, conf_thres=conf_thres, bern_included=bern_included,
                                     ppo_with_bern=ppo_with_bern)

                # 开火命中监督更新：从滚动缓冲采样，仅更新 actor.net.fc_bern，不影响机动策略
                if use_supervised_fire and len(fire_records_buffer) > 100: #  and\
                    # (student_agent.max_fire_prob / student_agent.min_fire_prob <= sup_on_prob_ratio  or  ppo_with_bern==0): # PPO不更新，就全程带上，否则只作为开火分布保护器

                    fire_sl_batch = random.sample(list(fire_records_buffer), min(int(supervised_fire_batch_size), len(fire_records_buffer)))
                    student_agent.supervised_fire_update(fire_sl_batch, epochs=1, batch_size=int(supervised_fire_batch_size), fire_lr_ratio=1)
                    logger.add("train_plus/supervised_fire_buffer_size", len(fire_records_buffer), total_steps)
                    logger.add("train_plus/supervised_fire_samples", batch_fire_supervised, total_steps)
                    logger.add("train/7b supervised_fire_loss", getattr(student_agent, 'last_supervised_fire_loss', 0.0), total_steps)


                # # 开火概率保护，如果策略向满开火/不开一发坍缩，直接用有监督暴力修正开火概率
                # if batch_idx % 10 == 0:
                #     student_agent.fire_prob_protection(transition_dict, protect_epochs=4)
                
                # 计算/更新 PPO actor pre-clip 梯度的 EMA 值
                current_ppo_grad = student_agent.pre_clip_actor_grad
                if current_ppo_grad is not None and not np.isnan(current_ppo_grad):
                    if ppo_grad_ema is None:
                        ppo_grad_ema = current_ppo_grad
                    else:
                        ppo_grad_ema = 0.95 * ppo_grad_ema + 0.05 * current_ppo_grad

                # 记录 Log

                # [Modification] 保留原有梯度监控代码
                actor_pre_clip_grad = student_agent.pre_clip_actor_grad
                critic_pre_clip_grad = student_agent.pre_clip_critic_grad

                # 梯度监控
                # logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
                # logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
                # 损失函数监控
                logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
                logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
                # 强化学习actor特殊项监控
                logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
                logger.add("train/9 entropy_cat", student_agent.entropy_cat, total_steps)
                logger.add("train/9 entropy_bern", student_agent.entropy_bern, total_steps)
                logger.add("train_plus/max_fire_prob", student_agent.max_fire_prob, total_steps)
                logger.add("train_plus/min_fire_prob", student_agent.min_fire_prob, total_steps)

                # logger.add("train/10 advantage", student_agent.advantage, total_steps) 
                # 强化学习
                # logger.add("train/10 explained_var", student_agent.explained_var, total_steps)
                # logger.add("train/10 approx_kl", student_agent.approx_kl, total_steps)
                # logger.add("train/10 clip_frac", student_agent.clip_frac, total_steps)
                
                # [新增] 诊断监控
                logger.add("train_plus/td_error_var", student_agent.td_error_var, total_steps)

                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

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
                if total_steps >= WARM_UP_STEPS:
                    # 复制当前主代理的蓝方行为统计到该历史版本，记录它"作为蓝方时"的行为特征
                    if "__CURRENT_MAIN__" in Elite_Fire_Stats:
                        Elite_Fire_Stats[actor_key] = copy.deepcopy(Elite_Fire_Stats["__CURRENT_MAIN__"])
                    else:
                        Elite_Fire_Stats[actor_key] = [0.0, 0.0, 0.0, 0.0, 0.0]  # [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                    elo_ratings[opp_name] = main_agent_elo

                    
                
                # 逻辑分支 B: 维护“全量历史记录” (Full JSON)
                
                # 目标：记录所有产生过的 Agent 的最后一次已知 Elo，无论它是否在精英池里
                # 无论是否进入精英池，全量表都要记录
                elo_ratings[actor_key] = main_agent_elo
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                
                
                # 逻辑分支 B: 定期重刷精英池 (每 1M step)
                
                if total_steps > 0 and total_steps % 1e6 == 0:
                    print(f"\n>>> [Elite Pool Refresh] Refreshing elite_elo_ratings at {total_steps} steps...")
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

                
                # 逻辑分支 C: 保存“精英池快照” (Elite JSON)
                
                # 这才是下次训练 resume 时应该读取的文件
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                
                
                # 逻辑分支 D: 保存名人堂 (hall_of_fame.json)
                
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                
                # 逻辑分支 E: 保存胜率表 (用于核心更新与选择)
                
                with open(Elite_Fire_Stats_path, "w", encoding="utf-8") as f:
                    json.dump(Elite_Fire_Stats, f, ensure_ascii=False, indent=2)

                #  日志记录 (Logging) - 保持不变，展示的是精英池状态
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps)
                    
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
                    logger.add("Elo_Centered/EloDiffToMean", elo_diff_to_thres, total_steps)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps)

                    curr_rank = 0.5 if elo_spread == 0 else (main_agent_elo - min_elo) / elo_spread

                    # Elo分数没有稳定不许记录
                    if total_steps >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rank_normed %", curr_rank * 100, total_steps)

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
                        
                        #  正式入池
                        if hist_agent_as_opponent:
                            elite_elo_ratings[actor_key] = main_agent_elo
                            print(f"Accepted {actor_key} into Elite Pool.")

                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps)

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, total_steps)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, total_steps)

                #  例行保存优化器状态和步数
                torch.save({
                    'actor_optimizer': student_agent.actor_optimizer.state_dict(),
                    'critic_optimizer': student_agent.critic_optimizer.state_dict(),
                }, os.path.join(log_dir, "optimizers_state.pt"))

                # 保存开火监督数据 buffer
                if use_supervised_fire:
                    fire_buffer_path = os.path.join(log_dir, "fire_records_buffer.pkl")
                    with open(fire_buffer_path, "wb") as f:
                        pickle.dump({"records": list(fire_records_buffer), "maxlen": fire_records_buffer.maxlen}, f)
                    print(f"Saved fire records buffer to {fire_buffer_path}. Size: {len(fire_records_buffer)}")
                    
                # print(f"Optimizers routinely saved to optimizers_state.pt")
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

        #  [新增] 达到 max_steps 后的交互逻辑
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        # 【新增】明确告知用户 Worker 当前状态
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