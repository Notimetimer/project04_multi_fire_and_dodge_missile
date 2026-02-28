import argparse
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from ChoosingStrategyTrain_Big_dt_load_agents_PFSP_ELO import *
import re
from Envs.Tasks.ChooseStrategyEnv_load_agents import *
import glob
import json

use_tacview = 1  # 是否可视化

# 找出日期最大的目录
def get_latest_log_dir(pre_log_dir, mission_name=None):
    # 匹配 run-YYYYMMDD-HHMMSS 目录
    # pattern = re.compile(r"run-(\d{8})-(\d{6})")
    if mission_name:
        pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    else:
        pattern = re.compile(r"run-(\d{8})-(\d{6})")
    max_dt = None
    latest_dir = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if m:
            dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
            if max_dt is None or dt_str > max_dt:
                max_dt = dt_str
                latest_dir = d
    if latest_dir:
        return os.path.join(pre_log_dir, latest_dir)
    else:
        return None
# pre_log_dir = os.path.join("./logs")


pre_log_dir = os.path.join(project_root, "logs/combat")

log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Escape-run-20251023-105633")

print("log目录", log_dir)


start_time = time.time()
launch_time_count = 0

state_dim = 35  # len(b_obs_spaces)
action_dim = 4  # 5 #######################

# 超参数
dt_maneuver = 0.2  # 0.2  # 0.2 2
action_cycle_multiplier = 30 # 30
actor_lr = 1e-4  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# max_episodes = 1000 # 1000
max_steps = 120e4  # 120e4 65e4
hidden_dim = [128, 128, 128]  # 128
gamma = 0.95  # 0.9
lmbda = 0.95  # 0.9
epochs = 10  # 10
eps = 0.2
pre_train_rate = 0  # 0.25 # 0.25
k_entropy = 0.01  # 熵系数
mission_name = 'CombatPFSP'


env = ChooseStrategyEnv(args, tacview_show=use_tacview)

r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




def creat_initial_state():
    # 飞机出生状态指定
    # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
    blue_height = np.random.uniform(4000, 12000)
    red_height = blue_height + np.random.uniform(-2000, 2500)
    red_psi = np.random.choice([-1, 1]) * pi/2 # random.uniform(-pi, pi)
    blue_psi = sub_of_radian(red_psi, -pi)
    # blue_beta = red_psi
    red_N = 0
    red_E = -np.sign(red_psi) * 40e3 # 40e3
    blue_N = red_N
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                        'psi': red_psi
                        }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- ELO Rating System ---
K_FACTOR = 32
INITIAL_ELO = 1200

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

def update_elo(player_elo, opponent_elo, score):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    new_player_elo = player_elo + K_FACTOR * (score - expected)
    return new_player_elo

def get_opponent_probabilities(elo_ratings, target_elo=None, sigma=200.0):
    """返回与 elo_ratings.keys() 顺序对应的概率数组。
    - target_elo: 要优先靠近的 ELO（传入 main_agent_elo）。
    - sigma: 高斯核标准差，越小越只选接近 target_elo 的对手。
    """
    keys = list(elo_ratings.keys())
    if len(keys) == 0:
        return np.array([])

    elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

    if target_elo is None:
        target_elo = np.mean(elos)

    # 以高斯核度量相似度（基于差的平方）
    diffs = elos - float(target_elo)
    # 数值稳定性：将 exponent 的常数项减去 max
    scores = np.exp(-0.5 * (diffs / float(sigma))**2)
    probs = scores / (scores.sum() + 1e-12)
    return probs

# 新增：将完整路径或名字缩短为 actor_rein<数字> 形式的 key（优先匹配 actor_*）
def shorten_actor_key(path_or_name):
    """将完整路径或文件名缩短为 actor_*<数字> 形式的 key（优先匹配 actor_ 开头带数字的）"""
    base = os.path.basename(path_or_name)
    name, _ = os.path.splitext(base)
    m = re.search(r'(actor_[A-Za-z]*\d+)', name)
    if m:
        return m.group(1)
    return name

if __name__=="__main__":
    
    # Define the action cycle multiplier
    
    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2) # 2,2

    # 为FSP实例化一个红方策略网络
    red_actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

    # --- ELO 初始化 ---
    main_agent_elo = INITIAL_ELO

    # 查找所有 actor 文件并用短 key 建立 path_map 与 elo_ratings（短 key -> latest path）
    actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt")) + glob.glob(os.path.join(log_dir, "actor_sup*.pt"))
    path_map = {}
    if not actor_files:
        print("Warning: No opponent actor files found for ELO rating.")
        elo_ratings = {}
    else:
        for f in actor_files:
            key = shorten_actor_key(f)
            # 对于同一 key 保留最新文件
            if key not in path_map or os.path.getmtime(f) > os.path.getmtime(path_map[key]):
                path_map[key] = f
        # 初始化短 key 的 ELO
        elo_ratings = {k: INITIAL_ELO for k in path_map.keys()}
    # --- ELO 初始化结束 ---

    if log_dir is None:
        raise ValueError("No valid log directory found. Please check the `pre_log_dir` or `mission_name`.")

    def latest_actor_by_index(paths):
        best = None
        best_idx = -1
        for p in paths:
            m = re.search(r'actor_rein.*?(\d+)\.pt$', os.path.basename(p))
            if m:
                idx = int(m.group(1))
                if idx > best_idx:
                    best_idx = idx
                    best = p
        # fallback to most-recent-modified if no numeric match
        if best is None and paths:
            best = max(paths, key=os.path.getmtime)
        return best
    rein_list = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    latest_actor_path = latest_actor_by_index(rein_list)
    if latest_actor_path:
        # 直接加载权重到现有的 agent
        sd = th.load(latest_actor_path, map_location=device)
        agent.actor.load_state_dict(sd)  # , strict=False)  # 忽略缺失的键
        print(f"Loaded actor for test from: {latest_actor_path}")


    steps_count = 0

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=300,  # 8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60, 
                        help="")
    args = parser.parse_args()

    decide_steps_after_update = 0
    try:
        r_action_list = []
        b_action_list = []
        
        # 强化学习训练
        for i_episode in range(1):

            test_run = 1

            # --- FSP核心：为红方加载一个历史策略 ---
            if not elo_ratings:
                # 如果没有历史模型，红方使用固定策略（例如总是进攻）
                red_actor_loaded = False
                opponent_path = None
                opponent_key = None
            else:
                # 根据短 key 的 ELO 分数计算概率并选择对手
                opponent_keys = list(elo_ratings.keys())
                probabilities = get_opponent_probabilities(elo_ratings, target_elo=main_agent_elo, sigma=200.0)
                opponent_key = np.random.choice(opponent_keys, p=probabilities)
                opponent_path = path_map.get(opponent_key, None)
                
                try:
                    if opponent_path is not None:
                        red_actor.load_state_dict(torch.load(opponent_path, map_location=device))
                        red_actor.eval() # 设置为评估模式
                        red_actor_loaded = True
                        if i_episode % 20 == 0: # 每20回合打印一次对手信息
                            opponent_elo = elo_ratings.get(opponent_key, "N/A")
                            print(f"Episode {i_episode}: Red opponent is {opponent_key} (ELO: {opponent_elo:.0f})")
                    else:
                        raise FileNotFoundError("No actor file found for selected opponent key.")
                except Exception as e:
                    print(f"Warning: Failed to load opponent model {opponent_path}. Error: {e}")
                    red_actor_loaded = False

            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],}

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            r_action_label=0
            b_action_label=0
            last_decision_state = None
            current_action = None
            b_reward = None

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 回合结束判断
                # print(env.running)
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                r_check_obs = env.base_obs('r')
                b_check_obs = env.base_obs('b')
                b_obs_n = flatten_obs(b_check_obs, env.key_order)
                # 在这里将观测信息压入记忆
                # env.RUAV.obs_memory = r_check_obs.copy()
                # env.BUAV.obs_memory = b_check_obs.copy()
                b_obs = np.squeeze(b_obs_n)
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                # if env.is_action_complete('b', b_action_label):
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    # if steps_of_this_eps > 0:
                    #     transition_dict['states'].append(last_decision_state)
                    #     transition_dict['actions'].append(current_action)
                    #     transition_dict['rewards'].append(b_reward)
                    #     transition_dict['next_states'].append(b_obs) # 当前状态是上个周期的 next_state
                    #     transition_dict['dones'].append(False) # 没结束，所以是 False

                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作
                    if not test_run:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=True)
                    else:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=False)
                    decide_steps_after_update += 1
                    b_action_options = [
                        "attack",
                        "escape",
                        "left",
                        "right",
                    ]
                    # print("蓝方动作", b_action_options[b_action_label]) # Renamed b_action_list to b_action_options
                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    current_action = b_action_label

                    # --- 红方决策 ---
                    if not red_actor_loaded:
                        r_action_label = 0 # 如果没有加载模型，则执行默认动作
                    else:
                        r_obs_n = flatten_obs(r_check_obs, env.key_order)
                        r_obs = np.squeeze(r_obs_n)
                        r_action_label = take_action_from_policy_discrete(red_actor, r_obs, device, explore=False)
                    

                ### 发射导弹，这部分不受10step约束
                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                # 发射导弹判决
                if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
                    launch_time_count = 0
                    if r_action_label==0:
                        # launch_missile_if_possible(env, side='r')
                        launch_missile_with_basic_rules(env, side='r')
                    if b_action_label==0:
                        # launch_missile_if_possible(env, side='b')
                        launch_missile_with_basic_rules(env, side='b')
                
                env.step(r_action_label, b_action_label) # Environment updates every dt_maneuver
                done, b_reward, b_event_reward = env.combat_terminate_and_reward('b', b_action_label)
                done = done

                # Accumulate rewards between agent decisions
                episode_return += b_reward * env.dt_maneuver

                next_b_check_obs = env.base_obs('b')
                next_b_obs = flatten_obs(next_b_check_obs, env.key_order)


                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            # --- ELO 更新 ---
            if red_actor_loaded and opponent_key in elo_ratings:
                opponent_elo = elo_ratings[opponent_key]
                if env.win: # 蓝方(主智能体)胜利
                    score_for_main = 1.0
                    score_for_opponent = 0.0
                elif env.lose: # 蓝方失败
                    score_for_main = 0.0
                    score_for_opponent = 1.0
                else: # 平局或无胜负结果
                    score_for_main = 0.5
                    score_for_opponent = 0.5
                
                new_main_elo = update_elo(main_agent_elo, opponent_elo, score_for_main)
                new_opponent_elo = update_elo(opponent_elo, main_agent_elo, score_for_opponent)
                
                main_agent_elo = new_main_elo
                elo_ratings[opponent_key] = new_opponent_elo
            # --- ELO 更新结束 ---
            
            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            
            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            r_action_list = np.array(r_action_list)
            # b_action_list is no longer appended every dt_maneuver, need to rethink if you need this for logging


        training_end_time = time.time()  # 记录结束时间
        


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送

        # 打印最终的ELO排名
        if elo_ratings:
            print("\n--- Final ELO Rankings ---")
            sorted_elos = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
            for key, elo in sorted_elos:
                print(f"ELO: {elo:.0f} - {key}")
            print(f"Main Agent Final ELO: {main_agent_elo:.0f}")
            print("------------------------\n")

            # 尝试保存短 key 的 elo json 到 log_dir
            try:
                elo_json_path = os.path.join(log_dir, "elo_ratings.json")
                tmp_path = elo_json_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, elo_json_path)
                print(f"Saved ELO ratings (short keys) to: {elo_json_path}")
            except Exception as e:
                print(f"Warning: failed to save elo_ratings.json: {e}")

        b_action_arrays = np.array(b_action_list)

        import matplotlib.pyplot as plt

        if b_action_arrays.size == 0:
            print("b_action_arrays is empty, nothing to plot.")
        else:
            x = b_action_arrays[:, 0].astype(float)
            y = b_action_arrays[:, 1].astype(float)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, y, marker='o', linestyle='-')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('b_action_label')
            ax.set_title('b_action over time')

            # 自定义 x 轴刻度：每 10s 一个刻度；若刻度能被60整除，额外在刻度下方显示整除后的结果（分钟数），
            # 否则显示该刻度除以60后的余数（秒）
            step = 10
            xmin, xmax = x.min(), x.max()
            ticks = np.arange(np.floor(xmin / step) * step, np.ceil(xmax / step) * step + 1, step)
            labels = []
            for t in ticks:
                ti = int(round(t))
                if ti % 60 == 0:
                    labels.append(f"{ti}\n{ti//60}")
                else:
                    labels.append(str(ti % 60))
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

            ax.grid(True)
            plt.tight_layout()
            plt.show()
            print()



