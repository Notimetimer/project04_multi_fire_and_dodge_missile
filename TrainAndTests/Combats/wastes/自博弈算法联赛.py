import argparse
import time
import sys
import os
sys.path.append(project_root)
from TrainAndTests.Combats.ChoosingStrategyTrain_Big_dt_load_agents import *
import re
from Envs.Tasks.ChooseStrategyEnv_load_agents import *

use_tacview = 1 # 1  # 是否可视化

# # 找出日期最大的目录
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
# # pre_log_dir = os.path.join("./logs")


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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre_log_dir = os.path.join(project_root, "logs/combat")

# log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

# print("log目录", log_dir)


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
mission_name = 'Combat'


env = ChooseStrategyEnv(args, tacview_show=use_tacview)

r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

if __name__=="__main__":

    # Define the action cycle multiplier
    
    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    # 实例化一个红方策略网络
    red_actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

    # 实例化一个蓝方策略网络
    blue_actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

    # 加载红方actor
    log_dir1 = os.path.join(pre_log_dir, "Combat-run-20251112-121232") # "CombatFSP-run-20251117-192217")
    rein_list = glob.glob(os.path.join(log_dir1, "actor_rein*.pt"))
    latest_actor_path = latest_actor_by_index(rein_list)
    if not latest_actor_path or not os.path.isfile(latest_actor_path):
        raise FileNotFoundError(f"No actor file found in {log_dir1}. matched files: {rein_list}")
    sd = th.load(latest_actor_path, map_location=device, weights_only=True)  # sd may be a state_dict
    # 仅加载两者共有且形状匹配的参数键，避免因结构差异（如 prelu）导致报错
    model_sd = red_actor.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    if len(filtered) == 0:
        print(f"[加载警告] 从 {latest_actor_path} 未找到匹配的参数键，保持 red_actor 随机初始化")
    else:
        model_sd.update(filtered)
        red_actor.load_state_dict(model_sd)
        print(f"[加载] red_actor 从 {latest_actor_path} 加载 {len(filtered)} 个匹配参数")
    red_actor.eval() # 设置为评估模式

    # 加载蓝方actor MAPPO似乎用了prelustate，无法加载
    log_dir2 = os.path.join(pre_log_dir, "CombatMAPPO-run-20251120-113422") # "CombatFSP-run-20251114-110706")
    # rein_list = sorted(glob.glob(os.path.join(log_dir2, "actor_rein*.pt")))
    # latest_actor_path = rein_list[-1]
    rein_list = glob.glob(os.path.join(log_dir2, "actor_rein*.pt"))
    latest_actor_path = latest_actor_by_index(rein_list)
    if not latest_actor_path or not os.path.isfile(latest_actor_path):
        raise FileNotFoundError(f"No actor file found in {log_dir2}. matched files: {rein_list}")
    sd = th.load(latest_actor_path, map_location=device, weights_only=True)
    # 同样采用匹配键加载，避免 PReLU 等差异引发错误
    model_sd = blue_actor.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    if len(filtered) == 0:
        print(f"[加载警告] 从 {latest_actor_path} 未找到匹配的参数键，保持 blue_actor 随机初始化")
    else:
        model_sd.update(filtered)
        blue_actor.load_state_dict(model_sd)
        print(f"[加载] blue_actor 从 {latest_actor_path} 加载 {len(filtered)} 个匹配参数")
    blue_actor.eval() # 设置为评估模式

    steps_count = 0

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=100e3,  # 8 * 60, 
                        help="")
    args = parser.parse_args()

    decide_steps_after_update = 0
    try:
        r_action_list = []
        b_action_list = []
        
        # 测试算法效果
        for i_episode in range(1):

            test_run = 1

            # # 红方算法加载
            # # 查找所有已保存的 actor 模型
            # actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
            # if not actor_files:
            #     # 如果没有历史模型，红方使用固定策略（例如总是进攻）
            #     red_actor_loaded = False
            # else:
            #     # 随机选择一个历史模型
            #     opponent_path = random.choice(actor_files)
            #     try:
            #         red_actor.load_state_dict(torch.load(opponent_path, map_location=device))
            #         red_actor.eval() # 设置为评估模式
            #         red_actor_loaded = True
            #         if i_episode % 20 == 0: # 每20回合打印一次对手信息
            #             print(f"Episode {i_episode}: Red opponent is {os.path.basename(opponent_path)}")
            #     except Exception as e:
            #         print(f"Warning: Failed to load opponent model {opponent_path}. Error: {e}")
            #         red_actor_loaded = False

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

                    # **关键点 2: 开始【新的】一个动作周期**
                    b_action_options = [
                        "attack",
                        "escape",
                        "left",
                        "right",
                    ]

                    # --- 红方决策 ---
                    r_obs_n = flatten_obs(r_check_obs, env.key_order)
                    r_obs = np.squeeze(r_obs_n)
                    r_action_label = take_action_from_policy_discrete(red_actor, r_obs, device, explore=False)

                    # --- 蓝方决策 ---
                    b_obs_n = flatten_obs(b_check_obs, env.key_order)
                    b_obs = np.squeeze(b_obs_n)
                    b_action_label = take_action_from_policy_discrete(blue_actor, r_obs, device, explore=False)

                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    current_action = b_action_label

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



