import argparse
import time
import sys
import os
sys.path.append(project_root)
from TrainAndTests.ChoosingStrategyTrain_Expandable_dt import *
import re

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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre_log_dir = os.path.join(project_root, "logs")

log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Escape-run-20251023-105633")

print("log目录", log_dir)


start_time = time.time()
launch_time_count = 0

state_dim = 35  # len(b_obs_spaces)
action_dim = 4  # 5 #######################

# 超参数
dt_maneuver = 0.2  # 0.2 2
action_cycle_multiplier = 30



env = ChooseStrategyEnv(args, tacview_show=use_tacview)

r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

state_dim = 35  # len(b_obs_spaces)
action_dim = 4  # 5 #######################

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
    red_E = -np.sign(red_psi) * 40e3
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

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2) # 2,2

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
        for i_episode in range(3):

            test_run = 1

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

            steps_after_decision = 0

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
                # if steps_after_decision >= action_cycle_multiplier:
                if env.is_action_complete('b', b_action_label):
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    
                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作
                    if not test_run:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=True)
                    else:
                        b_action_probs, b_action_label = agent.take_action(b_obs, explore=False)
                    decide_steps_after_update += 1
                    steps_after_decision = 0
                    b_action_options = [
                        "attack",
                        "escape",
                        "left",
                        "right",
                    ]
                    # print("蓝方动作", b_action_options[b_action_label]) # Renamed b_action_list to b_action_options
                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    current_action = b_action_label

                    r_action_label = 0 # Red's action, assuming it's fixed or from another policy
                    # r_action_list.append(r_action_label)
                    

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

                steps_after_decision += 1

                # Accumulate rewards between agent decisions
                episode_return += b_reward * env.dt_maneuver

                next_b_check_obs = env.base_obs('b')
                next_b_obs = flatten_obs(next_b_check_obs, env.key_order)


                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            # # --- 回合结束处理 ---
            # # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
            
            
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



