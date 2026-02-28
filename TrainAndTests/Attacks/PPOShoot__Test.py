'''
训练进攻策略:

红方闲庭信步
蓝方追赶红方
导弹发射归智能体控制
'''

'''

'''

import argparse
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from PPOShoot__Train import *
import re

# from tqdm import tqdm

dt_maneuver = 0.2  # 0.2

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


pre_log_dir = os.path.join(project_root, "logs/shoot")

log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

# log_dir = os.path.join(pre_log_dir, "Shoot_old-run-20251114-093439")

# print("log目录", log_dir)


# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')

parser = argparse.ArgumentParser("UAV swarm confrontation")

# Environment
parser.add_argument("--max-episode-len", type=float, default=8*60,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# num_episodes = 1000 # 10000 1000
max_steps = 65e4
hidden_dim = [128, 128, 128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
pre_train_rate = 0  # 0.05 # 0.25 # 0.25
k_entropy = 0.01  # 熵系数
mission_name = 'Shoot'

env = ShootTrainEnv(args, tacview_show=use_tacview)
# env = Battle(args, tacview_show=use_tacview)
# r_obs_spaces = env.get_obs_spaces('r') # todo 子策略的训练不要用这个
# b_obs_spaces = env.get_obs_spaces('b')
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

state_dim = 1+1+1+ 8 + 7 + 1  # len(b_obs_spaces)
action_dim = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    agent = PPO_bernouli(state_dim, hidden_dim, 1, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)
    
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

    # rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
    # sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
    # latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)

    if latest_actor_path:
        # 直接加载权重到现有的 agent
        sd = th.load(latest_actor_path, map_location=device)
        agent.actor.load_state_dict(sd)  # , strict=False)  # 忽略缺失的键
        print(f"Loaded actor for test from: {latest_actor_path}")

    out_range_count = 0
    return_list = []
    win_list = []
    # steps_count = 0
    total_steps = 0
    i_episode = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    training_start_time = time.time()
    
    t_bias = 0

    try:
        # 强化学习训练
        rl_steps = 0
        return_list = []
        win_list = []
        # with tqdm(total=int(num_episodes*(1-pre_train_rate)), desc='Iteration') as pbar:  # 进度条
        # for i_episode in range(int(num_episodes*(1-pre_train_rate))):
        for i_episode in range(4):  # 10
            test_run = 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

            # 飞机出生状态指定
            red_E = 45e3 # random.uniform(30e3, 40e3)  # 20, 60 特意训练一个近的，测试一个远的
            blue_E= -red_E
            red_height = random.uniform(3e3, 12e3)
            blue_height = random.uniform(3e3, 12e3)

            DEFAULT_RED_BIRTH_STATE = {'position': np.array([0.0, red_height, red_E]),
                                       'psi': -pi/2
                                       }
            DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, blue_E]),
                                        'psi': pi/2
                                        }
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                      red_init_ammo=0, blue_init_ammo=6)
            RUAV_set_speed = random.uniform(0.7, 1.5) * 340
            RUAV_set_height = red_height
            BUAV_set_speed = random.uniform(0.7, 1.5) * 340
            
            last_launch_time = - np.inf
            random_theta_plus = 0
            random_psi_plus = 0

            # a1 = env.BUAV.pos_  # 58000,7750,20000
            # a2 = env.RUAV.pos_  # 2000,7750,20000
            # b1 = env.UAVs[0].pos_
            # b2 = env.UAVs[1].pos_
            done = False

            actor_grad_list = []
            critc_grad_list = []
            actor_loss_list = []
            critic_loss_list = []
            entropy_list = []
            ratio_list = []

            r_action_list = []
            b_action_list = []

            episode_start_time = time.time()

            # 环境运行一轮的情况
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 回合结束判断
                # print(env.running)
                current_t = count * dt_maneuver
                if env.running == False or done:  # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                r_obs_n, r_obs_check = env.attack_obs('r')
                b_obs_n, b_obs_check = env.attack_obs('b')

                # 在这里将观测信息压入记忆
                env.RUAV.obs_memory = r_obs_check.copy()
                env.BUAV.obs_memory = b_obs_check.copy()

                b_obs = np.squeeze(b_obs_n)

                cos_delta_psi = b_obs_check["target_information"][0]
                sin_delta_psi = b_obs_check["target_information"][1]
                delta_psi = atan2(sin_delta_psi, cos_delta_psi)

                distance = b_obs_check["target_information"][3]*10e3
                alpha = b_obs_check["target_information"][4]
                AA_hor = b_obs_check["target_information"][6]
                launch_interval = b_obs_check["weapon"]*120
                missile_in_mid_term = b_obs_check["missile_in_mid_term"]

                
                # 发射导弹判决
                u, _ = agent.take_action(b_obs_n, explore=0) # 0 1
                ut = u[0]
                at = ut

                # Shield
                at, _ = shoot_action_shield(at, distance, alpha, AA_hor, launch_interval)

                if at == 1:
                    last_launch_time = env.t
                    launch_missile_immediately(env, side='b')                  


                # 机动决策
                r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                           enm_pos_=env.BUAV.pos_, distance=distance,
                                           ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                           o00=o00, R_cage=env.R_cage, wander=0,
                                           set_height=RUAV_set_height, set_speed=RUAV_set_speed
                                           )

                random_theta_plus = generate_ar1_value(random_theta_plus, 0.9, 0.1)
                random_psi_plus = generate_ar1_value(random_psi_plus, 0.9, 0.1)
                

                b_action_n = np.array([env.RUAV.alt-env.BUAV.alt + 1000 * random_theta_plus, 
                                       delta_psi + pi/4 * random_psi_plus, 
                                       BUAV_set_speed])

                r_action_list.append(r_action_n)
                b_action_list.append(b_action_n)

                env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                done, b_reward, _ = env.get_terminate_and_reward('b', u)
                next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
                env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作
                total_steps += 1

                # state = next_state
                episode_return += b_reward * env.dt_maneuver

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)

            

            if env.lose:
                print("回合失败")
            if env.win:
                print("回合胜利")

            return_list.append(episode_return)
            win_list.append(1 - env.lose)

            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    
        