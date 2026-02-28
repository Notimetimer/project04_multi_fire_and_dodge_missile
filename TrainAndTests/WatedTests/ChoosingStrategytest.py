import sys
import os

sys.path.append(project_root)

from TrainAndTests.ChoosingStrategyTrain import *
import re

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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre_log_dir = os.path.join(project_root, "logs")

log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Escape-run-20251023-105633")

print("log目录", log_dir)

start_time = time.time()
launch_time_count = 0



env = ChooseStrategyEnv(args, tacview_show=use_tacview)
# env = Battle(args, tacview_show=use_tacview)

state_dim = 35  # len(b_obs_spaces)
action_dim = 4  # 5 #######################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data_dir = os.path.join(project_root, "data")
# save_path = os.path.join(data_dir, "Crankinitial_states.npy")
# initial_states = np.load(save_path)
# print("读取的数据\n", initial_states)


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
    
    transition_dict_threshold = env.args.max_episode_len//env.dt_maneuver + 1

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

    t_bias = 0

    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=180,  # 8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=170e3,  # 8 * 60,
                        help="")

    # parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
    # parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
    args = parser.parse_args()

    return_list = []
    win_list = []
    steps_count = 0

    total_steps = 0
    i_episode = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0
    steps_since_update = 0

    # 测试
    for i_episode in range(3):  # 10
        i_episode += 1
        test_run = 1 # 0

        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}

        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)

        done = False

        env.dt_maneuver = dt_maneuver
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
            
            # 动作重映射 todo

            # 机动决策
            r_action_label = 0

            if np.isnan(b_obs).any() or np.isinf(b_obs).any():
                print('b_obs', b_check_obs)
                print()

            # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
            if not test_run:
                b_action_label, probs_b = agent.take_action(b_obs, explore=True)
            else:
                b_action_label, probs_b = agent.take_action(b_obs, explore=False)
            
            # 动作裁剪 todo
            # 根据probs np裁剪，裁剪

            b_action_list = [
                "attack",
                "escape",
                "left",
                "right",
            ]
            print("蓝方动作", b_action_list[b_action_label])

            b_action_label = 0

            ### 发射导弹
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            # 发射导弹判决
            if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
                launch_time_count = 0
                launch_missile_if_possible(env, side='r')
                launch_missile_if_possible(env, side='b')

            r_action_list.append(r_action_label)
            b_action_list.append(b_action_label)

            env.step(r_action_label, b_action_label)  # 2、环境更新并反馈
            done, b_reward, b_event_reward = env.combat_terminate_and_reward('b')
            next_b_check_obs = env.base_obs('b')
            next_b_obs = flatten_obs(next_b_check_obs, env.key_order)

            done = done
            
            # state = next_state
            episode_return += b_reward * env.dt_maneuver
            steps_since_update += 1

            # 只有并行训练情况下才需要在回合没结束的时候就更新智能体，否则可以等到回合结束

            '''显示运行轨迹'''
            # 可视化
            env.render(t_bias=t_bias)
      
        
        # print(t_bias)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t
        r_action_list = np.array(r_action_list)
        b_action_list = np.array(b_action_list)

        

