import sys
import os

sys.path.append(project_root)

from TrainAndTests.Wasted.PPOEscapeFromPlaneTrain import *
import re

dt_maneuver = 0.2  # 0.2
action_eps = 0.0  # 0.7  # 动作平滑度


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

###
# log_dir = os.path.join(pre_log_dir, "EscapeFromPlane-run-20251027-162521")

print("log目录", log_dir)

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

if log_dir is None:
    raise ValueError("No valid log directory found. Please check the `pre_log_dir` or `mission_name`.")

rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
if latest_actor_path:
    # 直接加载权重到现有的 agent
    sd = th.load(latest_actor_path, map_location=device)
    agent.actor.load_state_dict(sd)  # , strict=False)  # 忽略缺失的键
    print(f"Loaded actor for test from: {latest_actor_path}")


# # 测试特定训练结果
# latest_actor_path = os.path.join(log_dir, "actor_rein420.pt")
# sd = th.load(latest_actor_path, map_location=device)
# agent.actor.load_state_dict(sd)


t_bias = 0

parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=240,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=170e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

def creat_initial_state():
    # 飞机出生状态指定
    # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
    cases = initial_states.shape[0]  # 读取攻击区数据
    line_number = np.random.choice(cases)

    red_height = initial_states[line_number, 0]
    blue_height = initial_states[line_number, 1]

    red_height = 13e3
    blue_height = 12e3

    if blue_height <= 9e3:
        initial_dist = 30e3
    elif blue_height < 11e3:
        initial_dist = 35e3
    elif blue_height < 12e3:
        initial_dist = 42e3
    else:
        initial_dist = 50e3

    red_psi = np.random.choice([-1, 1]) * pi/2  # random.uniform(-pi, pi)
    blue_psi = -red_psi
    # blue_beta = red_psi
    red_N = 0  # (random.randint(0, 1)*2-1)*47e3  # (random.randint(0,1)*2-1)*57e3
    red_E = -np.sign(red_psi) * initial_dist
    blue_N = red_N
    blue_E = 0
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi
                                }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE, line_number


try:
    env = EscapeTrainEnv(args, tacview_show=1)  # Battle(args, tacview_show=1)
    for i_episode in range(1):  # 10
        r_action_list = []
        b_action_list = []

        battle_control_on_line = 0

        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE, line_number = creat_initial_state()
        initial_blue_height = DEFAULT_BLUE_BIRTH_STATE['position'][1]
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                  red_init_ammo=0, blue_init_ammo=0)

        done = False

        env.dt_maneuver = dt_maneuver
        step = 0
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            # 获取观测信息
            r_obs_n, r_obs_check = env.escape_obs('r')
            b_obs_n, b_obs_check = env.escape_obs('b')

            b_obs = np.squeeze(b_obs_n)

            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            
            # 红方开局就发射一枚导弹
            if env.RUAV.ammo>0:
                new_missile = env.RUAV.launch_missile(env.BUAV, env.t, missile_class)
                env.RUAV.ammo -= 1
                new_missile.side = 'r'
                env.Rmissiles.append(new_missile)
                env.missiles = env.Rmissiles + env.Bmissiles

            height_ego = env.BUAV.alt
            delta_psi = atan2(b_obs_check["target_information"][1], b_obs_check["target_information"][0])
            RWR_b = b_obs_check['warning']

            b_state = env.get_state('b')
            dist = b_state["threat"][3]
            # if dist<20e3:
            #     RWR_b = 1
            # else:
            #     RWR_b = 0
            # if RWR_b==1 and battle_control_on_line==0:
            battle_control_on_line = 1

            # 动作重映射不做了

            # 机动决策
            # 红方发射完导弹飞慢点，给蓝方机会
            L_ = env.BUAV.pos_ - env.RUAV.pos_
            q_beta = atan2(L_[2], L_[0])
            L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
            L_v = L_[1]
            q_epsilon = atan2(L_v, L_h)
            delta_psi = sub_of_radian(q_beta, env.RUAV.psi)
            r_action_n_0 = np.clip(env.BUAV.pos_[1], env.min_alt_safe, env.max_alt_safe)-env.RUAV.pos_[1]
            r_action_n_1 = delta_psi
            r_action_n_2 = env.BUAV.speed  ###
            r_action_n = [r_action_n_0, r_action_n_1, r_action_n_2]

            if np.isnan(b_obs_n).any() or np.isinf(b_obs_n).any():
                print('b_obs_n', b_obs_check)
                print()

            if not battle_control_on_line:
                L_ = env.RUAV.pos_ - env.BUAV.pos_
                q_beta = atan2(L_[2], L_[0])
                L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
                L_v = L_[1]
                q_epsilon = atan2(L_v, L_h)
                delta_psi = sub_of_radian(q_beta, env.BUAV.psi)
                b_action_n_0 = initial_blue_height - env.BUAV.pos_[1]
                b_action_n_1 = delta_psi               + 0*pi/180
                b_action_n_2 = 340
                b_action_n = np.array([b_action_n_0, b_action_n_1, b_action_n_2])
                print("prepare")
                print(env.t)
            else:
                print("escape")
                print(env.t)
                # print(env.BUAV.p)
                # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
                b_action_n, u = agent.take_action(b_obs, action_bounds=action_bound, explore=False)

                # 动作裁剪
                # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
                # if delta_psi>0:
                #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
                # else:
                #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])

            if b_obs_check["warning"] == 1:
                # print(b_obs_check["threat"])
                # print(b_obs_check["border"])
                print()

            # # 规则动作
            # delta_psi = b_obs_check["target_information"][1]
            # delta_height = b_obs_check["target_information"][0]
            # b_action_n = crank_behavior(delta_psi, delta_height*5000-2000)

            # # 动作平滑（实验性）
            b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.escape_terminate_and_reward('b')
            done = done  # debug 这里需要解决下仿真结束判断的问题

            if env.RUAV.dead:
                print()

            step += 1
            env.render(t_bias=t_bias)
            # time.sleep(0.01)

        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")
