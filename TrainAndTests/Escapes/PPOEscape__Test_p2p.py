import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Escapes.PPOEscape__Train_p2p import *

import re

dt_maneuver = 0.2  # 0.2
action_eps = 0.7  # 动作平滑度

print("action_dim", action_dim)

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)


from Utilities.LocateDirAndAgents import *
pre_log_dir = os.path.join(project_root, "logs/escape")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)


# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None) # 1200 1140
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

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

def creat_initial_state():
    # 飞机出生状态指定
    # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
    cases = initial_states.shape[0]  # 读取攻击区数据
    line_number = np.random.choice(cases)

    red_height = initial_states[line_number, 0]
    blue_height = initial_states[line_number, 1]

    # red_height = 13e3
    # blue_height = 12e3

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
                               'psi': red_psi,
                               'p2p': False
                               }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi,
                                'p2p': True
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE, line_number


try:
    env = EscapeTrainEnv(args, tacview_show=1)  # Battle(args, tacview_show=1)
    success_list = []
    for i_episode in range(20):  # 10
        r_action_list = []
        b_action_list = []

        battle_control_on_line = 0

        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE, line_number = creat_initial_state()
        initial_blue_height = DEFAULT_BLUE_BIRTH_STATE['position'][1]
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                  red_init_ammo=1, blue_init_ammo=0)

        done = False

        env.dt_maneuver = dt_maneuver
        step = 0
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            # 获取观测信息
            r_obs_n, r_obs_check = env.escape_obs('r')
            b_obs_n, b_obs_check = env.escape_obs('b')

            # # 在这里将观测信息压入记忆
            # env.RUAV.obs_memory = r_obs_check.copy()
            # env.BUAV.obs_memory = b_obs_check.copy()

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
            if dist<20e3:
                RWR_b = 1
            else:
                RWR_b = 0
            if RWR_b==1 and battle_control_on_line==0:
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
            r_action_n_1 = delta_psi  # + 20*pi/180*random.choice([-1,1])
            r_action_n_2 = 300
            r_action_n = [r_action_n_0, r_action_n_1, r_action_n_2]

            if np.isnan(b_obs_n).any() or np.isinf(b_obs_n).any():
                print('b_obs_n', b_obs_check)
                # print()

            if not battle_control_on_line:
                env.BLUE_BIRTH_STATE['p2p']=False
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
                # print("prepare")
                # print(env.t)
            else:
                env.BLUE_BIRTH_STATE['p2p']=True
                # print("escape")
                # print(env.t)
                # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
                b_action_n, u = agent.take_action(b_obs, action_bounds=action_bound, explore=False)

                # # 可躲性测试
                # L_ = env.RUAV.pos_ - env.BUAV.pos_
                # q_beta = atan2(L_[2], L_[0])
                # b_action_n[0] = np.clip(-300, env.min_alt_safe-env.BUAV.pos_[1] , 5000)
                # b_action_n[1] = sub_of_radian(q_beta+pi, env.BUAV.psi)
                # b_action_n[2] = 400

                # 动作裁剪
                # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
                # if delta_psi>0:
                #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
                # else:
                #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])

            # if b_obs_check["warning"] == 1:
                # print()

            # # 规则动作
            # delta_psi = b_obs_check["target_information"][1]
            # delta_height = b_obs_check["target_information"][0]
            # b_action_n = crank_behavior(delta_psi, delta_height*5000-2000)

            # # # 动作平滑（实验性）
            # b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            # hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.escape_terminate_and_reward('b')
            done = done  # debug 这里需要解决下仿真结束判断的问题

            # if env.RUAV.dead:
            #     print()

            step += 1
            env.render(t_bias=t_bias)
            # time.sleep(0.01)

        if env.win:
            success_list.append(1)
        else:
            success_list.append(0)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")
finally:
    print("success_rate:", sum(success_list)/len(success_list))