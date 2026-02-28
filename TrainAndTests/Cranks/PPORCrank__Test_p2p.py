import sys
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from PPORCrank__Train_std_clip_p2p import *

import re

dt_maneuver= 0.2 # 0.2 
action_eps = 0 # 动作平滑度

# 作为参考的规则动作
def crank_behavior(delta_psi, delta_height):
    """
    crank 行为：返回 (heading_cmd, speed_cmd)
    """
    delta_psi_angle = delta_psi*180/pi
    temp = delta_psi_angle - 55
    heading_cmd = temp*pi/180
    speed_cmd = 1.1 * 340
    return delta_height, heading_cmd, speed_cmd

# # 找出日期最大的目录
# def get_latest_log_dir(pre_log_dir, mission_name=None):
#     # 匹配 run-YYYYMMDD-HHMMSS 目录
#     # pattern = re.compile(r"run-(\d{8})-(\d{6})")
#     if mission_name:
#         pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
#     else:
#         pattern = re.compile(r"run-(\d{8})-(\d{6})")
#     max_dt = None
#     latest_dir = None
#     for d in os.listdir(pre_log_dir):
#         m = pattern.match(d)
#         if m:
#             dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
#             if max_dt is None or dt_str > max_dt:
#                 max_dt = dt_str
#                 latest_dir = d
#     if latest_dir:
#         return os.path.join(pre_log_dir, latest_dir)
#     else:
#         return None

# # pre_log_dir = os.path.join("./logs")
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

# # log_dir = os.path.join(pre_log_dir, "LCrank-run-20251024-112053")

# print("\n\n\n")
# print("logdir", log_dir)
# print("\n\n\n")

print("action_dim", action_dim)
# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

# rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
# sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
# latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
# if latest_actor_path:
#     # 直接加载权重到现有的 agent
#     sd = th.load(latest_actor_path, map_location=device)
#     agent.actor.load_state_dict(sd) # , strict=False)  # 忽略缺失的键
#     print(f"Loaded actor for test from: {latest_actor_path}")

from Utilities.LocateDirAndAgents import *
pre_log_dir = os.path.join(project_root, "logs/rcrank")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "LCrank_with_std_clipping-run-20251025-180743")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None)
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

try:
    env = CrankTrainEnv(args, tacview_show=1) # Battle(args, tacview_show=1)

    for i_episode in range(3):  # 10
        r_action_list=[]
        b_action_list=[]
        # 
        blue_height = random.uniform(5e3, 9e3)
        red_height = blue_height

        blue_psi = pi/2
        red_psi = -pi/2
        red_N = 54e3  # random.choice([-54e3, 54e3])
        red_E = 35e3
        blue_N = red_N
        blue_E = -35e3

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                    'psi': red_psi,
                                    'p2p': False}
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi,
                                    'p2p': True}

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=0, blue_init_ammo=0)
        

        done = False
        
        env.dt_maneuver = dt_maneuver
        step = 0
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            # 获取观测信息
            r_obs_n, r_obs_check = env.crank_obs('r')
            b_obs_n, b_obs_check = env.crank_obs('b')

            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()
            state = np.squeeze(b_obs_n)
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            # 开局就发射一枚导弹
            if env.BUAV.ammo>0:
                new_missile = env.BUAV.launch_missile(env.RUAV, env.t, missile_class)
                env.BUAV.ammo -= 1
                new_missile.side = 'blue'
                env.Bmissiles.append(new_missile)
                env.missiles = env.Rmissiles + env.Bmissiles
            # 机动决策
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1
                                    )
            if np.isnan(b_obs_n).any() or np.isinf(b_obs_n).any():
                print('b_obs_n', b_obs_check)
                print()
            
            # 神经网络输出动作
            b_action_n, u = agent.take_action(state, action_bounds=action_bound, explore=0) 

            # # 规则动作
            delta_psi = b_obs_check["target_information"][1]
            # delta_height = b_check_obs["target_information"][0]
            # b_action_n = crank_behavior(delta_psi, delta_height*5000-2000)
            height_ego = env.BUAV.alt

            # # # 动作裁剪
            # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
            # if delta_psi>0:
            #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
            # else:
            #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])


            # # 动作平滑（实验性）
            # b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.left_crank_terminate_and_reward('b')
            done = done # debug 这里需要解决下仿真结束判断的问题

            if env.RUAV.dead:
                print()

            step += 1
            env.render(t_bias=t_bias)
            time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

        if env.lose:
            lose = 1
        else:
            lose = 0
        
        win = not lose
        print("本回合结果", win)

except KeyboardInterrupt:
    print("验证已中断")