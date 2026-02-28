import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Attacks.PPOAttack__Train import *
import re
import glob

dt_maneuver= 0.2 
POMDP = 1

from Utilities.LocateDirAndAgents import *

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs/attack")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Attack-run-20251031-094218")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None, rein_prefix="actor_rein") # actor_rein actor_save
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

try:
    env = AttackTrainEnv(args, tacview_show=1) # 是否用Tacview可视化
    for i_episode in range(3):  # 10
        r_action_list=[]
        b_action_list=[]
        # 飞机出生状态指定
        init_distance = 90e3
        red_R_ = init_distance/2 # random.uniform(20e3, 60e3)
        blue_R_ = init_distance/2
        red_beta = pi # random.uniform(0, 2*pi)
        red_psi = 0 # random.uniform(0, 2*pi)
        red_height = 8e3 # random.uniform(3e3, 10e3)
        red_N = red_R_*cos(red_beta)
        red_E = red_R_*sin(red_beta)
        blue_height = 8e3 # random.uniform(3e3, 10e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': np.random.choice([pi/2, -pi/2]) # red_psi
                                }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_R_, blue_height, 0.0]),
                                    'psi': np.random.choice([pi/2, -pi/2])
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        env.dt_maneuver = dt_maneuver
        step = 0
        done = False
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            r_obs_n, r_obs_check = env.attack_obs('r', pomdp=POMDP)
            b_obs_n, b_obs_check = env.attack_obs('b', pomdp=POMDP)

            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1
                                    )
            # r_action_n, u_r = agent.take_action(r_obs_n, action_bounds=action_bound, explore=False)
            b_action_n, u = agent.take_action(b_obs_n, action_bounds=action_bound, explore=False)
            

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            # ### 发射导弹，这部分不受10step约束
            # distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            # # 发射导弹判决
            # if distance <= 80e3 and distance >= 5e3:  # 在合适的距离范围内每0.2s判决一次导弹发射
            #     launch_time_count = 0
            #     launch_missile_with_basic_rules(env, side='r')
            #     launch_missile_with_basic_rules(env, side='b')

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.get_terminate_and_reward('b')

            step += 1
            env.render(t_bias=t_bias)
            # time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")