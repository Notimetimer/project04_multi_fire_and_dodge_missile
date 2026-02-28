import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Attacks.PPOAttack__TrainP2P import *
import re
import glob

dt_maneuver= 0.2
action_eps = 0 # np.array([0.5, 0.8, 0]) # 0.7 # 动作平滑度

from Utilities.LocateDirAndAgents import *

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs/attack")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Attack-run-20251031-094218")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None)
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

try:
    r_action_list=[]
    b_action_list=[]
    env = AttackTrainEnv(args, tacview_show=1) # Battle(args, tacview_show=1)
    for i_episode in range(3):  # 10
        
        # 飞机出生状态指定
        red_R_ = random.uniform(50e3, 60e3) # 60
        red_beta = random.uniform(0, 2*pi)
        red_psi = random.uniform(0, 2*pi)
        red_height = random.uniform(3e3, 10e3)
        red_N = red_R_*cos(red_beta)
        red_E = red_R_*sin(red_beta)
        blue_height = random.uniform(3e3, 10e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi,
                                'p2p': False
                                }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                    'psi': pi,
                                    'p2p': True
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=0, blue_init_ammo=0)
        env.dt_maneuver = dt_maneuver
        step = 0
        done = False
        hist_b_action = np.zeros(4)

        while not done:
            # print(env.t)
            r_obs_n, r_obs_check = env.attack_obs('r')
            b_obs_n, b_obs_check = env.attack_obs('b')
            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()
            state = np.squeeze(b_obs_n)
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1
                                    )
            b_action_n, u = agent.take_action(state, action_bounds=action_bound, explore=False)
            
            # 动作平滑（实验性）
            b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.get_terminate_and_reward('b')

            step += 1
            env.render(t_bias=t_bias)
            time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t
        1+1

except KeyboardInterrupt:
    print("验证已中断")

finally:
    episode_len = len(b_action_list)
    # 时间轴从 1 到 episode_len，每步乘以 env.dt_maneuver
    episode_ts = env.dt_maneuver * np.arange(1, episode_len + 1)

    try:
        import matplotlib.pyplot as plt

        actions = np.array(b_action_list)  # shape (n, 3)
        if actions.size == 0:
            print("没有动作数据可绘制。")
        else:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
            labels = ['elevator', 'aileron', 'throttle']
            for i, ax in enumerate(axes):
                ax.plot(episode_ts, actions[:, i], marker='o', markersize=3, linestyle='-')
                ax.set_ylabel(labels[i])
                ax.grid(True)
            axes[-1].set_xlabel('time (s)')
            axes[-1].set_xlim(episode_ts[0], episode_ts[-1])
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print("绘图失败:", e)
