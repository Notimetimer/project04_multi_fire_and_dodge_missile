import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Cranks.PPOLCrank__Train_std_clip import *
# from TrainAndTests.PPOLCrankTraining_std_clipping import *
import re

dt_maneuver= 0.2  # 0.2
action_eps = 0  # 动作平滑度
use_tacview = 1  # 可视化仿真

# 作为参考的规则动作
def crank_behavior(delta_psi, delta_height):
    """
    crank 行为：返回 (heading_cmd, speed_cmd)
    """
    delta_psi_angle = delta_psi*180/pi
    temp = delta_psi_angle - 40
    heading_cmd = temp*pi/180
    speed_cmd = 1.1 * 340
    return np.array([delta_height, heading_cmd, speed_cmd])

t_bias = 0

parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=180,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")
args = parser.parse_args()

try:
    env = CrankTrainEnv(args, tacview_show=use_tacview) # Battle(args, tacview_show=1)
    win_list = []
    return_list = []
    for i_episode in range(100):  # 10
        r_action_list=[]
        b_action_list=[]
        # 
        blue_height = random.uniform(5e3, 9e3)
        red_height = blue_height

        blue_psi = pi/2
        red_psi = -pi/2
        red_N = 0  # 54e3  # random.choice([-54e3, 54e3])
        red_E = 35e3
        blue_N = red_N
        blue_E = -35e3

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi}
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi}

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=0, blue_init_ammo=0)
        

        done = False
        episode_return = 0
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
            
            # # 规则动作
            cos_delta_psi = b_obs_check["target_information"][0]
            sin_delta_psi = b_obs_check["target_information"][1]
            delta_psi = atan2(sin_delta_psi, cos_delta_psi)
            delta_theta = b_obs_check["target_information"][2]
            dist = b_obs_check["target_information"][3]
            delta_height = env.RUAV.alt - env.BUAV.alt
            b_action_n = crank_behavior(delta_psi, delta_height)
            height_ego = env.BUAV.alt

            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.left_crank_terminate_and_reward('b')
            done = done # debug 这里需要解决下仿真结束判断的问题

            if env.RUAV.dead:
                print()

            step += 1
            episode_return += b_reward
            env.render(t_bias=t_bias)
            if use_tacview:
                time.sleep(0.01)

        env.clear_render(t_bias=t_bias)
        t_bias += env.t

        if env.lose:
            lose = 1
        else:
            lose = 0
        
        win = not lose
        print("本回合结果", win)
        win_list.append(win)
        return_list.append(episode_return)


except KeyboardInterrupt:
    print("验证已中断")


finally:
    # 绘制胜利次数和回合奖励曲线
    import matplotlib.pyplot as plt

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 绘制胜率曲线
    ax1.plot(range(len(win_list)), win_list)  # , 'b-o')
    ax1.set_title('Win/Loss per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win (1) / Loss (0)')
    ax1.grid(True)

    # 绘制累积奖励曲线
    ax2.plot(range(len(return_list)), return_list)  # , 'r-o')
    ax2.set_title('Episode Return')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    ax2.grid(True)

    # 调整布局并显示
    plt.tight_layout()
    plt.show()
