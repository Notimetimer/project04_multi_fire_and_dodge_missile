import sys
import os
sys.path.append(project_root)

from TrainAndTests.PPOAttackTrain import *
import re

dt_maneuver= 0.2 
action_eps = 0 # np.array([0.5, 0.8, 0]) # 0.7 # 动作平滑度

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

#
log_dir_shoot = os.path.join(pre_log_dir, "Shoot-run-20251106-174339")

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

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
    agent.actor.load_state_dict(sd) # , strict=False)  # 忽略缺失的键
    print(f"Loaded actor for test from: {latest_actor_path}")

t_bias = 0

try:
    env = AttackTrainEnv(args, tacview_show=1) # Battle(args, tacview_show=1)
    for i_episode in range(3):  # 10
        r_action_list=[]
        b_action_list=[]
        # 飞机出生状态指定
        red_R_ = random.uniform(20e3, 60e3)
        red_beta = random.uniform(0, 2*pi)
        red_psi = random.uniform(0, 2*pi)
        red_height = random.uniform(3e3, 10e3)
        red_N = red_R_*cos(red_beta)
        red_E = red_R_*sin(red_beta)
        blue_height = random.uniform(3e3, 10e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi
                                }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                    'psi': pi
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=0, blue_init_ammo=0)
        env.dt_maneuver = dt_maneuver
        step = 0
        done = False
        hist_b_action = np.zeros(3)

        while not done:
            print(env.t)
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

except KeyboardInterrupt:
    print("验证已中断")