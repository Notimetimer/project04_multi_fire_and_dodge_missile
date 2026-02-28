import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *

# 获取project目录
def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
    try:
        shell = get_ipython().__class__.__name__  # ← 误报，不用管
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
            # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
            return os.getcwd()
        else:  # 其他 shell
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))
current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from TrainAndTests.Controls.FlightControl_Train_dual_a_out import *

dt_maneuver= 0.2
action_eps = 0 # np.array([0.5, 0.8, 0]) # 0.7 # 动作平滑度

from Utilities.LocateDirAndAgents import *

state_dim = 8 + 7 + 2  # len(b_obs_spaces)
action_dim = 4  # b_action_spaces[0].shape[0]

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)
env = track_env(tacview_show=0)

mission_name = 'AttackP2P' # 加载别的actor使用

# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs\\attack")
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
out_range_count = 0

t_bias = 0
# 强化学习训练
rl_steps = 0
i_episode = 0
while i_episode<=3:
    i_episode += 1
    episode_return = 0
    
    init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

    birth_state={'position': np.array([0.0, init_height, 0.0]),
                        'psi': np.random.uniform(-pi/6, pi/6)
                        }
    height_req = np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
    psi_req = np.random.uniform(-pi, pi)
    v_req = np.random.uniform(0.8, 2.5)*340

    env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
    state, state_check = env.get_obs()
    done = False

    while not done:  # 每个训练回合
        # 1.执行动作得到环境反馈
        state, state_check = env.get_obs()
        env_obs_check = state_check
        actor_obs_check = env_obs_check.copy()
        actor_obs_check["ego_main"]=env_obs_check["ego_main"]
        actor_obs_check["border"]=np.zeros(2)
        actor_obs_check["border"]=np.array([1,0])
        actor_obs_check["target_information"]=np.zeros(8)
        actor_obs_check["target_information"][0] = env_obs_check["flight_cmd"][0] # cos delta psi
        actor_obs_check["target_information"][1] = env_obs_check["flight_cmd"][1] # sin delta psi
        actor_obs_check["target_information"][2] = env_obs_check["flight_cmd"][2] # delta theta
        actor_obs_check["target_information"][3] = 1 # 距离
        actor_obs_check["target_information"][4] = 0 # 夹角？？？
        actor_obs_check["target_information"][5] = 1 # 速度
        actor_obs = {k: (actor_obs_check[k].copy() if hasattr(actor_obs_check[k], "copy") else actor_obs_check[k]) \
                    for k in [
                        "target_information",  # 8
                        "ego_main",  # 7
                        "border",  # 2
                    ]}
        actor_obs = flatten_obs(actor_obs, [
                        "target_information",  # 8
                        "ego_main",  # 7
                        "border",  # 2
                    ])

        action, u = agent.take_action(actor_obs, action_bounds=action_bound, explore=0)
        rl_steps += 1
        
        # action[2] = 1
        next_state, reward, done = env.step(action)

        # debug 用
        height_req = env.height_req/1000
        height = env.RUAV.alt/1000
        psi_req = env.psi_req*180/pi
        psi = env.RUAV.psi*180/pi
        v_req = env.v_req
        v = env.RUAV.speed

        env.render(t_bias)

    env.clear_render(t_bias)
    t_bias += env.t

    if env.fail==1:
        out_range_count+=1