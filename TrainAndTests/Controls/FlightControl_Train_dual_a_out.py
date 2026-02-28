'''
当前做法的一个问题：
所有奖励是相加的，而且目标方向并不会随着姿态的变化而改变，当前训练的是什么？
是沿着一条安排好的轨迹动态跟踪，还是说只是照着指令里的速度、角速度和俯仰角去飞？
如果是后者，应该要区分“指令要你俯冲你俯冲撞地和指令没有要你俯冲你俯冲撞地的情况”
'''

use_tacview = 0

import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
from gym import spaces
import copy
import matplotlib.pyplot as plt
import json
import glob

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# # 获取project目录
# def get_current_file_dir():
#     # 判断是否在 Jupyter Notebook 环境
#     try:
#         shell = get_ipython().__class__.__name__  # ← 误报，不用管
#         if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
#             # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
#             return os.getcwd()
#         else:  # 其他 shell
#             return os.path.dirname(os.path.abspath(__file__))
#     except NameError:
#         # 普通 Python 脚本
#         return os.path.dirname(os.path.abspath(__file__))
# current_dir = get_current_file_dir()
# sys.path.append(os.path.dirname(current_dir))

from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *
from Visualize.tensorboard_visualize import *
from Algorithms.Wasted.PPOcontinues_dual_a_out import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Math_calculates.Calc_dist2border import calc_intern_dist2cylinder

class track_env():
    def __init__(self, dt_move=0.02, tacview_show=0):
        super(track_env, self).__init__()
        self.RUAV_ids = None
        self.dt_report = None
        self.dt_move = dt_move
        self.t = None
        # self.done = None
        self.success = None # 胜
        self.fail = None # 负
        self.draw = None # 平
        self.action_space = [spaces.Box(low=-1, high=+1, shape=(4,), dtype=np.float32)]
        self.DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, 0.0]),
                               'psi': 0
                               }
        
        self.time_limit = 8*60 # 300 t_last
        self.min_alt = 1e3
        self.min_alt_safe = 3e3

        self.max_alt_safe = 13e3
        self.max_alt = 15e3

        self.flight_key_order = [
            "ego_main",  # 7
            # "ego_control",  # 7
            "flight_cmd", # 3
        ]
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
    
    def reset(self, o00=None, birth_state=None, height_req=8e3, psi_req=0, v_req=340, dt_report=0.2, t0=0):
        self.t = t0
        self.success = 0
        # self.done = 0
        self.fail = 0
        self.draw = 0
        if o00 == None:
            o00 = np.array([118, 30])  # 地理原点的经纬
            self.o00 = o00
        if birth_state == None:
            birth_state = self.DEFAULT_RED_BIRTH_STATE
        self.dt_report = dt_report
        UAV = UAVModel(dt=self.dt_move)
        UAV.ammo = 0
        UAV.id = 1
        UAV.red = True
        UAV.blue = False
        UAV.side = "r"
        UAV.dead = 0
        UAV.color = np.array([1, 0, 0])
        # 红方出生点
        UAV.pos_ = self.DEFAULT_RED_BIRTH_STATE['position']
        UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
        speed = UAV.speed
        UAV.psi = self.DEFAULT_RED_BIRTH_STATE['psi']
        UAV.theta = 0 * pi / 180
        UAV.gamma = 0 * pi / 180
        UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                            sin(UAV.theta),
                                            cos(UAV.theta) * sin(UAV.psi)])
        lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
        UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                    theta0=UAV.theta, o00=o00)
        self.RUAV = UAV
        
        # △h动作输出有效性测试
        self.height_req = height_req
        self.psi_req = psi_req
        self.v_req = v_req


    def get_state(self, side='r'):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        own = self.RUAV

        # 夹角
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        
        # 速度观测量
        v_own = v
        # 本机高度
        h_own = own.alt
        # 本机俯仰角
        sin_theta = sin(own.theta)
        cos_theta = cos(own.theta)
        # 本机滚转角
        sin_phi = sin(own.phi)
        cos_phi = cos(own.phi)

        # 剩余导弹量
        ammo = 0

        p = own.p
        q = own.q
        r = own.r

        theta_v = own.theta_v
        psi_v = own.psi_v

        alpha_air = own.alpha_air
        beta_air = own.beta_air

        delta_psi_control = sub_of_radian(self.psi_req, self.RUAV.psi)
        cos_delta_psi = cos(delta_psi_control)
        sin_delta_psi = sin(delta_psi_control)
        delta_theta_control = np.clip((self.height_req-self.RUAV.alt)/5000, -1, 1)*pi/2

        one_side_states = {
            "ego_main": np.array([
                float(v_own),  # 0本机速度 m/s
                float(h_own),  # 1本机高度 m
                float(sin_theta),  # 2
                float(cos_theta),  # 3
                float(sin_phi),  # 4
                float(cos_phi),  # 5
                int(ammo)  # 6剩余导弹数量
            ]),

            "ego_control": np.array([
                float(p),  # 0 p rad/s act1_last
                float(q),  # 1 q rad/s act2_last
                float(r),  # 2 r rad/s act3_last
                float(theta_v),  # 3
                float(psi_v),  # 4
                float(alpha_air),  # 5 rad
                float(beta_air)  # 6 rad
            ]),

            "flight_cmd":  np.array([
                cos_delta_psi,
                sin_delta_psi,
                delta_theta_control - self.RUAV.theta_v,
                self.v_req - self.RUAV.speed,
            ])
        }
        return one_side_states
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340

        s["flight_cmd"][2] /= pi*2
        s["flight_cmd"][3] /= 340
        return s
        
    def base_obs(self, side='r', pomdp=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        state = self.get_state(side)  # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
        self.state_init["flight_cmd"] = np.array([0, 0, 0, 0])

        observation = self.scale_state(state)
        self.obs_init = self.scale_state(self.state_init)
        return observation

    def get_obs(self, side='r'):
        pre_full_obs = self.base_obs()
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.flight_key_order}
        
        # 不能被看到
        full_obs["ego_main"][6] = 0
        # full_obs["ego_control"][4] = 0

        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.flight_key_order)
        return flat_obs, full_obs

    def step(self, action):
        self.action = action
        target_height, delta_heading, target_speed, rudder = action
        self.t += self.dt_report
        time_rate = int(round(self.dt_report/self.dt_move))
        for _ in range(time_rate):
            self.RUAV.move(target_height, delta_heading, target_speed, relevant_height=True, relevant_speed=False, with_theta_req=False, p2p=True, rudder=rudder)
            done = self.get_done()
            if done:
                break
        next_obs, _ = self.get_obs()
        reward = self.get_reward()
        
        return next_obs, reward, done

    def get_done(self,):
        done = 0
        # 超时结束
        if self.t > self.time_limit:
            done = 1
        ruav_state = self.get_state()
        alt = ruav_state["ego_main"][1]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        # 失败条件：失速、高度过低
        self.fail = 0
        if alt < self.min_alt or alpha_air < -20 or alpha_air > 45 or abs(beta_air) > 15:
            self.fail = 1
            done = 1
            self.RUAV.dead = 1
        return done

    # def get_reward(self, ):
    #     ruav_state = self.get_state()
    #     v = ruav_state["ego_main"][0]
    #     alt = ruav_state["ego_main"][1]
    #     sin_theta = ruav_state["ego_main"][2]
    #     cos_theta = ruav_state["ego_main"][3]
    #     sin_phi = ruav_state["ego_main"][4]
    #     cos_phi = ruav_state["ego_main"][5]
    #     p = ruav_state["ego_control"][0]
    #     q = ruav_state["ego_control"][1]
    #     r = ruav_state["ego_control"][2]
    #     theta_v = ruav_state["ego_control"][3]
    #     psi_v = ruav_state["ego_control"][4]
    #     alpha_air = ruav_state["ego_control"][5]*180/pi
    #     beta_air = ruav_state["ego_control"][6]*180/pi
    #     climb_rate = self.RUAV.vu

    #     self.get_done()

    #     # 存活奖励
    #     reward_alive = 0 # 10

    #     # 失败惩罚
    #     reward_end = 0
    #     if self.fail:
    #         reward_end -= 100

    #     # 高度奖励（高度变化率惩罚与高度限制惩罚）
    #     reward_height = 0
    #     height_error = np.clip(self.height_req-alt, -5000, 5000)
    #     theta_v_req = height_error/5000*pi/2
        
    #     self.theta_v_req = theta_v_req
    #     reward_height += 1-abs(theta_v-theta_v_req)/pi

    #     # if abs(height_error) < 500:
    #     #     reward_height += 1-abs(climb_rate)/100

    #     if alt<self.min_alt_safe:
    #         reward_height += climb_rate/100 + (alt-self.min_alt)/(self.min_alt_safe-self.min_alt)
        
    #     # 航向奖励（误差惩罚）
    #     # psi_error = sub_of_radian(self.psi_req, psi_v)
    #     # reward_psi_error = 1-abs(psi_error)/pi

    #     # 角度奖励
    #     L_ = np.array([cos(theta_v_req)*cos(self.psi_req), sin(theta_v_req), cos(theta_v_req)*sin(self.psi_req)])
    #     ATA = np.arccos(np.dot(L_, self.RUAV.point_) / (1*1 + 0.0001))  # 防止计算误差导致分子>分母
    #     reward_psi_error = 1 - ATA/pi

    #     # 速度奖励（速度误差惩罚）
    #     reward_v = 1-abs(self.v_req-v)/340

    #     # 滚转角奖励
    #     reward_phi = min(cos_phi, 0)

    #     # pqr奖励(快到位时候pqr越小越好)
    #     reward_pqr = 0
    #     reward_pqr = -abs(p/(2*pi))-abs(q/(20*pi/180))-abs(r/(20*pi/180))
    #     if abs(alpha_air) > 15:
    #         reward_pqr -= abs(p/(90*pi/180))

    #     # 迎角过载奖励(惩罚负迎角和过大的正迎角)
    #     reward_alpha = 0.5
    #     if alpha_air >= 15:
    #         reward_alpha -= alpha_air/15
    #     if alpha_air < 0:
    #         reward_alpha += alpha_air/2       
    #     ny = self.RUAV.Ny
    #     if ny<=-1 or ny > 9:
    #         reward_alpha -= 2
            

    #     # 侧滑角奖励（尽量少侧滑）
    #     reward_beta = 0.5-abs(beta_air/5)

    #     reward = np.sum([
    #         1 * reward_alive,
    #         1 * reward_end,
    #         2 * reward_height,
    #         2 * reward_psi_error,
    #         1 * reward_v,
    #         0.3 * reward_phi,
    #         0.1 * reward_pqr,
    #         1 * reward_alpha,
    #         1 * reward_beta,
    #     ])

    #     return reward
    

    def get_reward(self, ):
        ruav_state = self.get_state()
        speed = ruav_state["ego_main"][0]
        alt = ruav_state["ego_main"][1]
        sin_theta = ruav_state["ego_main"][2]
        cos_theta = ruav_state["ego_main"][3]
        sin_phi = ruav_state["ego_main"][4]
        cos_phi = ruav_state["ego_main"][5]
        phi = atan2(sin_phi, cos_phi)
        p = ruav_state["ego_control"][0]
        q = ruav_state["ego_control"][1]
        r = ruav_state["ego_control"][2]
        theta_v = ruav_state["ego_control"][3]
        psi_v = ruav_state["ego_control"][4]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        climb_rate = self.RUAV.vu

        self.get_done()

        # 存活奖励
        reward_alive = 0 # 10

        # 失败惩罚
        reward_end = 0
        if self.fail:
            reward_end -= 100

        height_error = np.clip(self.height_req-alt, -5000, 5000)
        theta_v_req = height_error/5000*pi/2
        
        self.theta_v_req = theta_v_req
        
        # 航向奖励（误差惩罚）
        # psi_error = sub_of_radian(self.psi_req, psi_v)
        # reward_psi_error = 1-abs(psi_error)/pi

        L_ = np.array([cos(theta_v_req)*cos(self.psi_req), sin(theta_v_req), cos(theta_v_req)*sin(self.psi_req)])
        ATA = np.arccos(np.dot(L_, self.RUAV.point_) / (1*1 + 0.0001))  # 防止计算误差导致分子>分母
        
        # 角度奖励
        r_angle = 1 - ATA / (pi / 3)  # 超出雷达范围就惩罚狠一点
        r_angle -= 0.05 * abs(phi) # 滚转角惩罚

        # 高度奖励
        pre_alt_opt = self.height_req
        alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)

        r_alt = (alt <= alt_opt) * (alt - self.min_alt) / (alt_opt - self.min_alt) + \
                (alt > alt_opt) * (1 - (alt - alt_opt) / (self.max_alt - alt_opt))
        # 高度限制奖励/惩罚
        r_alt += (alt <= self.min_alt_safe + 1e3) * np.clip(self.RUAV.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-self.RUAV.vu / 100, -1, 1)

        # 速度奖励
        r_speed = abs(self.v_req-speed) / (340)

        # 迎角过载奖励(惩罚负迎角和过大的正迎角)
        reward_alpha = 0.5
        if alpha_air >= 15:
            reward_alpha -= alpha_air/15
        if alpha_air < 0:
            reward_alpha += alpha_air/2       
        ny = self.RUAV.Ny
        if ny<=-1 or ny > 9:
            reward_alpha -= 2
            
        # 侧滑角奖励（尽量少侧滑）
        reward_beta = 0.5-abs(beta_air/5)

        reward = np.sum([
            1 * reward_alive,
            1 * reward_end,
            2 * r_angle,
            2 * r_alt,
            1 * r_speed,
            1 * reward_alpha,
            1 * reward_beta,
        ])

        # 其他奖励待续
        return reward
        

    def render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            loc_LLH = self.RUAV.lon, self.RUAV.lat, self.RUAV.alt
            if not self.RUAV.dead:
                pilot = 'Fool'
                color = 'Red'
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                            f"{self.RUAV.phi * 180 / pi:.6f}|{self.RUAV.theta * 180 / pi:.6f}|{self.RUAV.psi * 180 / pi:.6f},"
                            f"Name=F16,Pilot={pilot},Color={color}\n"
                        )
                # 绘制目标
                delta_N = 5e3*cos(self.theta_v_req)*cos(self.psi_req)
                delta_U = 5e3*sin(self.theta_v_req)
                delta_E = 5e3*cos(self.theta_v_req)*sin(self.psi_req)
                N, U, E = LLH2NUE(loc_LLH[0], loc_LLH[1], loc_LLH[2], lon_o=self.o00[0], lat_o=self.o00[1])
                delta_H = self.height_req
                lon_T, lat_T, _ = NUE2LLH(N+delta_N,U+delta_U,E+delta_E,lon_o=self.o00[0], lat_o=self.o00[1])
                
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id+1},T={(lon_T):.6f}|{(lat_T):.6f}|{delta_H:.6f},"
                            f"Name=Target,Color=Blue\n"
                        )

            else:
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"

            # loc_r = [self.RUAV.lon, self.RUAV.lat, self.RUAV.alt]
            # if self.tacview_show:
            #     data_to_send = ''
            #     data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            #             float(self.t + t_bias), self.RUAV.id, loc_r[0], loc_r[1], loc_r[2], self.RUAV.phi * 180 / pi, self.RUAV.theta * 180 / pi,
            #             self.RUAV.psi * 180 / pi)

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"
            self.tacview.send_data_to_client(data_to_send)


# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
max_steps = 10 * 65e4
hidden_dim = [128, 128]  # 128, 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
dt_decide = 0.2 # 0.2
pre_train_rate = 0 # 0.25 # 0.25

state_dim = 7+4  # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
action_dim = 4 # test
action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
mission_name = 'FlightControl'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
log_dir = os.path.join(project_root, "./logs/control", mission_name + "-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__=='__main__':
    env = track_env(tacview_show=use_tacview)
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)
        
    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    def save_meta_once(path, state_dict):
        if os.path.exists(path):
            return
        meta = {k: list(v.shape) for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(meta, f)

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Math_calculates.ScaleLearningRate import scale_learning_rate
    # 根据参数数量缩放学习率
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)

    from Visualize.tensorboard_visualize import TensorBoardLogger

    out_range_count = 0
    return_list = []
    steps_count = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    try:
        t_bias = 0
        # 强化学习训练
        rl_steps = 0
        i_episode = 0
        while rl_steps < max_steps:
            i_episode += 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
            
            init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

            birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)
                                }
            
            delta_height_generate = np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 * np.clip(i_episode/1000, 0, 1)
            height_req = np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
            psi_req = np.random.uniform(-pi, pi) * np.clip(i_episode/1000, 0, 1)
            v_req = np.random.uniform(0.8, 2.5)*340

            env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
            state, state_check = env.get_obs()
            done = False

            while not done:  # 每个训练回合
                # 1.执行动作得到环境反馈
                state, state_check = env.get_obs()
                action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
                rl_steps += 1
                
                next_state, reward, done = env.step(action)

                # debug 用
                height_req = env.height_req/1000
                height = env.RUAV.alt/1000
                psi_req = env.psi_req*180/pi
                psi = env.RUAV.psi*180/pi
                v_req = env.v_req
                v = env.RUAV.speed

                transition_dict['states'].append(state)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['action_bounds'].append(action_bound)
                state = next_state
                episode_return += reward * env.dt_report # 奖励按秒分析
                env.render(t_bias)

            env.clear_render(t_bias)
            t_bias += env.t

            if env.fail==1:
                out_range_count+=1
            return_list.append(episode_return)
            agent.update(transition_dict)

            # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
            if i_episode % 10 == 1:
                # critic overwrite
                critic_path = os.path.join(log_dir, "critic.pt")
                th.save(agent.critic.state_dict(), critic_path)
                # actor RL snapshot
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

            
            # tqdm 训练进度显示
            if (i_episode + 1) >= 10:
                print(f"episode {i_episode+1}, 进度: {rl_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")

            # tensorboard 训练进度显示
            logger.add("train/0 episode_return", episode_return, rl_steps)
            logger.add("train/0 survive", 1-env.fail, rl_steps)

            actor_grad_norm = model_grad_norm(agent.actor)
            critic_grad_norm = model_grad_norm(agent.critic)
            # 梯度监控
            logger.add("train/1 actor_grad_norm", actor_grad_norm, rl_steps)
            logger.add("train/2 critic_grad_norm", critic_grad_norm, rl_steps)
            # 损失函数监控
            logger.add("train/3 actor_loss", agent.actor_loss, rl_steps)
            logger.add("train/4 critic_loss", agent.critic_loss, rl_steps)
            # 强化学习actor特殊项监控
            logger.add("train/5 entropy", agent.entropy_mean, rl_steps)
            logger.add("train/6 ratio", agent.ratio_mean, rl_steps)
            logger.add("train/7 steps", i_episode + 1, rl_steps)


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()


        print(f"日志已保存到：{logger.run_dir}")

