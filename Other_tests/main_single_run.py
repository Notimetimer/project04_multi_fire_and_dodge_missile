from Envs.battle3dof1v1_proportion import *  # Battle
from math import pi
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
stop_flag = False  # 控制退出的全局变量
import time
import socket
import threading
# 地理原点
Longitude0 = 144 + 43 / 60
Latitude0 = 13 + 26 / 60
Height0 = 0
mark = np.array([Longitude0, Latitude0, Height0])  # 地理原点
def ENU2LLH(mark, NUE):
    # 东北天单位为m，经纬度单位是角度
    N, U, E = NUE
    # E, N, U = ENU
    longit0, latit0, height0 = mark
    R_earth = 6371004  # ???
    dlatit = N / R_earth * 180 / pi
    dlongit = E / (R_earth * cos(latit0 * pi / 180)) * 180 / pi
    dheight = U
    out = np.array([longit0 + dlongit, latit0 + dlatit, height0 + dheight])
    return out

# 我是谁的支线
# main_test_SAC2_3dim_PID, main_test_SAC2_3dim_P2P
from main_SAC_3dim_proportion import SACContinuous, hidden_dim, r_action_dim, action_bound, actor_lr0, critic_lr0, \
    alpha_lr, \
    tau, gamma, r_obs_dim, dt, weak_rolling_optim, args, env, device


class Tacview(object):
    def __init__(self):
        host = "localhost"
        port = 42674
        # host = input("请输入服务器IP地址：")
        # port = int(input("请输入服务器端口："))
        # 提示用户打开tacview软件高级版，点击"记录"-"实时遥测"
        print("请打开tacview软件高级版，点击\"记录\"-\"实时遥测\"，并使用以下设置：")
        print(f"IP地址：{host}")
        print(f"端口：{port}")

        # 创建套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host, port))

        # 启动监听
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        # 等待客户端连接
        client_socket, address = server_socket.accept()
        print(f"Accepted connection from {address}")

        self.client_socket = client_socket
        self.address = address

        # 构建握手数据
        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        # 发送握手数据
        client_socket.send(handshake_data.encode())


        # 接收客户端发送的数据
        data = client_socket.recv(1024)
        print(f"Received data from {address}: {data.decode()}")
        print("已建立连接")

        # 向客户端发送头部格式数据

        data_to_send = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                        "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
                        )
        client_socket.send(data_to_send.encode())

    def send_data_to_client(self, data):

        self.client_socket.send(data.encode())
tacview = Tacview()

target_entropy = -env.r_action_spaces[0].shape[0]

steps = 0
agent = SACContinuous(r_obs_dim, hidden_dim, r_action_dim, action_bound,
                      actor_lr0, critic_lr0, alpha_lr, target_entropy, tau,
                      gamma, device)


# agent.actor.load_state_dict(torch.load('3dim_rule_actor_test.pth',
#            map_location=torch.device('cpu'), weights_only=True))  # 自动选择设备且只读取权重和偏置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent.actor.load_state_dict(torch.load('3dim_rule_actor_test.pth',
           map_location=device, weights_only=True))  # 根据当前环境自动选择设备

# 3dim_rule_actor_test.pth, 3dim_rule_actor_p2p

steps_of_this_episode = 0

episode_return = 0
env.reset()
done = False
start_time = time.time()

# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))

Rtrajectory = []
Btrajectory = []

from guid2contr_abs import guided_control

# 环境运行一轮的情况
'episode start'
for count in range(round(args.max_episode_len / dt)):
    # 01
    steps += 1
    steps_of_this_episode += 1
    # 回合结束判断
    done = env.running == False or count == round(args.max_episode_len / dt) - 1
    if done:
        # print('回合结束，时间为：', env.t, 's')
        break

    # 03
    # 获取观测信息
    r_obs_n, b_obs_n = env.get_obs()
    mask = np.zeros(13)  # test 屏蔽暂时顾不了的观测信息
    mask[[7, 8]] = 1
    # mask[3] = 1 / 100  # 速率特征缩放
    r_obs_n = r_obs_n * mask

    state = np.squeeze(r_obs_n)
    # 执行动作
    # r_action_n = [[-1, 0.0, 0.001]]  # 机动动作输入结构，期望nx,ny和gamma，范围[-1,1]
    b_action_n = weak_rolling_optim(env, side='b')  # 短视优化
    # 红方使用SAC执行动作得到环境反馈

    red_required_plus = agent.take_action(np.squeeze(r_obs_n))

    '叠加当前角度和速度'
    # 计算当前速度和角度
    current_velocity = env.RUAV.vel_
    current_speed = np.linalg.norm(env.RUAV.vel_)
    current_psi = np.arctan2(env.RUAV.vel_[2], env.RUAV.vel_[0])
    current_theta = np.arctan2(env.RUAV.vel_[1], np.sqrt(env.RUAV.vel_[0] ** 2 + env.RUAV.vel_[2] ** 2))

    theta_req = red_required_plus[0] * np.pi / 2
    psi_req = red_required_plus[1] * np.pi + current_psi  # red_required_plus[1]*np.pi # + current_psi
    v_req = (0.5 + 0.5 * red_required_plus[2]) * (env.RUAV.speed_max - env.RUAV.speed_min) + \
            env.RUAV.speed_min  # 340  # current_R_speed + 1/2*(1+red_required_plus[0])*340

    v_req = np.clip(v_req, env.RUAV.speed_min, env.RUAV.speed_max)

    # 04 引导式控制
    r_action_n = guided_control(env.RUAV, theta_req, psi_req, v_req)

    # # test
    # b_action_n = guided_control(env.BUAV, -6*pi/180, -140*pi/180, 200)

    # 执行动作并获取环境反馈
    if 1:
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, record=True)
        # print(theta_req * 180 / pi)
        # 记录当前环境中飞机的运动状态
        red_pos_list = np.vstack((red_pos_list, env.RUAV.pos_))
        blue_pos_list = np.vstack((blue_pos_list, env.BUAV.pos_))
    done = r_dones or b_dones

    '''显示运行轨迹'''
    print("红方位置：", env.RUAV.pos_)
    print("蓝方位置：", env.BUAV.pos_)

    # 红
    position = env.RUAV.pos_
    angle = np.array([env.RUAV.gamma, env.RUAV.theta, env.RUAV.psi]) * 180 / pi
    Rtrajectory.append(np.hstack((position, angle)))
    # 蓝
    position = env.BUAV.pos_
    angle = np.array([env.BUAV.gamma, env.BUAV.theta, env.BUAV.psi]) * 180 / pi
    Btrajectory.append(np.hstack((position, angle)))

    if stop_flag:
        break
'episode end'

# print('steps_of_this_episode:', steps_of_this_episode)
print('period:', env.t)
# print('end_speed', env.RUAV.speed)
'num/10 end'

end_time = time.time()  # 记录结束时间
print(f"程序运行时间: {end_time - start_time} 秒")

if not stop_flag:
    # 最后一幕轨迹显示
    red_pos_list = env.RUAV.trajectory
    blue_pos_list = env.BUAV.trajectory
    # 可视化
    red_p_show_show = np.array(red_pos_list).T
    blue_p_show_show = np.array(blue_pos_list).T

    plt.figure(2)
    from show_trajectory import show_trajectory

    show_trajectory(red_pos_list, blue_pos_list, min_east, min_north, min_height, max_east, max_north, max_height,
                    r_show=1, b_show=1)

    # # 保存数据
    # Rtrajectory = np.array(Rtrajectory)
    # Btrajectory = np.array(Btrajectory)
    #
    # dataframe = pd.DataFrame({'N': Rtrajectory[:, 0], 'U': Rtrajectory[:, 1], 'E': Rtrajectory[:, 2],
    #                           'Roll': Rtrajectory[:, 3], 'Pitch': Rtrajectory[:, 4], 'Yaw': Rtrajectory[:, 5]})
    # dataframe.to_csv("testR.csv", index=False, sep=',')
    # dataframe = pd.DataFrame({'N': Btrajectory[:, 0], 'U': Btrajectory[:, 1], 'E': Btrajectory[:, 2],
    #                           'Roll': Btrajectory[:, 3], 'Pitch': Btrajectory[:, 4], 'Yaw': Btrajectory[:, 5]})
    # dataframe.to_csv("testB.csv", index=False, sep=',')

    print(steps)
