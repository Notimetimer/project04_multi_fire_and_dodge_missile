import argparse
import time
from Envs.battle3dof1v1_missile import *  # battle3dof1v1_proportion
from math import pi
import numpy as np
import matplotlib
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

'''本文件用于检测奖励函数是否正确，双方采用离散的贪婪策略或是固定航线，进行一个回合的仿真后结束并显示路径'''

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
dt_refer = dt

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=8 * 60,  # * 60.0,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
    # parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
    args = parser.parse_args()
    return args


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

# def main():
# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))

args = get_args()
env = Battle(args)
env.reset()
r_obs_spaces, b_obs_spaces = env.r_obs_spaces, env.b_obs_spaces
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
'''滚动时域优化开始'''

r_action = []
b_action = []


def strong_rolling_optim(env, side='r'):
    # 7种离散动作：
    action_list = [
        [[0, 0, 0]],
        [[1, 0, 0]],
        [[-1, 0, 0]],
        [[0, 1, 0]],
        [[0, 1, 1]],
        [[0, 0, 0.4]],
        [[0, 0, -0.4]]
    ]
    reward_list = []
    # 遍历7种动作并记录对应的奖励信息
    for i in range(len(action_list)):
        if side == 'r':
            r_action_n = action_list[i]
            b_action_n = [[0, 0, 0]]  # 假设对面没有动作
        elif side == 'b':
            b_action_n = action_list[i]
            r_action_n = [[0, 0, 0]]  # 假设对面没有动作
        else:
            raise RuntimeError("请输入正确的阵营")
        # 执行动作并记录奖励信息
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=True)
        if side == 'r':
            reward_list.append(r_reward_n[0])  # 获取当前动作状态奖励信息
        if side == 'b':
            reward_list.append(b_reward_n[0])  # 获取当前动作状态奖励信息
    chosen_action = reward_list.index(max(reward_list))
    global r_action, b_action
    if side == 'r':
        r_action.append(chosen_action)
    if side == 'b':
        b_action.append(chosen_action)
    return action_list[chosen_action]
    # return action_list[0]


def weak_rolling_optim(env, side='r'):
    # 7种离散动作：
    action_list = [
        [[0, 0, 0]],
        [[0.2, 0, 0]],
        [[-0.2, 0, 0]],
        [[0, 0.2, 0]],
        [[0, 0.2, 1]],
        [[0, 0, 0.1]],
        [[0, 0, -0.1]]
    ]
    reward_list = []
    # 遍历7种动作并记录对应的奖励信息
    for i in range(len(action_list)):
        if side == 'r':
            r_action_n = action_list[i]
            b_action_n = [[0, 0, 0]]  # 假设对面没有动作
        elif side == 'b':
            b_action_n = action_list[i]
            r_action_n = [[0, 0, 0]]  # 假设对面没有动作
        else:
            raise RuntimeError("请输入正确的阵营")
        # 执行动作并记录奖励信息
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=True)
        if side == 'r':
            reward_list.append(r_reward_n[0])  # 获取当前动作状态奖励信息
        if side == 'b':
            reward_list.append(b_reward_n[0])  # 获取当前动作状态奖励信息
    chosen_action = reward_list.index(max(reward_list))
    global r_action, b_action
    if side == 'r':
        r_action.append(chosen_action)
    if side == 'b':
        b_action.append(chosen_action)
    return action_list[chosen_action]
    # return action_list[0]
start_time = time.time()

# train_off_policy_agent
# return_list = []

for i in range(1):
    episode_return = 0
    env.reset()
    a1 = env.BUAV.pos_  # 58000,7750,20000
    a2 = env.RUAV.pos_  # 2000,7750,20000
    b1 = env.UAVs[0].pos_
    b2 = env.UAVs[1].pos_
    done = False
    r_action_list = []
    b_action_list = []
    Rtrajectory = []
    Btrajectory = []

    # 环境运行一轮的情况
    for count in range(round(args.max_episode_len / dt_refer)):
        # print(f"time: {env.t}")  # 打印当前的 count 值
        # 回合结束判断
        # print(env.running)
        if env.running == False or count == round(args.max_episode_len / dt_refer) - 1:
            # print('回合结束，时间为：', env.t, 's')
            break
        # 获取观测信息
        r_obs_n, b_obs_n = env.get_obs()
        state = np.squeeze(r_obs_n)
        # 执行动作
        # b_action_n = [[-1, 0, 0]]  # for i in range(args.num_BUAVs)]
        b_action_n = strong_rolling_optim(env, side='b')  # 短视优化
        # 红方使用DDPG执行动作得到环境反馈
        # r_action_n = [[1,0,0]]
        r_action_n = strong_rolling_optim(env, side='r')  # 短视优化
        # print('b_action_n=',b_action_n)
        # print('r_action_n=',r_action_n)
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=False,
                                                                       record=True)  # 2、环境更新并反馈

        '''显示运行轨迹'''
        print("红方位置：", env.RUAV.pos_)
        print("蓝方位置：", env.BUAV.pos_)
        # 如果导弹已发射
        print(f"当前发射的导弹数量：{len(env.Rmissiles) + len(env.Bmissiles)}")
        # 遍历导弹列表
        for missile in env.Rmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            print("红方导弹位置：", missile_pos)
        for missile in env.Bmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            print("蓝方导弹位置：", missile_pos)

        send_t = env.t
        name_R = env.RUAV.id
        name_B = env.BUAV.id
        loc_r = ENU2LLH(mark, env.RUAV.pos_)
        loc_b = ENU2LLH(mark, env.BUAV.pos_)
        data_to_send = ''
        if not env.RUAV.dead:
            data_to_send+=f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
        else:
            data_to_send += f"#{send_t:.2f}\n-{name_R}\n"
        if not env.BUAV.dead:
            data_to_send += f"#{send_t:.2f}\n{name_B},T={loc_b[0]:.6f}|{loc_b[1]:.6f}|{loc_b[2]:.6f},Name=F16,Color=Blue\n"
        else:
            data_to_send += f"#{send_t:.2f}\n-{name_B}\n"

            # data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n" + \
            #            f"#{send_t:.2f}\n{name_B},T={loc_b[0]:.6f}|{loc_b[1]:.6f}|{loc_b[2]:.6f},Name=F16,Color=Blue\n"
        # tacview.send_data_to_client(
        #     "#0.00\nA0100,T=119.99999999999999|59.999999999999986|8902.421354242131|5.124908336161374e-15|2.6380086088911072e-15|92.1278924460462,Name=F16,Color=Red\n")
        # tacview.send_data_to_client(data_to_send)

        # alive_missiles = [missile1 for missile1 in list_missiles if not missile1.dead]
        for j, missile in enumerate(env.missiles):
            if hasattr(missile, 'dead') and missile.dead:
                data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
            else:
                # 记录导弹的位置
                loc_m = ENU2LLH(mark, missile.pos_)
                if missile.side == 'red':
                    color = 'Orange'
                else:
                    color = 'Green'
                data_to_send += f"#{send_t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                f"Name=AIM-120C,Color={color}\n"

        print("data_to_send", data_to_send)
        tacview.send_data_to_client(data_to_send)
        time.sleep(0.01)
        # 红
        position = env.RUAV.pos_
        angle = np.array([env.RUAV.gamma, env.RUAV.theta, env.RUAV.psi]) * 180 / pi
        Rtrajectory.append(np.hstack((position, angle)))
        # 蓝
        position = env.BUAV.pos_
        angle = np.array([env.BUAV.gamma, env.BUAV.theta, env.BUAV.psi]) * 180 / pi
        Btrajectory.append(np.hstack((position, angle)))


        # if env.RUAV.missiles:
        #     for missile in env.RUAV.missiles[:]:  # 处理导弹爆炸
        #         if missile.dead:
        #             env.RUAV.missiles.remove(missile)

        if terminate == True:
            break

# 补充显示

loc_o = ENU2LLH(mark, np.zeros(3))
data_to_send = ''
data_to_send = f"#{send_t+dt:.2f}\n{900},T={loc_o[0]:.6f}|{loc_o[1]:.6f}|{loc_o[2]:.6f},Name=Game Over, Color=Black\n"
print("data_to_send", data_to_send)
tacview.send_data_to_client(data_to_send)

data_to_send = f"#{send_t+dt*10:.2f}\n{900},T={loc_o[0]:.6f}|{loc_o[1]:.6f}|{loc_o[2]:.6f},Name=Game Over, Color=Black\n"
print("data_to_send", data_to_send)
tacview.send_data_to_client(data_to_send)

end_time = time.time()  # 记录结束时间
print(f"程序运行时间: {end_time - start_time} 秒")

# red_pos_list = env.UAVs[0].trajectory
# blue_pos_list = env.UAVs[1].trajectory
#
# # 数据可视化准备
# red_p_show_show = np.array(red_pos_list).T
# blue_p_show_show = np.array(blue_pos_list).T
#
# r_action_list = np.squeeze(np.array(r_action_list))
# b_action_list = np.squeeze(np.array(b_action_list))
#
#
# # 可视化
# plt.figure(2)
# from show_trajectory import show_trajectory
# show_trajectory(red_pos_list, blue_pos_list, min_east, min_north, min_height, max_east, max_north, max_height, r_show=1, b_show=1)
#
# # 保存数据
# Rtrajectory = np.array(Rtrajectory)
# Btrajectory = np.array(Btrajectory)
# # dataframe = pd.DataFrame({'N': red_pos_list[:, 0], 'U': red_pos_list[:, 1], 'E': red_pos_list[:, 2]},'Roll', )
# dataframe = pd.DataFrame({'N': Rtrajectory[:, 0], 'U': Rtrajectory[:, 1], 'E': Rtrajectory[:, 2],
#                           'Roll': Rtrajectory[:, 3], 'Pitch': Rtrajectory[:, 4], 'Yaw': Rtrajectory[:, 5]})
# dataframe.to_csv("test.csv", index=False, sep=',')
