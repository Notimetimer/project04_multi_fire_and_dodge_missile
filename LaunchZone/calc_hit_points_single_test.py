'''
遍历机动样式，根据命中的结果数占的比例获取发射概率
'''

import numpy as np
from math import *
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Envs.MissileModel1112 import * # MissileModel1

g = 9.81

dt = 0.02
t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180

from LaunchZone.calc_hit_points_maneuver_from_RWR import *

# 为了可视化，sim_hit函数有所变动
def sim_hit(pm0_, vm0_, pt0_, vt0_, target_move, datalink=1, show=0):
    g = 9.81
    theta_limit = 85 * pi / 180
    t = 0
    # 初始化导弹和目标实例
    missile1 = missile_class(pm0_, vm0_, pt0_, vt0_, t)
    Target = target(pt0_, vt0_)
    t_max = 120  # 假设电池工作120s
    dt_small = 0.08
    dt_big = 0.2
    break_flag = 0
    ptt_ = pt0_.copy()
    pmt_ = pm0_.copy()
    vtt_ = vt0_.copy()
    vmt_ = vm0_.copy()

    while t < t_max:
        distance = norm(pmt_ - ptt_)
        if distance > 5e3:
            dt = dt_big
        else:
            dt = dt_small
        t += dt

        target_information = missile1.observe(vmt_, vtt_, pmt_, ptt_)
        # 导弹移动
        vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = missile1.step(
            target_information, dt=dt, datalink=datalink, record=False)
        vmt = norm(vmt_)

        # 目标移动
        ptt_, vtt_ = Target.step(pmt_, dt=dt, target_move=target_move)

        # 毁伤判定
        # 判断命中情况并终止运行
        if vmt < missile1.speed_min and t > 0.5 + missile1.stage1_time + missile1.stage2_time:
            missile1.dead = True
            break_flag = 1
        if pmt_[1] < missile1.minH_m:  # 高度小于限高自爆
            missile1.dead = True
            break_flag = 1
        if missile1.t > missile1.t_max:  # 超时自爆
            missile1.dead = True
            break_flag = 1
        if norm(missile1.vel_) < missile1.speed_min:  # 低速自爆
            missile1.dead = True
            break_flag = 1
        if t >= 0 + dt:
            hit, point1, point2 = hit_target(missile1.pos_, missile1.vel_, Target.pos_, Target.vel_, dt)
            if hit:
                if show:
                    print('Target hit')
                missile1.dead = True
                missile1.hit = True
                break_flag = 1

        if show:
            # 在tacview上显示运动
            send_t = t
            name_R = '001'
            loc_r = ENU2LLH(mark, ptt_)
            data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
            loc_m = ENU2LLH(mark, missile1.pos_)
            data_to_send += f"#{send_t:.2f}\n{10},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                            f"Name=AIM-120C,Color=Orange\n"
            tacview.send_data_to_client(data_to_send)
            time.sleep(0.01)
        if break_flag: break
    return missile1.hit


if __name__ == '__main__':
    height_init = 11e3
    p_carrier_ = np.array([0, height_init, 0], dtype='float64')
    v_carrier_ = np.array([350, 0, 0], dtype='float64')
    p_target_ = np.array([35e3, height_init, 0e3], dtype='float64')
    v_target_ = np.array([-300, 0, 0], dtype='float64')

    move_pattern = 2

    show = 1

    if show:
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


        # 联动tacview
        import socket
        import threading
        import time

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
            R_earth = 6371004
            dlatit = N / R_earth * 180 / pi
            dlongit = E / (R_earth * cos(latit0 * pi / 180)) * 180 / pi
            dheight = U
            out = np.array([longit0 + dlongit, latit0 + dlatit, height0 + dheight])
            return out


        tacview = Tacview()

    # 变步长解算导弹和目标的运动直到目标被命中或导弹超时/速度过低/错过目标
    # target_move = move_pattern
    datalink = 1
    # show_tacview = show

    hit = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                  show=show)



