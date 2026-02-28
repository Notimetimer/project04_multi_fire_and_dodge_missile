from math import *
import numpy as np
import socket

# 地理原点
Longitude0 = 144 + 43 / 60
Latitude0 = 13 + 26 / 60
Height0 = 0
mark = np.array([Longitude0, Latitude0, Height0])  # 地理原点

# 已弃用
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

