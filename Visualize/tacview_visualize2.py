import socket
import threading
class Tacview(object):
    def __init__(self):
        # 初始化线程锁与渲染状态，避免在断开后继续发送
        self.client_socket = None
        self.server_socket = None
        self.address = None
        self.rendering = False
        self.lock = threading.Lock()

    def handshake(self):
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

        # 保存 socket，以便后续安全关闭
        with self.lock:
            self.server_socket = server_socket
            self.client_socket = client_socket
            self.address = address
            self.rendering = True

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
        # 如果已调用 end_render 或尚未连接，直接返回，不发送数据
        with self.lock:
            if not getattr(self, "rendering", False):
                return
            sock = self.client_socket

        if sock is None:
            return

        try:
            sock.send(data.encode())
        except Exception:
            # 发送失败时保证安全断开
            try:
                self.end_render()
            except Exception:
                pass

    def end_render(self):
        """停止向 Tacview 发送任何数据并断开连接，程序可继续运行。"""
        with self.lock:
            # 标记停止发送
            self.rendering = False
            # 关闭客户端 socket
            if getattr(self, "client_socket", None) is not None:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self.client_socket.close()
                except Exception:
                    pass
                self.client_socket = None
            # 关闭 listener socket（如果存在）
            if getattr(self, "server_socket", None) is not None:
                try:
                    self.server_socket.close()
                except Exception:
                    pass
                self.server_socket = None
            self.address = None
