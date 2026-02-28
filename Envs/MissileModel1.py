"""
尚未更改使用矢量计算导弹导引率，避免欧拉角死锁
发现observe的调用逻辑、脱锁的效果需要重写
取消根据距离决定步长的逻辑，导弹固定用小步长，飞机固定用大步长

欠缺重构：脱锁的逻辑没有写在observe中，而是在step里面反映在输出结果中
"""

import numpy as np
from math import cos, sin, tan, pi, atan, atan2, acos, asin
from numpy.linalg import norm
import socket
import threading
import time

g = 9.81
plane_missile_time_rate = 8  # 8
dt_refer = 0.8  # 0.2  # 0.02 * 5

g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180



# 提取第一行
def earlist(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[0]  # 直接返回第一行

# 提取最后一行
def latest(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[-1]  # 直接返回最后一行


# 定义大气计算等通用部分
# 大气密度计算
def rho(h):
    # 输入高度(m)
    answer = 1.225 * np.exp(-h / 9300)
    return answer


# 马赫数计算：
def calc_mach(v, height):
    sound_speed = 20.0463 * np.sqrt(288.15 - 0.00651122 * height) if height <= 11000 else 295.069
    return v / sound_speed, sound_speed

# 计算命中情况
def hit_target(pmt_1, vmt_1, ptt_1, vtt_1, dt=dt_refer, kill_range=20):
    # # 计算点与点距离
    # distance1 = np.linalg.norm(pmt_ - ptt_)
    # return distance1 <= kill_range, pmt_

    # 解析计算
    killed = False
    # 基于当前点和线性外推得的dt后下一点的判据
    M1 = pmt_1
    T1 = ptt_1
    M2 = pmt_1 + dt * vmt_1
    T2 = ptt_1 + dt * vtt_1

    t1 = 0
    delta_t = dt
    T1T2_ = T2 - T1
    M1M2_ = M2 - M1
    M1T1_ = T1 - M1
    M2T2_ = T2 - M2
    if np.linalg.norm(T1T2_ - M1M2_) < 1e-3:
        # 平行移动了，另一套公式
        d_min = np.linalg.norm(M1T1_)
        posm_ = M1
        post_ = T1
    else:
        # 至少运动的方向和大小中有一项是不一样的
        t_min = t1 - (np.dot(M1T1_, T1T2_) - np.dot(M1T1_, M1M2_)) / np.linalg.norm(
            T1T2_ - M1M2_) ** 2 * delta_t
        if t_min < t1:
            t_min = t1
        elif t_min > t1 + delta_t:
            t_min = t1 + delta_t
        posm_ = M1 + (t_min - t1) / delta_t * (M2 - M1)
        post_ = T1 + (t_min - t1) / delta_t * (T2 - T1)
        d_min = np.linalg.norm(M1T1_ + (t_min - t1) / delta_t * (T1T2_ - M1M2_))
        # if t_min < t1 or t_min > t1 + delta_t:
        #     # 没办法用这个结果了，此时最近点要么是开头，要么就是结尾
        #     d_min = min(np.linalg.norm(M1T1_), np.linalg.norm(M2T2_))
        # else:
        #     # 最近点取得的时间在t1~t2之间
        #     d_min = np.linalg.norm(M1T1_ + (t_min - t1) / delta_t * (T1T2_ - M1M2_))
        # posm_=pmt_
    if d_min <= kill_range:
        # print(d_min)
        killed = True
    return killed, posm_, post_

# 导弹模型
class missile_class:
    def __init__(self, pos0_, vel0_, pt0_, vt0_, launch_time=0):
        super(missile_class, self).__init__()
        self.dead = False  # 导弹是否死亡
        self.hit = False  # 导弹是否命中目标
        self.launch_time = launch_time  # 导弹发射时间
        self.lock_time = None
        self.pos_ = pos0_.copy()
        self.vel_ = vel0_.copy()
        self.pt0_ = pt0_.copy()
        self.vt0_ = vt0_.copy()
        self.latest_time_of_target = launch_time  # 上一次观测到目标的时间 todo 没有考虑载机转走的情况

        # 最高和最低高度
        self.maxH_m = 25000
        self.minH_m = 500
        # 重量参数和燃烧时间参数
        self.empty_weight = 96.82  # kg
        self.stage1_weight = 20.28  # kg
        self.stage2_weight = 44.9  # kg
        self.stage1_time = 2.3  # s
        self.stage2_time = 11  # s
        self.stage1_burn_rate = self.stage1_weight / self.stage1_time  # 一级燃烧率kg/s
        self.stage2_burn_rate = self.stage2_weight / self.stage2_time  # 二级燃烧率kg/s
        self.stage1_thrust = 20393  # N
        self.stage2_thrust = 9360.5  # N
        self.stage1_start = 0.5  # s
        self.stage2_start = self.stage1_start + self.stage1_time  # s
        self.stage2_end = self.stage1_start + self.stage1_time + self.stage2_time  # s
        # 杀伤半径
        self.kill_range = 20
        # 最大过载
        self.max_g = 40
        # 最大马赫数
        self.max_mach = 4
        # 特征面积
        self.area = 0.4  # m2
        # 阻力系数是一个函数，不在这里定义
        # 最小速度
        self.speed_min = 0.65 * 340  # m/s
        # 最大视角
        self.sight_angle_max = pi / 2  # rad
        # 最大跟踪视角速度
        self.sight_angle_rate_max = 0.7  # rad/s
        # 截获距离
        self.detect_range = 20e3  # 20e3  # m todo 计算截获距离
        # 初制导下最大速度倾角
        self.v_theta_of_initial_guidance_max = 45 * pi / 180
        self.t = 0  # 导弹初始计时
        self.t_max = 120  # 最大运行时间
        self.trajectory = np.empty((0, 7))  # 导弹轨迹, 结构为时间、位置（3）、速度（3）
        self.guidance_stage = 2  # 2为中制导，3为末制导
        self.nzt = 0  # 过载量记忆值
        self.nyt = 0
        self.last_target_pos = None
        self.last_target_t = None
        self.last_target_v = None
        self.radar_on = False
        self.radar_lock_state = False
        self.side = None
        self.datalink = None

    def Cd(self, mach):
        if 0 < mach <= 0.9:
            cd = 0.16
        if 0.9 < mach <= 1.1:
            cd = 0.16 + 0.29 * (mach - 0.9) / 0.2
        if 1.1 < mach <= 3:
            cd = 0.45 - 0.25 * (mach - 1.1) / 1.9
        else:
            cd = 0.2
        return cd * 0.23  # 调整这个系数

    # 计算推力和质量
    def calc_burn(self):
        # 一级点火
        if 0.5 < self.t <= 0.5 + self.stage1_time:
            Fp = self.stage1_thrust
            m_missile = self.empty_weight + self.stage2_weight + \
                        self.stage1_weight * (1 - (self.t - 0.5) / self.stage1_time)
        # 二级点火
        elif 0.5 + self.stage1_time < self.t <= 0.5 + self.stage1_time + self.stage2_time:
            Fp = self.stage2_thrust
            m_missile = self.empty_weight + \
                        self.stage2_weight * (1 - (self.t - 0.5 - self.stage1_time) / self.stage2_time)
        # 滑翔段
        else:
            Fp = 0
            m_missile = self.empty_weight
        return Fp, m_missile

    def observe(self, v_missile_, v_target_, p_missile_, p_target_):
        # todo 产生target_information，包含visable，为0时后续所有都是0,在step中引用无目标时的导引率（尚未写的类方法），否则为1时正常调用制导律
        # 当前写法后面还要改
        target_information = np.zeros(7)
        target_information[0] = 1
        target_information[1:4] = p_target_
        target_information[4:7] = v_target_
        # 输入参数
        vmt_ = v_missile_
        ptt_ = p_target_
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = v_target_ - v_missile_
        distance = np.linalg.norm(line_t_)
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)

        # # 超出距离、跟踪角度或是跟踪角速度脱锁
        # if np.linalg.norm(distance) > self.detect_range or \
        #         np.dot(v_missile_, line_t_) / vmt / distance < cos(self.sight_angle_max) or \
        #         np.linalg.norm(np.cross(line_t_, vrt_)) / distance ** 2 > self.sight_angle_rate_max:
        #     target_information = np.zeros(7)

        return target_information

    # 中制导
    def mid_term_guidance(self, v_missile_, v_target_, p_missile_, p_target_, datalink=True):
        # print("中制导")
        if not datalink:
            self.radar_on = True
            if not self.last_target_t:
                vtt_predict = self.vt0_
                ptt_predict = self.pt0_ + self.vt0_ * (self.t - self.latest_time_of_target)
            else:
                vtt_predict = self.last_target_v
                ptt_predict = self.last_target_pos + vtt_predict*(self.t - self.last_target_t)
        else:
            self.radar_on = False
            vtt_predict = v_target_
            ptt_predict = p_target_
            self.last_target_pos = ptt_predict.copy()
            self.last_target_v = vtt_predict.copy()
            self.last_target_t = self.t

        # 输入参数
        vmt_ = v_missile_
        ptt_ = ptt_predict
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = vtt_predict - v_missile_
        distance = np.linalg.norm(line_t_)
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)
        vrx, vry, vrz = vrt_
        # 制导指令算法
        omega_LOS_y = (line_t_[0] * vrz - vrx * line_t_[2]) / (line_t_[0] ** 2 + line_t_[2] ** 2)  # 视线偏转角速度
        q_beta_dot = omega_LOS_y
        omega_LOS_z = (vry * (line_t_[0] ** 2 + line_t_[2] ** 2) - line_t_[1] * (
                line_t_[0] * vrx + line_t_[2] * vrz)) / (
                              distance ** 2 * distance_hor)  # 视线俯仰角速度
        q_epsilon_dot = omega_LOS_z
        nyt1 = 4 * max(vmt, np.linalg.norm(vrt_)) * q_epsilon_dot / g + cos(theta_mt1)  # test
        nzt1 = 4 * max(vmt, np.linalg.norm(vrt_)) * q_beta_dot / g * cos(theta_mt1)

        return 2, [nzt1, nyt1]

    # 末制导
    def terminal_guidance(self, v_missile_, v_target_, p_missile_, p_target_):
        self.radar_on = True
        # 输入参数
        vmt_ = v_missile_
        ptt_ = p_target_
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = v_target_ - v_missile_
        distance = np.linalg.norm(line_t_)
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)

        vrx, vry, vrz = vrt_
        # 制导指令算法
        omega_LOS_y = (line_t_[0] * vrz - vrx * line_t_[2]) / (line_t_[0] ** 2 + line_t_[2] ** 2)  # 视线偏转角速度
        q_beta_dot = omega_LOS_y
        omega_LOS_z = (vry * (line_t_[0] ** 2 + line_t_[2] ** 2) - line_t_[1] * (
                line_t_[0] * vrx + line_t_[2] * vrz)) / (
                              distance ** 2 * distance_hor)  # 视线俯仰角速度
        q_epsilon_dot = omega_LOS_z
        nyt1 = 4 * max(vmt, np.linalg.norm(vrt_)) * q_epsilon_dot / g + cos(theta_mt1)
        nzt1 = 4 * max(vmt, np.linalg.norm(vrt_)) * q_beta_dot / g * cos(theta_mt1)  # debug

        # 导引头脱锁模拟, 速度方向当做导弹头部方向
        off_lock = False
        # 假设脱锁会导致导弹沿直线继续飞
        # # 超出距离脱锁
        # if np.linalg.norm(distance) > self.detect_range:
        #     off_lock = True
        # 超出导引头视角导致脱锁
        if np.dot(v_missile_, line_t_) / vmt / distance < cos(self.sight_angle_max):
            off_lock = True
        # 超出跟踪角速度导致脱锁
        if np.linalg.norm(np.cross(line_t_, vrt_)) / distance ** 2 > self.sight_angle_rate_max:
            off_lock = True
        if off_lock:
            nzt1 = self.nzt  # 0
            nyt1 = self.nyt  # cos(theta_mt1)
        else:
            self.nzt = nzt1
            self.nyt = nyt1

        return 3, [nzt1, nyt1]

    # 制导律选择
    def guidance(self, v_missile_, v_target_, p_missile_, p_target_, datalink):
        ptt_ = p_target_
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        distance = np.linalg.norm(line_t_)
        if np.linalg.norm(distance) <= self.detect_range:
            # print(str(self.t)+'s,末制导')
            self.guidance_stage = 3

        if self.guidance_stage == 2:
            return self.mid_term_guidance(v_missile_, v_target_, p_missile_, p_target_, datalink)
        if self.guidance_stage == 3:
            if not self.lock_time:
                self.lock_time = self.t
            return self.terminal_guidance(v_missile_, v_target_, p_missile_, p_target_)
        # return self.terminal_guidance(v_missile_, v_target_, p_missile_, p_target_)

    def step(self, target_information, dt=dt_refer, datalink=True, record=False):
        '''
        输入结构：是否看到目标(1)，目标的位置(3)、目标的速度(3)
        '''
        # 根据目标信息产生制导指令
        visable = target_information[0]
        # if visable==0:
        #     nzt=
        ptt_ = target_information[1:4]
        vtt_ = target_information[4:7]
        pmt_ = self.pos_
        vmt_ = self.vel_

        # test
        line_t_ = ptt_ - pmt_
        distance = np.linalg.norm(line_t_)
        # print('导弹感知到的距离:' + str(distance))

        case, temp = self.guidance(vmt_, vtt_, pmt_, ptt_, datalink=datalink)

        # 仅为画图准备的部分start


        vmt = np.linalg.norm(vmt_)
        vtt = np.linalg.norm(vtt_)
        psi_mt = np.arctan2(vmt_[2], vmt_[0])
        psi_tt = np.arctan2(vtt_[2], vtt_[0])
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        vt_hor = np.linalg.norm([vtt_[2], vtt_[0]])  # 目标水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        q_beta_t = np.arctan2(line_t_[2], line_t_[0])  # 目标线偏角
        q_epsilon_t = np.arctan2(line_t_[1], distance_hor)  # 目标线倾角
        theta_mt = np.arctan2(vmt_[1], vm_hor)
        theta_tt = np.arctan2(vtt_[1], vt_hor)
        theta = atan2(vmt_[1], np.linalg.norm([vmt_[0], vmt_[2]]))  # 速度倾角
        psi = atan2(vmt_[2], vmt_[0])  # 速度航向角
        # 仅为画图准备的部分end

        # 继续处理制导指令
        if case == 1:
            theta_mt = temp[0]
            psi_mt = temp[1]
            nzt = 0  # 没法给
            nyt = 0

        else:
            nzt = temp[0]
            nyt = temp[1]

        # 根据制导指令更新动力学
        Fp, m_missile1 = self.calc_burn()
        # 导弹马赫数
        mach, sound_speed = calc_mach(vmt, pmt_[1])  # 速率，高度
        # 导弹阻力计算
        cd = self.Cd(mach)
        # 空气密度
        Rho = rho(pmt_[1])
        # 空气阻力
        Fx = 1 / 2 * Rho * vmt ** 2 * self.area * cd
        # 速率更新
        v_dot = (Fp - Fx) / m_missile1 - g * sin(theta_mt)
        vmt += v_dot * dt
        # 限马赫数
        vmt = min(vmt, self.max_mach * sound_speed)
        # 过载限制2
        nt = np.clip(np.linalg.norm([nyt, nzt]), 0, 40)
        [nzt, nyt] = np.array([nzt, nyt]) * nt / np.sqrt(nyt ** 2 + nzt ** 2) if np.abs(nt) > 0 else [0.0, 0.0]
        theta_mt += ((nyt - cos(theta_mt)) * g / vmt) * dt
        psi_mt += nzt * g / vmt / cos(theta_mt) * dt
        # 欧拉角反奇异
        theta_mt = np.clip(theta_mt, -theta_limit, theta_limit)
        if psi_mt > pi:
            psi_mt -= 2 * pi
        if psi_mt < -pi:
            psi_mt += 2 * pi
        vmt_ = vmt * np.array([cos(theta_mt) * cos(psi_mt), sin(theta_mt), cos(theta_mt) * sin(psi_mt)])
        self.vel_ = vmt_

        # print(vmt_)
        # print(self.pos_)

        self.pos_ += vmt_ * dt  # 欧拉积分
        # 更新时间
        self.t += dt
        self.t = round(self.t, 2)  # 保留两位小数
        # print(self.t)
        # 记录运行轨迹
        if record:
            traj_add = np.array(
                [self.t + self.launch_time, self.pos_[0], self.pos_[1], self.pos_[2], self.vel_[0], self.vel_[1],
                 self.vel_[2]])
            self.trajectory = np.vstack((self.trajectory, traj_add.copy()))  # 记录当前位置
        return vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt


class tacview(object):
    def __init__(self):
        host = "localhost"
        port = 42674
        # host = input("请输入服务器IP地址：")
        # port = int(input("请输入服务器端口："))
        # 提示用户打开tacview软件高级版，点击“记录”-“实时遥测”
        print("请打开tacview软件高级版，点击“记录”-“实时遥测”，并使用以下设置：")
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

