"""
多枚导弹连续发射，无变步长
"""

import numpy as np
from math import cos, sin, tan, pi, atan, atan2, acos, asin
from numpy.linalg import norm

g = 9.81

dt = 0.02
t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180


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


# 导弹模型
class missile_class:
    def __init__(self, pos0_, vel0_, launch_time):
        super(missile_class, self).__init__()
        self.dead = False  # 导弹是否死亡
        self.hit = False  # 导弹是否命中目标
        self.launch_time = launch_time  # 导弹发射时间
        self.pos_ = pos0_.copy()
        self.vel_ = vel0_.copy()
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
        self.stage1_start = 0.3  # s
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
        self.speed_min = 140  # m/s
        # 最大视角
        self.sight_angle_max = pi / 2  # rad
        # 最大跟踪视角速度
        self.sight_angle_rate_max = 0.7  # rad/s
        # 截获距离
        self.detect_range = 20 * 1000  # 20 * 1000  # m todo 计算截获距离
        # 初制导下最大速度倾角
        self.v_theta_of_initial_guidance_max = 45 * pi / 180
        self.t = 0  # 导弹初始计时
        self.t_max = 120  # 最大运行时间
        self.trajectory = np.empty((0, 7))  # 导弹轨迹, 结构为时间、位置（3）、速度（3）

    # 计算命中情况
    def hit_target(self, target):
        # 计算导弹与目标的欧氏距离
        # distance1 = np.linalg.norm(self.pos_ - target_position)
        # return distance1 <= self.kill_range, self.pos_
        killed = False
        # 基于当前点和线性外推得的dt后下一点的判据
        M1 = self.pos_
        T1 = target.pos_
        M2 = self.pos_ + dt * self.vel_
        T2 = target.pos_ + dt * target.vel_
        t1 = 0
        delta_t = dt
        T1T2_ = T2 - T1
        M1M2_ = M2 - M1
        M1T1_ = T1 - M1
        M2T2_ = T2 - M2
        if np.linalg.norm(T1T2_ - M1M2_) < 1e-3:
            # 平行移动了，另一套公式
            d_min = np.linalg.norm(M1T1_)
        else:
            # 至少运动的方向和大小中有一项是不一样的
            t_min = t1 - (np.dot(M1T1_, T1T2_) - np.dot(M1T1_, M1M2_)) / np.linalg.norm(
                T1T2_ - M1M2_) ** 2 * delta_t
            if t_min < t1 or t_min > t1 + delta_t:
                # 没办法用这个结果了，此时最近点要么是开头，要么就是结尾
                d_min = min(np.linalg.norm(M1T1_), np.linalg.norm(M2T2_))
            else:
                # 最近点取得的时间在t1~t2之间
                d_min = np.linalg.norm(M1T1_ + (t_min - t1) / delta_t * (T1T2_ - M1M2_))

        if d_min <= self.kill_range:
            # print(d_min)
            killed = True
        return killed, self.pos_

    # 计算阻力系数
    def Cd(self, mach):
        if 0 < mach <= 0.9:
            cd = 0.16
        if 0.9 < mach <= 1.1:
            cd = 0.16 + 0.29 * (mach - 0.9) / 0.2
        if 1.1 < mach <= 3:
            cd = 0.45 - 0.25 * (mach - 1.1) / 1.9
        else:
            cd = 0.2
        return cd *0.23  # fixme 调整这个系数

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
        return target_information

    # 末制导
    def terminal_guidance(self, v_missile_, v_target_, p_missile_, p_target_):
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
        nyt1 = 4 * np.linalg.norm(vrt_) * q_epsilon_dot / g + cos(theta_mt1)
        nzt1 = 4 * np.linalg.norm(vrt_) * q_beta_dot / g * cos(theta_mt1)  # debug

        # todo 当前的脱锁逻辑是一旦脱锁就转为直线飞行
        #  ，未来将脱锁的计算从导引率部分转移到新写的一个observe方法中，否则导引阶段分不清

        # 导引头脱锁模拟, 速度方向当做导弹头部方向
        # 假设脱锁会导致导弹沿直线继续飞
        # 超出距离脱锁
        if np.linalg.norm(distance) > self.detect_range:
            nzt1 = 0
            nyt1 = cos(theta_mt1)
        # 超出导引头视角导致脱锁
        if np.dot(v_missile_, line_t_) / vmt / distance < cos(self.sight_angle_max):
            nzt1 = 0
            nyt1 = cos(theta_mt1)
        # 超出跟踪角速度导致脱锁
        if np.linalg.norm(np.cross(line_t_, vrt_)) / distance ** 2 > self.sight_angle_rate_max:
            nzt1 = 0
            nyt1 = cos(theta_mt1)

        return 3, [nzt1, nyt1]

    # 制导律选择
    def guidance(self, v_missile_, v_target_, p_missile_, p_target_):
        return self.terminal_guidance(v_missile_, v_target_, p_missile_, p_target_)

    def step(self, target_information, dt=dt, record=True):
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
        case, temp = self.guidance(vmt_, vtt_, pmt_, ptt_)

        # 仅为画图准备的部分start
        line_t_ = ptt_ - pmt_
        distance = np.linalg.norm(line_t_)
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

        # 导弹自毁判定
        if vmt < self.speed_min and self.t > 0.5 + self.stage1_time + self.stage2_time:  # 速度过低自爆
            self.dead = True
        if pmt_[1] < self.minH_m:  # 高度小于限高自爆
            self.dead = True
        if self.t > self.t_max:  # 超时自爆
            self.dead = True

        return vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt


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



if __name__ == '__main__':
    list_missiles = []

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
        R_earth = 6371004  # ???
        dlatit = N / R_earth * 180 / pi
        dlongit = E / (R_earth * cos(latit0 * pi / 180)) * 180 / pi
        dheight = U
        out = np.array([longit0 + dlongit, latit0 + dlatit, height0 + dheight])
        return out


    # 目标运动类,不考虑机动
    class target:
        def __init__(self, pos0_, vel0_):
            super(target, self).__init__()
            self.pos_ = pos0_
            self.vel_ = vel0_

        def step(self, dt=dt):
            self.pos_ += self.vel_ * dt
            self.vel_ = self.vel_
            return self.pos_, self.vel_


    # 调用导弹模型仿真
    t_max = 60 * 2  # 最大仿真时间
    i_list = np.arange(0, int(t_max / dt), 1)
    t_range = np.round(i_list * dt, 2)

    p_carrier_ = np.array([0, 7.5e3, 0])
    v_carrier_ = np.array([300, 0, 0])

    missile_used = 0
    hit = False

    list_pm_ = p_carrier_
    list_vm_ = v_carrier_
    list_vm = np.linalg.norm(list_vm_)
    list_pt_ = np.array([20e3, 6e3, 0])  # 目标
    list_vt_ = np.array([0, 0, -300])
    # missile1 = missile_class(list_pm_, list_vm_, t)
    Target = target(list_pt_, list_vt_)

    list_line_ = list_pt_ - list_pm_
    list_distance = norm(list_pm_ - list_pt_)
    vmt_ = list_vm_
    vmt = list_vm
    pmt_ = list_pm_
    ptt_ = list_pt_
    vtt_ = list_vt_

    # 多枚导弹连续发
    # 遍历导弹列表，如果导弹炸了，则跳过
    # 如果导弹没炸，则获取目标位置、动力学计算以及毁伤判定、记录轨迹
    end_flag = 0
    t_count_start = -5
    tacview = Tacview()
    for i in i_list:
        t = np.round(i * dt, 2)
        # 更新载机位置和速度
        p_carrier_ += v_carrier_ * dt
        # 更新目标位置和速度
        ptt_, vtt_ = Target.step()
        L_t_ = ptt_ - p_carrier_
        distance_of_planes = norm(L_t_)

        off_axis_angle_radian = np.arccos(np.dot(v_carrier_, L_t_) / norm(v_carrier_) / distance_of_planes)

        in_range = 1e3 < distance_of_planes < 20e3 and off_axis_angle_radian < 30 * pi / 180

        # 最快每隔5s发射一枚导弹，一共4枚导弹
        if t - t_count_start >= 10 and in_range and missile_used < 4:
            missile1 = missile_class(p_carrier_, v_carrier_, t)
            list_missiles.append(missile1)
            t_count_start = t  # 重置计时器
            missile_used += 1
            print('missile launched')
        # 对每一枚导弹做判断
        for missile1 in list_missiles:
            if not missile1.dead:
                # 目标位置传给导弹

                target_information = missile1.observe(vmt_, vtt_, pmt_, ptt_)
                # 导弹移动
                vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = missile1.step(
                    target_information)
                vmt = norm(vmt_)

                # 毁伤判定
                # 判断命中情况并终止运行
                if vmt < missile1.speed_min and t > 0.5 + missile1.stage1_time + missile1.stage2_time:
                    missile1.dead = True
                if pmt_[1] < missile1.minH_m:  # 高度小于限高自爆
                    missile1.dead = True
                if missile1.t > missile1.t_max:  # 超时自爆
                    missile1.dead = True
                if t >= 0 + dt:
                    # if distance < missile1.kill_range*50:  # 启动变步长计算命中情况的条件
                    #     pass
                    #     # 假定目标在dt的时间内线性运动，导弹可以以更细分的时间取得观测信息和机动决策

                    hit, point = missile1.hit_target(Target)
                    if hit:
                        print('Target hit')
                        missile1.dead = True
                        missile1.hit = True

                        end_flag = 1

        # 在tacview上显示运动
        send_t = t
        name_R = '001'
        name_B = '002'
        loc_r = ENU2LLH(mark, p_carrier_)
        loc_b = ENU2LLH(mark, ptt_)
        data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n" + \
                       f"#{send_t:.2f}\n{name_B},T={loc_b[0]:.6f}|{loc_b[1]:.6f}|{loc_b[2]:.6f},Name=F16,Color=Blue\n"
        # tacview.send_data_to_client(
        #     "#0.00\nA0100,T=119.99999999999999|59.999999999999986|8902.421354242131|5.124908336161374e-15|2.6380086088911072e-15|92.1278924460462,Name=F16,Color=Red\n")
        # tacview.send_data_to_client(data_to_send)

        # alive_missiles = [missile1 for missile1 in list_missiles if not missile1.dead]
        if not len(list_missiles) == 0:
            for j, missile1 in enumerate(list_missiles):
                if not missile1.dead:
                    loc_m = ENU2LLH(mark, missile1.pos_)
                    data_to_send += f"#{send_t:.2f}\n{10+j},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                    f"Name=AIM-120C,Color=Orange\n"

        tacview.send_data_to_client(data_to_send)
        time.sleep(0.001)

        if end_flag:
            break
