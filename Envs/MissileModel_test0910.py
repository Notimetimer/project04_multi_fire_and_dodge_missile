"""
这是一个中距弹的模型，然而存在欧拉角死锁问题，且只有末制导
，无变步长
"""

import numpy as np
from math import cos, sin, tan, pi, atan, atan2, acos, asin
from numpy.linalg import norm

g = 9.81
dt = 0.02 * 5
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


from Envs.MissileModel1112 import missile_class, hit_target  # test

if __name__ == '__main__':
    from Envs.tacview_visualize import Tacview

    tacview = Tacview()
    import time

    list_missiles = []
    t = 0
    dt = 0.5  # 0.2  # 0.02 * 5
    # 联动tacview
    # import socket
    # import threading
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

    p_carrier_ = np.array([0, 8e3, 0])
    v_carrier_ = np.array([300, 0, 0])

    missile_used = 0
    hit = False
    plane_missile_time_rate = 8
    list_pm_ = p_carrier_
    list_vm_ = v_carrier_
    list_vm = np.linalg.norm(list_vm_)
    list_pt_ = np.array([30e3, 6e3, 0])  # 目标
    list_vt_ = np.array([-400, 0, -40])
    # missile1 = missile(list_pm_, list_vm_, t)
    Target = target(list_pt_.copy(), list_vt_.copy())

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
    # tacview = tacview()
    for i in i_list:
        t = np.round(i * dt, 2)
        # 更新载机位置和速度
        p_carrier_ += v_carrier_ * dt
        last_pmt_ = p_carrier_
        # 更新目标位置和速度
        last_ptt_ = ptt_
        last_vtt_ = vtt_
        ptt_, vtt_ = Target.step(dt=dt)
        L_t_ = ptt_ - p_carrier_
        distance_of_planes = norm(L_t_)

        off_axis_angle_radian = np.arccos(np.dot(v_carrier_, L_t_) / norm(v_carrier_) / distance_of_planes)

        in_range = 1e3 < distance_of_planes < 30e3 and off_axis_angle_radian < 30 * pi / 180

        # 最快每隔15s发射一枚导弹，一共4枚导弹
        if t - t_count_start >= 15 and in_range and missile_used < 4:
            missile1 = missile_class(p_carrier_.copy(), v_carrier_.copy(), last_ptt_, last_vtt_, t)

            ### 测试导弹跳过加速阶段
            missile1.vel_ = missile1.vel_* 4*340/norm(missile1.vel_)
            missile1.t = 0.5 + missile1.stage1_time + missile1.stage2_time

            list_missiles.append(missile1)
            t_count_start = t  # 重置计时器
            missile_used += 1
            print('missile launched')

        # 对每一枚导弹做判断
        for missile1 in list_missiles:
            if not missile1.dead:
                print('导弹时间：', missile1.t)
                print('导弹雷达状态', missile1.radar_on)
                # 用更小的步长计算导弹的运动
                for j in range(int(plane_missile_time_rate)):
                    # 插值计算目标位置
                    ptt1_ = last_ptt_ + last_vtt_ * dt / plane_missile_time_rate * j
                    # 目标位置传给导弹
                    target_information = missile1.observe(vmt_, last_vtt_, pmt_, ptt1_)
                    # 导弹移动
                    vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = missile1.step(
                        target_information, dt=dt / plane_missile_time_rate)

                    # 毁伤判定
                    # 判断命中情况并终止运行
                    vmt = norm(vmt_)
                    if vmt < missile1.speed_min and t > 0.5 + missile1.stage1_time + missile1.stage2_time:
                        missile1.dead = True
                    if pmt_[1] < missile1.minH_m:  # 高度小于限高自爆
                        missile1.dead = True
                    if missile1.t > missile1.t_max:  # 超时自爆
                        missile1.dead = True
                    if t >= 0 + dt:
                        hit, point_m, point_t = hit_target(pmt_, vmt_, ptt1_, last_vtt_,
                                                           dt=dt / plane_missile_time_rate)
                        if hit:
                            print('Target hit')
                            missile1.dead = True
                            missile1.hit = True

                            pmt_ = point_m
                            ptt_ = point_t
                            missile1.pos_ = pmt_
                            # print(norm(pmt_-ptt_))
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
                if 1:  # not missile1.dead:
                    loc_m = ENU2LLH(mark, missile1.pos_)
                    data_to_send += f"#{send_t:.2f}\n{10 + j},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                    f"Name=AIM-120C,Color=Orange\n"

        tacview.send_data_to_client(data_to_send)
        time.sleep(0.01)

        if end_flag:
            break
        # time.sleep(0.01)
