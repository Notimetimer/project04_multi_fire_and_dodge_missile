from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from math import *
from typing import Literal
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Controller.baseline_actor import BaselineActor
import socket
import threading
import time
import jsbsim
import matplotlib.pyplot as plt
import math
import numpy as np

def get_observation(obs_inter):
    obs = np.array(obs_inter, dtype=np.float32)
    norm_obs = np.zeros(12)
    norm_obs[0] = obs[0] / 1000                  #  0. ego delta altitude  (unit: 1km)
    norm_obs[1] = (obs[1])                       #  1. ego delta heading   (unit rad)
    norm_obs[2] = obs[2]  / 340                  #  2. ego delta velocities_u  (unit: mh)
    norm_obs[3] = obs[3] / 5000                  #  3. ego_altitude (unit: km)
    norm_obs[4] = np.sin(obs[4])                 #  4. ego_roll_sin
    norm_obs[5] = np.cos(obs[4])                 #  5. ego_roll_cos
    norm_obs[6] = np.sin(obs[5])                 #  6. ego_pitch_sin
    norm_obs[7] = np.cos(obs[5])                 #  7. ego_pitch_cos
    norm_obs[8] = obs[6] / 340                   #  8. ego_v_x   (unit: m/s)
    norm_obs[9] = obs[7] / 340                   #  9. ego_v_y   (unit: m/s)
    norm_obs[10] = obs[8] / 340                  #  10. ego_v_z  (unit: m/s)
    norm_obs[11] = obs[9] / 340                  #  11. ego_vc   (unit: m/s)
    norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
    return norm_obs

def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff

def F16control(obs_jsbsim, rnn_states, hist_act):
    actor = BaselineActor(use_mlp_actlayer=True) # F16baseline_model用False，actor_latest用True
    model_root = os.path.dirname(os.path.abspath(__file__))
    dir2 = os.path.join(model_root, 'actor_latest.pt')
    model_path = dir2  # 'actor_latest.pt' # 'actor_latest.pt' 'F16baseline_model.pt'
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    actor.eval()
    rnn_states = np.zeros((1, 1, 128))
    norm_act = np.zeros(4)  # action: [delta_altitude, delta_heading, delta_velocities_u, vc]

    # 对称弥补左转控制不好的问题
    delta_heading = obs_jsbsim[1]
    if delta_heading < 0:
        obs_jsbsim[1] *= -1
        obs_jsbsim[4] *= -1
        obs_jsbsim[7] *= -1

    obs = get_observation(obs_jsbsim)
    norm_act = np.zeros(4)  # action input: [delta_altitude, delta_heading, delta_velocities_u, vc]
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action, rnn_states = actor(obs_tensor, rnn_states)
        norm_act[0] = action[0, 0] / 20 - 1.
        norm_act[1] = action[0, 1] / 20 - 1.
        norm_act[2] = action[0, 2] / 20 - 1.
        norm_act[3] = action[0, 3] / 58 + 0.4

    # 对称弥补左转控制不好的问题
    if delta_heading < 0:
        norm_act[0] *= -1
        norm_act[2] *= -1

    # 控制量平滑
    # norm_act = hist_act * 0.5 + norm_act * 0.5

    hist_act = norm_act

    return norm_act, rnn_states, hist_act

if __name__=='__main__':
    dt = 0.02 # 0.05  # 0.02
    # 连续输出并tacview中可视化
    start_time = time.time()
    target_height = 3000 * 3.2808  # 6000 m to ft
    target_heading = 0 * pi/180  # 度 to rad
    target_velocity = 300 * 1.9438  # 200 m/s to knots
    t_last = 120.0
    
    tacview_show = 1  # 是否显示Tacview

    if tacview_show:
        print('please prepare tacview')
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Visualize.tacview_visualize import Tacview
        tacview = Tacview()

    # 启动 JSBSim
    sim = jsbsim.FGFDMExec(None, None)
    sim.set_debug_level(0)
    sim.set_dt(dt)  # 解算步长 dt 秒

    # 设置模型路径（一般 JSBSim pip 包自动包含）
    sim.load_model("f16") # f15, p51d, ball 等模型可选

    # 设置初始状态（单位：英尺、节、角度）
    sim[ "ic/h-sl-ft"    ] = 5000 * 3.2808         # 高度：m -> ft
    sim[ "ic/vt-kts"     ] = 200 * 1.9438          # 空速： m/s-> 节
    sim[ "ic/psi-true-deg" ] = 0             # 航向角: °
    sim[ "ic/phi-deg"    ] = 0
    sim[ "ic/theta-deg"  ] = 0
    sim[ "ic/alpha-deg"  ] = 0
    sim[ "ic/beta-deg"   ] = 0
    sim["ic/long-gc-deg"] = -118   # 设置初始经度（单位：度）
    sim["ic/lat-gc-deg"] = 34     # 设置初始纬度（单位：度）


    # 初始化状态
    sim.run_ic()
    sim.set_property_value('propulsion/set-running', -1)

    rnn_states = np.zeros((1, 1, 128))

    # 记录轨迹和状态数据
    positions = []
    attitudes = []
    velocities = []
    thrust_data = []  # 添加推力数据记录
    time_steps = []

    # 记录控制量
    aileron_cmd = []
    elevator_cmd = []
    rudder_cmd = []
    throttle_cmd = []

    hist_act=np.array([0,0,0,1])
    for step in range(int(t_last / dt)):
        sim.run()
        current_time = step * dt
        time_steps.append(current_time)

        delta_height = (target_height - sim["position/h-sl-ft"]) / 3.2808
        delta_heading = sub_of_radian(target_heading, sim["attitude/psi-deg"] * pi / 180)
        # delta_heading = 19 * pi / 180 # test
        delta_speed = (target_velocity - sim["velocities/vt-fps"] * 0.3048) / 1.9438

        # 构建观测向量
        obs_jsbsim = [
            delta_height,  # ego delta altitude (unit: m)
            delta_heading,  # ego delta heading (unit rad)
            delta_speed,  # ego delta velocities_u (unit: m/s)
            sim["position/h-sl-ft"] / 3.2808,  # ego_altitude (unit: m)
            sim["attitude/phi-deg"] * pi / 180,  # ego_roll (unit: rad)
            sim["attitude/theta-deg"] * pi / 180,  # ego_pitch (unit: rad)
            sim["velocities/u-fps"] * 0.3048,  # ego_body_v_x (unit: f/s -> m/s)
            sim["velocities/v-fps"] * 0.3048,  # ego_body_v_y (unit: f/s -> m/s)
            sim["velocities/w-fps"] * 0.3048,  # ego_body_v_z (unit: f/s -> m/s)
            sim["velocities/vt-fps"] * 0.3048  # ego_vc (unit: m/s)
        ]

        # norm_act由F16control函数输出
        norm_act, rnn_states, hist_act = F16control(obs_jsbsim, rnn_states, hist_act)
        # 取姿态角度
        phi = sim["attitude/phi-deg"]      # 滚转角 (roll)
        theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
        psi = sim["attitude/psi-deg"]      # 航向角 (yaw)
        alpha = sim["aero/alpha-deg"]      # 迎角
        beta = sim["aero/beta-deg"]        # 侧滑角
        
        # 副翼增加角速度阻尼
        # ...existing code...
        roll_rate = sim["velocities/p-rad_sec"] # 角速度
        pitch_rate = sim["velocities/q-rad_sec"] # 角速度
        yaw_rate = sim["velocities/r-rad_sec"] # 角速度

        # norm_act[0] -= roll_rate * 0.2/(pi/3) # 滚转控制加阻尼, 符号可能反了
        # norm_act[1] += pitch_rate * 0.2/(pi/3) # 俯仰控制加阻尼
        # norm_act[2] -= yaw_rate * 0.1/(pi/3) # 偏航控制加阻尼, 符号可能反了

        # 过载量限制
        nz_g = sim["accelerations/Nz"]  # 垂直过载
        ny_g = sim["accelerations/Ny"]  # 侧向过载
        nx_g = sim["accelerations/Nx"]  # 纵向过载
        # print(nz_g)
        norm_act[1] += nz_g * 0.2/8 # 俯仰控制加阻尼

        if np.linalg.norm(delta_heading)<20*pi/180:
            norm_act[0]= - 1 * (phi-delta_heading/pi*9 * 5) -0.5*roll_rate/(pi/3)


        sim["fcs/aileron-cmd-norm"], \
            sim["fcs/elevator-cmd-norm"],  \
                sim["fcs/rudder-cmd-norm"], \
                    sim["fcs/throttle-cmd-norm"] = norm_act  # 设置控制量
        

        
        # 记录控制量
        aileron_cmd.append(sim["fcs/aileron-cmd-norm"])
        elevator_cmd.append(sim["fcs/elevator-cmd-norm"])
        rudder_cmd.append(sim["fcs/rudder-cmd-norm"])
        throttle_cmd.append(sim["fcs/throttle-cmd-norm"])

        # 取当前位置
        lon = sim["position/long-gc-deg"]  # 经度
        lat = sim["position/lat-gc-deg"]   # 纬度
        alt = sim["position/h-sl-ft"] * 0.3048  # 高度（英尺转米）
        
        # 简单的相对位置计算
        if step == 0:
            start_lon, start_lat = lon, lat
        
        x = (lon - start_lon) * 111320  # 经度差转米（近似）
        y = (lat - start_lat) * 110540  # 纬度差转米（近似）
        z = alt
        positions.append((x, y, z))
        
        attitudes.append((phi, theta, psi, alpha, beta))
        
        # sim.run()

        # 取速度分量
        u = sim["velocities/u-fps"] * 0.3048  # X轴速度 (fps转m/s)
        v = sim["velocities/v-fps"] * 0.3048  # Y轴速度 (fps转m/s)
        w = sim["velocities/w-fps"] * 0.3048  # Z轴速度 (fps转m/s)
        velocities.append((u, v, w))
        
        # 记录推力和发动机参数
        try:
            thrust=sim.get_property_value('propulsion/engine/thrust-lbs')
            fuel_flow = sim["propulsion/engine/fuel-flow-rate-pps"]  # 燃油流量
            total_speed = sim["velocities/vt-fps"] * 0.3048  # 总速度 (m/s)
            thrust_data.append((thrust, fuel_flow, total_speed))

            # # 打印关键参数
            # if step % np.round(1/dt) == 0:  # 每10秒打印一次
            #     print(f"Time: {current_time:.1f}s, Throttle: {sim['fcs/throttle-cmd-norm']:.1f}, "
            #         f"Thrust: {thrust:.0f} lbs, Speed: {total_speed:.1f} m/s")
        except:
            thrust_data.append((0, 0, 0))

        # 通过tacview可视化
        if tacview_show and step % np.round(1/dt) == 0:
            send_t = f"{current_time:.2f}"
            name_R = '001'
            loc_r = [float(lon), float(lat), float(alt)]
            # data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
            data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], phi, theta, psi)
            tacview.send_data_to_client(data_to_send)
            # time.sleep(0.001)

        mach = sim["velocities/mach"]
        # # 可以记录或打印
        # print(f"Time: {current_time:.1f}s, Mach: {mach:.3f}")
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")

