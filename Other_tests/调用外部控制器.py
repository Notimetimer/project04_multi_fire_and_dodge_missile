import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import *
import time
import jsbsim
import numpy as np
from numpy.linalg import norm
from Controller.F16PIDController2 import F16PIDController, active_rotation, sub_of_radian, sub_of_degree

tacview_show = 1  # 是否显示Tacview

dt = 0.02  # 0.02

if tacview_show:
    print('please prepare tacview')
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Visualize.tacview_visualize import Tacview

    tacview = Tacview()

# 启动 JSBSim
sim = jsbsim.FGFDMExec(None, None)
sim.set_debug_level(0)
sim.set_dt(dt)  # 解算步长 dt 秒

# 设置模型路径（一般 JSBSim pip 包自动包含）
sim.load_model("f16")  # f15, p51d, ball 等模型可选

# 初始姿态与目标姿态
# 连续输出并tacview中可视化
start_time = time.time()
target_height = 5000  # m
target_heading = 120  # 度 to rad
target_speed = 300  # m/s
t_last = 60*3

# 设置初始状态（单位：英尺、节、角度）
sim["ic/h-sl-ft"] = 12000 * 3.2808  # 高度：m -> ft
sim["ic/vt-kts"] = 400 * 1.9438  # 空速： m/s-> 节
sim["ic/psi-true-deg"] = 0  # 航向角: °
sim["ic/phi-deg"] = 0
sim["ic/theta-deg"] = 0
sim["ic/alpha-deg"] = 0
sim["ic/beta-deg"] = 0
sim["ic/long-gc-deg"] = -118  # 设置初始经度（单位：度）
sim["ic/lat-gc-deg"] = 34  # 设置初始纬度（单位：度）

# 初始化状态
sim.run_ic()
sim.set_property_value('propulsion/set-running', -1)

# 记录轨迹和状态数据
psis = []
thetas = []
phis = []
heights=[]
alphas=[]
betas=[]

thrust_data = []  # 添加推力数据记录
time_steps = []

# 记录控制量
aileron_cmd = []
elevator_cmd = []
rudder_cmd = []
throttle_cmd = []

# 指数滑动平均控制量
aileron_last = 0
elevator_last = 0
rudder_last = 0
throttle_last = 0.5

f16PIDController=F16PIDController() # ??? 正确的位置带来错误的结果？

for step in range(int(t_last / dt)):
    current_t = step*dt
    if current_t<15:
        target_height = 5000  # m
        target_heading = 90  # 度 to rad
    elif current_t<1*60:
        target_height = 10000  # m
        target_heading = -120  # 度 to rad
    elif current_t<1*60+27:
        target_height = 7000  # m
        target_heading = 0  # 度 to rad
    elif current_t<2*60+10: 
        target_height = 8000  # m
        target_heading = sub_of_degree(sim["attitude/psi-deg"], 60)  # 度 to rad
    else:
        target_heading = sub_of_degree(sim["attitude/psi-deg"], -10)

    sim.run()
    current_time = step * dt
    time_steps.append(current_time)

    delta_heading = sub_of_degree(target_heading, sim["attitude/psi-deg"])

    # 取姿态角度
    phi = sim["attitude/phi-deg"]      # 滚转角 (roll)
    theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
    psi = sim["attitude/psi-deg"]      # 航向角 (yaw)

    # 速度矢量关于地面的角度
    vn = sim["velocities/v-north-fps"]    # 向北分量
    ve = sim["velocities/v-east-fps"]     # 向东分量
    vu = -sim["velocities/v-down-fps"]     # 向下分量（正表示下降）

    gamma_angle = atan2(vu, sqrt(vn**2+ve**2))*180/pi      # 爬升角（度）
    course_angle = atan2(ve, vn)*180/pi    # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

    # 构建观测向量
    obs_jsbsim = np.zeros(14)
    obs_jsbsim[0] = target_height/5000  # 期望高度
    obs_jsbsim[1] = delta_heading * pi / 180  # 期望相对航向角
    obs_jsbsim[2] = target_speed / 340  # 期望速度
    obs_jsbsim[3] = sim["attitude/theta-deg"] * pi / 180  # 当前俯仰角
    obs_jsbsim[4] = sim["velocities/vt-fps"] * 0.3048 / 340  # 当前速度
    obs_jsbsim[5] = sim["attitude/phi-deg"] * pi / 180  # 当前滚转角
    obs_jsbsim[6] = sim["aero/alpha-deg"] * pi / 180  # 当前迎角
    obs_jsbsim[7] = sim["aero/beta-deg"] * pi / 180  # 当前侧滑角
    obs_jsbsim[8] = sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
    obs_jsbsim[9] = sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
    obs_jsbsim[10] = sim["velocities/r-rad_sec"]  # 偏航角速度（弧度/秒）
    obs_jsbsim[11] = gamma_angle * pi / 180 # 爬升角
    obs_jsbsim[12] = sub_of_degree(target_heading, course_angle) * pi / 180 # 航迹角
    obs_jsbsim[13] = sim["position/h-sl-ft"] * 0.3048 /5000  # 高度/5000（英尺转米）

    # f16PIDController=F16PIDController() # ???? 错误的位置带来正确的结果？

    # 输出姿态控制指令
    norm_act = f16PIDController.flight_output(obs_jsbsim, dt=dt)

    # 指数滑动平均控制量
    last_control = np.array([aileron_last, elevator_last, rudder_last, throttle_last])
    norm_act=last_control*0.1+0.9*np.array(norm_act)

    sim["fcs/aileron-cmd-norm"], \
        sim["fcs/elevator-cmd-norm"], \
        sim["fcs/rudder-cmd-norm"], \
        sim["fcs/throttle-cmd-norm"] = norm_act.tolist()  # 设置控制量

    # # 记录控制量
    aileron_cmd.append(sim["fcs/aileron-cmd-norm"])
    elevator_cmd.append(sim["fcs/elevator-cmd-norm"])
    rudder_cmd.append(sim["fcs/rudder-cmd-norm"])
    throttle_cmd.append(sim["fcs/throttle-cmd-norm"])

    # 取当前位置
    lon = sim["position/long-gc-deg"]  # 经度
    lat = sim["position/lat-gc-deg"]  # 纬度
    alt = sim["position/h-sl-ft"] * 0.3048  # 高度（英尺转米）

    # 取速度分量
    u = sim["velocities/u-fps"] * 0.3048  # X轴速度 (fps转m/s)
    v = sim["velocities/v-fps"] * 0.3048  # Y轴速度 (fps转m/s)
    w = sim["velocities/w-fps"] * 0.3048  # Z轴速度 (fps转m/s)

    # 记录状态量
    phis.append(phi)
    psis.append(psi)
    thetas.append(theta)
    heights.append(alt)

    # 通过tacview可视化
    if tacview_show and step % np.round(0.4 / dt) == 0:
        send_t = f"{current_time:.2f}"
        name_R = '001'
        loc_r = [float(lon), float(lat), float(alt)]
        # data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
        data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
        float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], phi, theta, psi)
        tacview.send_data_to_client(data_to_send)
        # time.sleep(0.001)

    mach = sim["velocities/mach"]
    # # 可以记录或打印
    # print(f"Time: {current_time:.1f}s, Mach: {mach:.3f}")

    # time.sleep(0.01)

end_time = time.time()
print(f"程序运行时间: {end_time - start_time:.2f} 秒")

