import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
import time
import torch
import matplotlib.pyplot as plt


# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from Math_calculates.sub_of_angles import *
from TrainAndTests.Controls.FlightControl_Train_dual_a_out2 import *
# from TrainAndTests.Controls.parallel_FlightControl_Train_dual_a_out import *
from Utilities.LocateDirAndAgents import *
from TrainAndTests.Controls.UPolicyWrapper import *

# 设备设置
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 网络初始化
action_dims_dict = {'cont': 4, 'cat': [], 'bern': 0}
policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
print(action_bound)
actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)

# 模型加载逻辑
pre_log_dir = os.path.join(project_root, "logs/control")
mission_name = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr"
# 可选其它控制器
"PID"
"FlightControl_parallel无课程无蒸馏_有过载限制_动态lr"
"FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr"

if mission_name != "PID":
    log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
    # log_dir = os.path.join(pre_log_dir, "FlightControl-run-20260308-211329")

    # 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
    actor_path = load_actor_from_log(log_dir, number=15800)
    if not actor_path:
        print(f"No actor checkpoint found in {log_dir}")
        sys.exit()
    else:
        sd = th.load(actor_path, map_location=device, weights_only=True)
        actor.load_state_dict(sd)
        print(f"Loaded actor for test from: {actor_path}")

# Benchmark 参数
height_list = [9000]
speed_list = [300]
dt_decide = 0.05
dt_move = 0.01
time_limit = 2 * 60  # 每组测试限时 5 分钟

target_range = 3e3

# 是否跟踪动目标（会导致超调量记录失效）
chasing_wave = 1
realistic = 1

avg_height_overshoot = 0
avg_heading_overshoot = 0
avg_v_error = 0
avg_ao = 0
avg_psi_error = 0
avg_theta_error = 0
survive_rate = 0

beta_ao = 0.01 ** (dt_decide / 10.0) # 超出最后10s以前的误差忽略不计
max_ny = -float('inf')
min_ny = float('inf')
max_alpha = 0
min_alpha = float('inf')
max_beta = 0

env = track_env(dt_move=dt_move, tacview_show=1, time_limit=time_limit)
env.realistic = realistic

# PID 策略初始化
pidcontroller = UnifiedPolicyWrapper(env, dt_decide=dt_decide) # 

# 目标变化的波动参数（正弦波轨迹，在不同测试中保持一致）
A_psi_dot = 6 * (pi / 180)  # deg/s 振幅
w_psi = 2 * pi / 120         # s 一个周期
A_h_dot = 100                # m/s 振幅
w_h = 2 * pi / 150           # s 一个周期

total_cases = len(height_list) * len(speed_list)
success_count = 0

t_bias = 0 # 用于 Tacview 时间偏移，防止轨道重叠

print(f"\nBenchmark 开始，当前测试配置 [{mission_name}]，共 {total_cases} 组测试案例...")
i=0
for init_h in height_list:
    height_req = np.clip(init_h - 5000, 1000, 13000)
    for target_v in speed_list:
        i+=1
        print(f"\n>>> 正在测试: 初始高度 {init_h}m, 目标速度 {target_v}m/s (t_bias: {t_bias:.1f}s)")
        
        # 固定初始化
        birth_state = {'position': np.array([0.0, init_h, 0.0]), 'psi': 1*pi/180}
        env.reset(birth_state=birth_state, height_req=init_h, psi_req=0, v_req=target_v, dt_report=dt_decide)
        
        obs, obs_check = env.get_obs()
        done = False
        # 初始化数据记录
        history = {
            'time': [], 'alpha': [], 'Ny': [], 'phi': [], 'h': [],
            'aileron': [], 'elevator': [], 'rudder': [], 'throttle': [],
            'psi': [], 'psi_req': [], 'theta': [], 'theta_req': [], 'v': [], 'v_req': []
        }
        
        ao_ema_episode = 0
        v_error_ema_episode = 0
        psi_error_ema_episode = 0.0
        theta_error_ema_episode = 0.0

        while not done:
            # 更新动态目标 (按照预设的正弦曲线变化)
            current_t = env.t
            
            if chasing_wave:
                # 航向角速波动
                # psi_dot_t = A_psi_dot * sin(w_psi * current_t)
                # env.psi_req += psi_dot_t * dt_decide
                env.psi_req = pi/2 * sin(w_psi * current_t)

                env.psi_req = sub_of_radian(env.psi_req, 0)
                
                # 高度变化率波动
                # h_dot_t = A_h_dot * - sin(w_h * current_t)
                # env.height_req += h_dot_t * dt_decide

                theta_req = 60 * (pi/180) * -sin(w_h * current_t)
                env.height_req = env.RUAV.alt + theta_req * 5000/(pi/2)

                env.height_req = np.clip(env.height_req, 1000, 13000)
            else:
                if env.t >= 30:
                    env.height_req = 5000
                    theta_req = (env.height_req-env.RUAV.alt) /5000*pi/2
                    env.psi_req = sub_of_radian(birth_state['psi'], 179*pi/180)
                    env.v_req = target_v
            
            # 决策
            obs, obs_check = env.get_obs()
            if mission_name != "PID":
                # NN
                action, u, _, _ = actor.get_action(obs, explore=0)
            else:
                # PID
                action = pidcontroller.get_action(obs, explore=0)
            # 推进环境
            next_obs, reward, done = env.step(action)
            
            # 数据记录
            aileron_val, elevator_val, rudder_val, throttle_val = action['cont']
            history['time'].append(env.t)
            history['alpha'].append(env.RUAV.alpha_air * 180 / pi)
            history['Ny'].append(env.RUAV.Ny)
            history['phi'].append(env.RUAV.phi * 180 / pi)
            history['h'].append(env.RUAV.alt)
            history['aileron'].append(aileron_val)
            history['elevator'].append(elevator_val)
            history['rudder'].append(rudder_val)
            history['throttle'].append(throttle_val)
            
            history['psi'].append(env.RUAV.psi * 180 / pi)
            history['psi_req'].append(env.psi_req * 180 / pi)
            history['theta'].append(env.RUAV.theta * 180 / pi)
            # 获取 theta_req，如果环境里没有则尝试从局部提取
            t_req = getattr(env, 'theta_req', 0)
            if 'theta_req' in locals(): t_req = locals()['theta_req']
            history['theta_req'].append(t_req * 180 / pi)
            
            history['v'].append(norm(env.RUAV.speed))
            history['v_req'].append(env.v_req)
            
            # 累加采样误差
            ao_ema_episode = beta_ao * ao_ema_episode + (1 - beta_ao) * (env.AO)
            v_error_ema_episode = beta_ao * v_error_ema_episode + (1 - beta_ao) * abs(env.v_error)
            psi_error_ema_episode = beta_ao * psi_error_ema_episode + (1 - beta_ao) * abs(env.psi_error)
            theta_error_ema_episode = beta_ao * theta_error_ema_episode + (1 - beta_ao) * abs(env.theta_error)

            
            # 记录飞行包线极限值
            max_ny = max(max_ny, env.RUAV.Ny)
            min_ny = min(min_ny, env.RUAV.Ny)
            max_alpha = max(max_alpha, abs(env.RUAV.alpha_air) * 180 / pi)
            min_alpha = min(min_alpha, env.RUAV.alpha_air * 180 / pi)
            max_beta = max(max_beta, abs(env.RUAV.beta_air) * 180 / pi)

            # 可视化输出 (使用 t_bias)
            env.render(t_bias, target_range=target_range)
            # --- 快照“残影”逻辑 ---
            if not hasattr(env, 'render_ids'):
                env.render_ids = []
                env.last_snapshot_time = 0

            # 注意避免在 0s 时刻马上生成重叠残影
            if env.t - env.last_snapshot_time >= 30.0 or env.t == dt_decide:
                env.last_snapshot_time = env.t
                if hasattr(env, 'tacview_show') and env.tacview_show:
                    # 分配不冲突的虚假 ID
                    ghost_uav_id = env.RUAV.id + int(env.t)*10 + 20000
                    ghost_target_id = ghost_uav_id + 1
                    env.render_ids.extend([ghost_uav_id, ghost_target_id])
                    
                    send_t = env.t + t_bias
                    data_to_send = ''
                    loc_LLH = env.RUAV.lon, env.RUAV.lat, env.RUAV.alt
                    pilot = f'Ghost_{int(env.t)}s'
                    color = 'Red'
                    # 添加飞机残影
                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{ghost_uav_id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{env.RUAV.phi * 180 / pi:.6f}|{env.RUAV.theta * 180 / pi:.6f}|{env.RUAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Pilot={pilot},Color={color}\n"
                    )
                    
                    # 添加目标残影
                    N, U, E = LLH2NUE(loc_LLH[0], loc_LLH[1], loc_LLH[2], lon_o=env.o00[0], lat_o=env.o00[1])
                    delta_N = target_range * cos(env.theta_req) * cos(env.psi_req)
                    delta_U = target_range * sin(env.theta_req)
                    delta_E = target_range * cos(env.theta_req) * sin(env.psi_req)
                    
                    delta_H = env.height_req
                    lon_T, lat_T, _ = NUE2LLH(N+delta_N, U+delta_U, E+delta_E, lon_o=env.o00[0], lat_o=env.o00[1])
                    
                    data_to_send += (
                        f"#{send_t:.2f}\n"
                        f"{ghost_target_id},T={(lon_T):.6f}|{(lat_T):.6f}|{delta_H:.6f},"
                        f"Name=Carrot,Color=Blue\n"
                    )
                    
                    env.tacview.send_data_to_client(data_to_send)
        
        env.clear_render(t_bias)
        
        # --- 清空快照残影 ---
        if hasattr(env, 'tacview_show') and env.tacview_show and hasattr(env, 'render_ids'):
            send_t = env.t + t_bias
            data_to_send = ''
            for ghost_id in env.render_ids:
                data_to_send += f"#{send_t:.2f}\n-{ghost_id}\n"
            if data_to_send:
                env.tacview.send_data_to_client(data_to_send)
            env.render_ids.clear()
            env.last_snapshot_time = 0
            
        t_bias += env.t # 累加偏置，使下一条轨迹衔接在后面
        
        steps_run = int(env.t/dt_decide)
        avg_height_overshoot += abs(env.height_overshoot)/total_cases
        avg_heading_overshoot += abs(env.heading_overshoot)*180/pi/total_cases
        avg_ao += ao_ema_episode/(1 - beta_ao**max(1, steps_run))/total_cases
        avg_v_error += v_error_ema_episode/(1 - beta_ao**max(1, steps_run))/total_cases
        avg_psi_error += psi_error_ema_episode/(1 - beta_ao**max(1, steps_run))/total_cases
        avg_theta_error += theta_error_ema_episode/(1 - beta_ao**max(1, steps_run))/total_cases
        
        # 判断本轮是否成功
        if not env.fail and env.t >= time_limit:
            success_count += 1
            print(f"结果: 成功. 耗时 {env.t:.1f}s")
            survive_rate += 1/total_cases
        else:
            print(f"结果: 失败. 因 {'失速/坠毁' if env.fail else '提前终止'}. 耗时 {env.t:.1f}s")

# 统计输出
print("\n" + "="*40)
print("             BENCHMARK SUMMARY")
print("="*40)
print(f"当前测试模型: {mission_name}")
print(f"测试总例数: {total_cases}")
print(f"成功例数:   {success_count}")
print(f"总成功率:   {success_count/total_cases*100:.1f}%")

# 全部回合平均误差指标 (EMA 偏差修正后的全口径汇报)
print(f"\n全部回合平均 EMA 误差指标:")
print(f" - 速度误差 (Speed Err):   {avg_v_error:.3f} m/s")
print(f" - 航向误差 (Heading Err): {avg_psi_error:.3f} deg")
print(f" - 高度误差 (Altitude Err): {avg_height_overshoot:.3f} m")
print(f" - 俯仰角误差 (Pitch Err): {avg_theta_error:.3f} deg")
print(f" - 指向误差 (AO):         {avg_ao:.3f} deg")
print("="*40)

# print("survive_rate", round(survive_rate,2)) # 重复了
print("avg_height_overshoot", avg_height_overshoot)
print("avg_heading_overshoot", avg_heading_overshoot)


print("\n飞行包线极限统计:")
print(f" - 最大正过载 (Max Ny): {max_ny:.3f}")
print(f" - 最大负过载 (Min Ny): {min_ny:.3f}")
print(f" - 最大迎角 (Max Alpha): {max_alpha:.3f} deg")
print(f" - 最小迎角 (Min Alpha): {min_alpha:.3f} deg")
print(f" - 最大侧滑角 (Max Beta): {max_beta:.3f} deg")

# --- 自动化绘图 (展示最后一组测试案例) ---
if len(history['time']) > 0:
    t = history['time']
    # 动态选择颜色：RL 为红色，PID 为蓝色
    color = 'r' if mission_name != "PID" else 'b'
    
    # Figure 1: 核心跟踪性能
    plt.figure(figsize=(15, 10))
    # 1. 航向角误差
    plt.subplot(3, 1, 1)
    psi_err = [abs(sub_of_degree(p, pr)) for p, pr in zip(history['psi'], history['psi_req'])]
    plt.plot(t, psi_err, color=color, label=f'{mission_name} Heading Error')
    plt.title("航向角误差 (Heading Error)")
    plt.ylabel(r"$\varepsilon_{\psi}$ (°)"); plt.legend(); plt.grid(True)
    
    # 2. 俯仰角跟踪
    plt.subplot(3, 1, 2)
    plt.plot(t, history['theta'], color=color, linestyle='-', label=f'{mission_name} Pitch')
    plt.plot(t, history['theta_req'], color=color, linestyle=':', alpha=0.6, label='Target Pitch')
    plt.title("俯仰角 (Pitch) 跟踪对比")
    plt.ylabel(r"$\theta$ (°)"); plt.legend(); plt.grid(True)
    
    # 3. 速度跟踪
    plt.subplot(3, 1, 3)
    plt.plot(t, history['v'], color=color, linestyle='-', label=f'{mission_name} Velocity')
    plt.plot(t, history['v_req'], color=color, linestyle=':', alpha=0.6, label='Target Velocity')
    plt.title("速度 (Velocity) 跟踪对比")
    plt.ylabel("v (m/s)"); plt.legend(); plt.grid(True)
    plt.tight_layout()

    # Figure 2: 飞行包线与控制量
    plt.figure(figsize=(15, 12))
    # 1. Alpha 与 Ny 对比
    ax1 = plt.subplot(2, 2, 1)
    ax1_r = ax1.twinx()
    ax1.plot(t, history['alpha'], color=color, linestyle='-', label='Alpha')
    ax1_r.plot(t, history['Ny'], color='g', linestyle=':', alpha=0.6, label='Ny')
    ax1.set_title("迎角 (Alpha) 与 法向过载 (Ny)")
    ax1.set_ylabel("Alpha (°)"); ax1_r.set_ylabel("Ny (g)")
    ax1.legend(loc='upper left'); ax1_r.legend(loc='upper right'); ax1.grid(True)
    
    # 2. Phi 与 高度 对比
    ax2 = plt.subplot(2, 2, 2)
    ax2_r = ax2.twinx()
    ax2.plot(t, history['phi'], color=color, linestyle='-', label='Phi')
    ax2_r.plot(t, history['h'], color='g', linestyle=':', alpha=0.6, label='Height')
    ax2_r.set_title("滚转角 (Phi) 与 高度 (Height)")
    ax2.set_ylabel("Phi (°)"); ax2_r.set_ylabel("Height (m)")
    ax2.legend(loc='upper left'); ax2_r.legend(loc='upper right'); ax2.grid(True)
    
    # 3. 控制量
    plt.subplot(2, 1, 2)
    ctrl_labels = ['aileron', 'elevator', 'rudder', 'throttle']
    ctrl_data = [history['aileron'], history['elevator'], history['rudder'], history['throttle']]
    for label, data in zip(ctrl_labels, ctrl_data):
        plt.plot(t, data, label=label)
    plt.title(f"{mission_name} 控制器指令")
    plt.ylabel("Normalized Command")
    plt.legend(); plt.grid(True)
    
    plt.suptitle(f"测试任务性能分析: {mission_name}", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()