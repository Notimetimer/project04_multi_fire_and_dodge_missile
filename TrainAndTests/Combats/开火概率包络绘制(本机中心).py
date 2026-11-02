import os
import sys
import glob
import re
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from math import pi, sin, cos
from itertools import product
from _context import *

from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
from Math_calculates.sub_of_angles import sub_of_radian
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# ======================= 可配置参数区 =======================
# 模型来源：MAPPO 同构训练产出的 ActorWrapper（dict 观测：fixed/enm/ally）
from Algorithms.ActorWithATTN import ActorWrapper

DIR_NAME = 'MAPPO_homo1v1-run-20261028-224802'

EXPERIMENT_NAME = None

# 抽取进度为几%的actor参数 / 弹药量（循环遍历所有组合）
PROGRESS_LIST = [100]  # select_progress_percentage (%)
AMMO_LIST = [2]            # ammo 初始弹药量（训练 red_init_ammo=4）

# 观测覆盖：中制导标志 / 自发射以来的等待时间
MID_TERM = 0      # r_obs[3]  missile_in_mid_term
T_SHOOT = 31.0 if MID_TERM else 120     # r_obs[21] = T_SHOOT / 120

# 场景参数
RED_HEIGHT = 2e3
BLUE_HEIGHT = 2e3
AA_HOR_DEG = 180
DEVICE = 'cpu'

# 场地半径：训练用 62km，但 reset 会把超出 R_cage-5km 的出生点按比例缩回场内，
# 8~100km 的扫描点会被裁剪；放大到 105km 保证设定距离不被改动。
# 我机恒在场地中心附近，border 观测只有 d_hor2border（50km 处封顶 1）与 dist_from_center(=0)，
# 两者都不随 R_cage 变化，因此放大 R_cage 不会改变网络输入。
R_CAGE = 105e3

# 扫描网格
DIST_MIN_KM, DIST_MAX_KM, DIST_STEP_KM = 8, 100, 15
ANGLE_MIN_DEG, ANGLE_MAX_DEG, ANGLE_STEP_DEG = -30, 30, 5
ANGLE_TICK_DEG = 15   # 扇面角度刻度间隔

# 绘图：开火概率不归一化，固定 0=最深蓝、1=白色
COLOR_LEVELS = 21     # 颜色采样等级（越大过渡越细）




# 出图规格
FIG_DPI = 300 # 300            # 保存分辨率 (dpi)
FIG_WIDTH_CM = 10 # 4.0       # 图宽 (cm)
FIG_HEIGHT_CM = None     # 图高 (cm)；None 时极坐标按正方形、直角坐标按 0.7 倍宽自动

# 输出目录（相对 project_root）与是否弹窗显示
OUT_DIR_NAME = os.path.join('结果展示', 'exp_png_mappo1v1')
SHOW_PLOT = 1
SAVE_FIG = 0
# ===========================================================


def create_initial_states(red_height, blue_height, distance, delta_psi, AA_hor=0):
    """
    创建初始状态配置，借用BasicRules_new_hierarchical找攻击区.py的初始化方法
    
    Args:
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m) 
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad) - 这是我机-目标视线方位角和初始我机航向角之间的差值
        AA_hor: 水平进入角偏移 (rad)
    
    说明：
    - delta_psi = 目标视线方位角 - 我机航向角
    - 红方(我机)初始位置：[0, h, 0]，航向角0 (向北)
    - 蓝方(目标)初始位置：[distance*cos(delta_psi), h, distance*sin(delta_psi)]
      航向角 = 0 + AA_hor (基础向东 + 进入角偏移)
    - 通过调整蓝方的横向位置来实现不同的delta_psi，而不是调整航向角
    """
    # 红方在西边[0, h, 0]面向东
    DEFAULT_RED_BIRTH_STATE = {
        'position': np.array([0.0, red_height, 0.0]),
        'psi': 0,  # 面向北
        'e2e': False
    }
    
    # 蓝方在东边，航向角 = 基础向东(0) + AA_hor(进入角偏移)
    # delta_psi通过调整蓝方位置来实现，而不是航向角
    blue_psi = sub_of_radian(delta_psi + AA_hor, 0)
    
    # 根据delta_psi调整蓝方的横向位置，实现不同的视线角度
    blue_N = distance * cos(delta_psi)  # 横向偏移
    blue_E = distance * sin(delta_psi)  # 纵向距离
    
    DEFAULT_BLUE_BIRTH_STATE = {
        'position': np.array([blue_N, blue_height, blue_E]),
        'psi': blue_psi,
        'e2e': False
    }
    
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

def select_agent_by_progress(log_dir, percentage):
    """
    扫描目录中 actor_rein*.pt，按编号排序，返回最接近指定进度百分比的文件。
    进度按最大编号计算：target = max_step * percentage / 100，取编号差最小的文件。
    """
    files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    step_files = []
    for f in files:
        m = re.fullmatch(r'actor_rein(\d+(?:\.\d+)?)\.pt', os.path.basename(f))
        if m:
            step_files.append((float(m.group(1)), f))
    if not step_files:
        return None
    step_files.sort(key=lambda x: x[0])
    target = step_files[-1][0] * percentage / 100.0
    step, best = min(step_files, key=lambda x: abs(x[0] - target))
    print(f"选择进度 {percentage}%: 目标步数 {target:.0f}, 选中 {os.path.basename(best)}")
    return best

def load_trained_actor(model_path, device='cpu'):
    """加载训练好的actor模型"""
    # MAPPO 同构训练的 actor_rein{batch}.pt 由 torch.save(actor.state_dict()) 落盘，
    # 网络维度统一从 env_obs_dims.json 读取（与训练 worker 的 ActorWrapper 完全一致）
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    actor = ActorWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape,
                         ally_shape=ally_shape, device=device)
    
    # 加载权重
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    
    return actor

def run_single_step_firing_probability(actor, env, red_height, blue_height, distance, delta_psi,
                                       AA_hor=0, device='cpu', ammo=4, init_states=None):
    """
    运行单步并获取开火概率
    
    Args:
        actor: 策略网络（ActorWrapper）
        env: 已构建的环境（复用同一个实例，逐点 reset）
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m)
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad)
        AA_hor: 水平进入角偏移 (rad)
        device: 计算设备
        ammo: 初始弹药量
        init_states: init_states.json 内容（初始数量与存活数量）
    
    返回:
        (有效开火概率, 是否未被规则禁火)
        有效开火概率 = sigmoid(fire_head + 三条禁火规则 delta)，与训练时采样/评估口径一致；
        被规则禁火（不可攻击/刚发射/无弹）时为 0。
    """
    init_states = load_init_states() if init_states is None else init_states
    
    # 创建初始状态
    red_state, blue_state = create_initial_states(
        red_height=red_height, 
        blue_height=blue_height, 
        distance=distance, 
        delta_psi=delta_psi, 
        AA_hor=AA_hor
    )
    
    # 重置环境
    # random_pos=False：关闭队内出生点顺序打乱，保证存活机（固定击杀 id 最大的那架）
    # 与被击杀槽位的相对几何与设定一致；红蓝 index0 各偏移 -5km，相对几何不受影响
    env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
              red_init_num=init_states['red_init_num'],
              blue_init_num=init_states['blue_init_num'],
              red_init_ammo=ammo, blue_init_ammo=ammo,
              seed=(0, 0), options={"random_pos": False})
    
    # 刷新可见性：reset 时可见性矩阵仍为全 1（被当作“仅 RWR 方位可见”），
    # 此时 get_obs 会在 vis_level==1 分支把 enm_AA 强制写成 +π，与真实几何无关；
    # 先按当前几何算一次原始可见性，保证扫描到的是真实 enm_AA（本网格内应为完全可见）
    env.update_entity_states()

    # 获取观测（1v1 由 init_states.json 的 alive_num 决定：先按 init_num 补槽位，
    # 再按 id 升序保留存活机，因此存活红机是 id 最小的那架，enm/ally 序列长度仍为 2/1）
    red_id = min(i for i in env.RUAV_ids if not env.find_uav_with_id(i).dead)
    agent_input, mask_input, rule_input = env.get_obs(red_id, pomdp=1)
    
    # 验证开火等待时间和中制导标志位是否起到效果
    t_since_launch = rule_input['weapon_infos']['missile_t_fly']
    missile_in_mid_term = rule_input['weapon_infos']['mid_term']
    
    # 修改数值（rule_input 是 unscale 后的物理量：等待时间单位为 s）
    rule_input['weapon_infos']['missile_t_fly'] = T_SHOOT  # 设置等待时间
    rule_input['weapon_infos']['mid_term'] = MID_TERM      # 设置中制导标志位
    
    # 与训练 worker 相同的 rule_input 堆叠方式（B=1 的 dict-of-array）
    actor_rule = stack_field([take_some_of_rule_inputs(rule_input)])
    
    # 获取开火概率
    with torch.no_grad():
        packed_agent, packed_mask = actor.pack_inputs(agent_input, mask_input)
        # 1v1 只有一个存活敌机：直接用掩码选出它，不采样，避免同一点结果随机抖动
        enm_idx = torch.as_tensor([int(np.argmax(mask_input['enm']))],
                                  dtype=torch.long, device=device)
        selected_info, _, _, mini_rule_inputs, _, _, _, _ = actor.selector(
            packed_agent, packed_mask, actor_rule, enm_idx=enm_idx
        )
        pilot_logits = actor.pilot(selected_info, mini_rule_inputs)
        # 有效开火概率：fire_head 输出已叠加三条禁火规则 delta（-1e8 量级），sigmoid 后为 0
        firing_probability = torch.sigmoid(pilot_logits['fire'].squeeze(-1))[0].item()
        # fire_mask=1 表示该点未被规则禁火（0 表示禁火，此时概率恒为 0）
        fire_allowed = float(pilot_logits['fire_mask'][0].item())
    
    
    return firing_probability, fire_allowed

def plot_firing_probability_heatmap_polar(delta_psis, distances, probabilities):
    """
    在极坐标中绘制开火概率热图（扇形）
    
    Args:
        delta_psis: 角度数组 (rad)
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (delta_psi_count, distance_count)
    """
    # 创建扇形极坐标网格
    Theta, R = np.meshgrid(delta_psis, distances/1000)
    
    # 转置probabilities以匹配网格形状
    # probabilities形状: (delta_psi_count, distance_count)
    # 网格形状: (distance_count, delta_psi_count)
    probabilities_plot = probabilities.T
    
    # 图尺寸：cm -> inch；高度默认与宽度相同（正方形）
    fig_w = FIG_WIDTH_CM / 2.54
    fig_h = (FIG_HEIGHT_CM if FIG_HEIGHT_CM is not None else FIG_WIDTH_CM) / 2.54
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(fig_w, fig_h))
    
    # 绘制热图：固定 0~1 映射，0=最深蓝，1=白色，不做数据归一化
    c = ax.contourf(Theta, R, probabilities_plot,
                    levels=np.linspace(0, 1, COLOR_LEVELS + 1),
                    cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=1))
    # 添加颜色条：不显示标签，刻度固定为百分比
    cbar = plt.colorbar(c, ax=ax, pad=0.1)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # 设置标签
    ax.set_theta_zero_location('N')  # 0度在北边（正前方）
    ax.set_theta_direction(-1)  # 顺时针方向
    # ax.set_title('Firing Probability Heatmap (Sector View)', pad=20)
    ax.set_rlabel_position(45)  # 径向标签位置
    
    # 设置角度范围只显示数据范围
    ax.set_thetamin(ANGLE_MIN_DEG)
    ax.set_thetamax(ANGLE_MAX_DEG)
    
    # 设置角度标签：间隔 ANGLE_TICK_DEG
    theta_ticks_deg = np.arange(ANGLE_MIN_DEG, ANGLE_MAX_DEG + ANGLE_TICK_DEG, ANGLE_TICK_DEG)
    ax.set_thetagrids(theta_ticks_deg, [f'{int(t)}°' for t in theta_ticks_deg])
    
    # 添加中心线标记
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    return fig, ax

def plot_firing_probability_heatmap_cartesian(delta_psis, distances, probabilities):
    """
    在直角坐标系中绘制开火概率热图
    
    Args:
        delta_psis: 角度数组 (rad)
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (delta_psi_count, distance_count)
    """
    # 图尺寸：cm -> inch；高度默认 0.7 倍宽
    fig_w = FIG_WIDTH_CM / 2.54
    fig_h = (FIG_HEIGHT_CM if FIG_HEIGHT_CM is not None else FIG_WIDTH_CM * 0.7) / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # 创建网格 - 注意顺序要与probabilities数组匹配
    Delta_Psi, Distances = np.meshgrid(delta_psis, distances/1000)
    
    # 转置probabilities以匹配网格形状
    # probabilities形状: (delta_psi_count, distance_count)
    # 网格形状: (distance_count, delta_psi_count)
    probabilities_plot = probabilities.T
    
    # 绘制热图：固定 0~1 映射，0=最深蓝，1=白色，不做数据归一化
    c = ax.contourf(Delta_Psi, Distances, probabilities_plot,
                    levels=np.linspace(0, 1, COLOR_LEVELS + 1),
                    cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=1))
    
    # 添加颜色条：不显示标签，刻度固定为百分比
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # 设置标签
    ax.set_xlabel('Delta Psi (rad)')
    ax.set_ylabel('Distance (km)')
    ax.set_title('Firing Probability Heatmap (Cartesian Coordinates)')
    
    # 设置角度标签（转换为度数），间隔 ANGLE_TICK_DEG
    x_ticks_deg = np.arange(ANGLE_MIN_DEG, ANGLE_MAX_DEG + ANGLE_TICK_DEG, ANGLE_TICK_DEG)
    ax.set_xticks(np.radians(x_ticks_deg))
    ax.set_xticklabels([f'{int(t)}°' for t in x_ticks_deg])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def main():
    # 固定参数（全部来自文件顶部的可配置区）
    red_height = RED_HEIGHT
    blue_height = BLUE_HEIGHT
    AA_hor = np.radians(AA_HOR_DEG)
    device = DEVICE
    
    # 网格搜索参数
    distances = np.arange(DIST_MIN_KM * 1e3, DIST_MAX_KM * 1e3 + DIST_STEP_KM * 1e3, DIST_STEP_KM * 1e3)
    delta_psis = np.arange(np.radians(ANGLE_MIN_DEG),
                           np.radians(ANGLE_MAX_DEG) + np.radians(ANGLE_STEP_DEG),
                           np.radians(ANGLE_STEP_DEG))
    
    print(f"开始计算开火概率...")
    print(f"距离范围: {DIST_MIN_KM:.0f}km - {DIST_MAX_KM:.0f}km, 间隔: {DIST_STEP_KM:.0f}km")
    print(f"角度范围: {ANGLE_MIN_DEG:.0f}° - {ANGLE_MAX_DEG:.0f}°, 间隔: {ANGLE_STEP_DEG:.0f}°")
    print(f"总计算点数: {len(distances) * len(delta_psis)}")
    
    # 查找日志目录
    dir_name = DIR_NAME
    experiment_name = EXPERIMENT_NAME
    
    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, experiment_name)
    
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{experiment_name}'")
    
    # 输出目录（相对 project_root），不存在则创建
    out_dir = os.path.join(project_root, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # 环境只建一次，逐点 reset（初始数量/存活数量与训练一致）
    init_states = load_init_states()
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=R_CAGE)
    env = ChooseStrategyEnv(env_args, tacview_show=False, vertices=None)
    env.dt_move = 0.07
    env.dt_maneuver = 0.2
    
    total_points = len(distances) * len(delta_psis)
    
    # 循环遍历 progress × ammo 组合
    for prog in PROGRESS_LIST:
        # 按进度百分比选取 actor_rein*.pt
        agent_path = select_agent_by_progress(latest_log_dir, prog)
        if not agent_path:
            print(f"跳过 prog={prog}: '{latest_log_dir}' 中没有 actor_rein 文件")
            continue
        print(f"找到模型: {agent_path}")
        
        try:
            actor = load_trained_actor(agent_path, device=device)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            continue
        
        for ammo in AMMO_LIST:
            print(f"\n===== ammo={ammo}, progress={prog}% =====")
            # 计算开火概率网格
            probabilities = np.zeros((len(delta_psis), len(distances)))
            fire_allowed_grid = np.zeros((len(delta_psis), len(distances)))
            current_point = 0
            
            for i, delta_psi in enumerate(delta_psis):
                for j, distance in enumerate(distances):
                    current_point += 1
                    print(f"进度: {current_point}/{total_points} ({100*current_point/total_points:.1f}%)", end='\r')
                    
                    try:
                        prob, allowed = run_single_step_firing_probability(
                            actor=actor, 
                            env=env,
                            red_height=red_height, 
                            blue_height=blue_height, 
                            distance=distance, 
                            delta_psi=delta_psi, 
                            AA_hor=AA_hor, 
                            device=device,
                            ammo=ammo,
                            init_states=init_states
                        )
                        probabilities[i, j] = prob
                        fire_allowed_grid[i, j] = allowed
                    except Exception as e:
                        print(f"\n计算错误 (距离={distance/1000:.0f}km, 角度={np.degrees(delta_psi):.0f}°): {e}")
                        probabilities[i, j] = 0.0
                        fire_allowed_grid[i, j] = 0.0
            
            print(f"\n计算完成！")
            # 有效概率统计：被规则禁火的点概率恒为 0，单独统计便于区分“网络不想开火”与“规则禁火”
            allowed_num = int((fire_allowed_grid > 0.5).sum())
            print(f"有效开火概率: mean={probabilities.mean():.4f}, max={probabilities.max():.4f}, "
                  f"min={probabilities.min():.4f}, 规则禁火点={total_points - allowed_num}/{total_points}")
            
            base_name = f"fire_prob_prog{prog}_ammo{ammo}"
            
            # 保存结果
            results = {
                'distances_km': distances / 1000,
                'delta_psis_rad': delta_psis,
                'delta_psis_deg': np.degrees(delta_psis),
                'probabilities': probabilities,
                'fire_allowed': fire_allowed_grid
            }
            if SAVE_FIG:
                np.savez(os.path.join(out_dir, base_name + '.npz'), **results)
            
            # 绘制图形并保存 png + svg
            # 直接画单步有效开火概率（不再做 1-(1-p)^5 的连发累积变换），便于定位网络本身的开火倾向
            fig1, ax1 = plot_firing_probability_heatmap_polar(
                delta_psis=delta_psis, 
                distances=distances, 
                probabilities=probabilities
            )
            if SAVE_FIG:
                fig1.savefig(os.path.join(out_dir, base_name + '.png'), dpi=FIG_DPI, bbox_inches='tight')
                fig1.savefig(os.path.join(out_dir, base_name + '.svg'), bbox_inches='tight')
                print(f"已保存: {os.path.join(out_dir, base_name)}.png/.svg")
            if SHOW_PLOT:
                plt.show()
            plt.close(fig1)
    
    print("\n全部组合完成！")

if __name__ == '__main__':
    main()
