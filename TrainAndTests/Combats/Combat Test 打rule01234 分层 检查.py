import os
import sys
import numpy as np
import torch
import argparse
import glob
import re
from math import pi
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules_new_hierarchical import basic_rules
# from BasicRules_new_hierarchical2 import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper, infer_mask_cfg_from_actor_meta # 纯MLP

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2  # 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# def create_initial_state():
#     """创建固定的初始状态"""
#     blue_height, red_height = 8000, 8000
#     red_psi, blue_psi = -pi / 2, pi / 2
#     red_N, red_E = 0, 55e3  # 55e3
#     blue_N, blue_E = red_N, -red_E # -45e3
#     DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
#     DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
#     return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- 3. 主程序 ---
if __name__ == "__main__":

    gamma = 0.97

    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = None
    dir_name = "SLWSPFSP0.15_flymask_v0h0-run-20260908-223429"
    
    "SLWSPFSPNoIL_flymask_v0h0-run-20260906-171344"
    
    "SLWSPFSP0.3无引导奖励-run-20260726-091904"

    
    "SLWSA3C0.3-run-20260630-220403"
    
    "SLWSPFSP0.3-run-20260618-221044"

    
    # 次要
    experiment_name = None    
    'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项'


    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    parser.add_argument("--mission-name", type=str, default=experiment_name, help="Mission name to find the log directory.")
    args = parser.parse_args()    

    args.agent_id = None # 40
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3) # 55e3
    hidden_dim = [128, 128, 128]
    device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    # 构建场地边界
    vertices = None # 默认圆形边界
    # 南北长54km，东西宽100km的长方形边界
    # vertices = [[29.9e3, 50e3], [-29.9e3, 50e3], [-29.9e3, -50e3], [29.9e3, -50e3]]
    env = ChooseStrategyEnv(env_args, tacview_show=2, vertices=vertices)
    env.dt_move = 0.07 # 025 # 2 # 0.05 # 0.04 # 25

    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 (相对根目录指定 actor 路径) ---
    agent_path = os.path.join(project_root, "Controller", "NNfor1v1", "group1", "actor_rein805.pt")
    actor_meta_path = os.path.join(project_root, "Controller", "NNfor1v1", "group1", "actor.meta.json")

    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {agent_path}")
    if not os.path.exists(actor_meta_path):
        raise FileNotFoundError(f"Actor meta not found: {actor_meta_path}")

    print()
    print(f"Loading agent weights from: {agent_path}")
    print(f"Loading actor meta from: {actor_meta_path}")
    print()

    # [新增] 根据 actor.meta.json 自动推断 ver/hor 配置
    mask_cfg = infer_mask_cfg_from_actor_meta(actor_meta_path)

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device)
    # 注意：测试时只需要 Actor Wrapper，不需要完整的 PPO agent
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() # **非常重要**：设置为评估模式


    # if env.tacview_show:
    #     env.tacview = Tacview()
    #     env.tacview.handshake()
    #     env.visualize_cage()

    env.shielded = 1
    env.no_out = 0 # 强制防止出界，训练的时候为0，测试的时候为1
    
    # --- 循环测试 ---
    rule_opponents = [3] # [0,1,2,3,4] # [3]

    t_bias = 0

    try:
        for rule_num in rule_opponents:
            print("\n" + "="*50)
            print(f"--- Starting Test: Loaded Actor(Red) vs Rule_{rule_num}(Blue) ---")
            print("="*50)

            # 重置环境
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = None, None # create_initial_state()
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE, ego_side='r', 
                      red_init_ammo=4, blue_init_ammo=4)

            done = False
            last_r_action_label = 0
            last_b_action_label = 0
            r_action_label = 0
            b_action_label = 0

            # --- 初始化数据记录 ---
            history = {
                'time': [],
                'r_ny': [], 'r_alpha': [], 'r_alt': [], 'r_mach': [],
                'b_ny': [], 'b_alpha': [], 'b_alt': [], 'b_mach': [],
                'r_cat_entropy': [],
                'r_cat_conf': [], 'r_bern_fire_prob': [],
                'r_warning': [], 'r_missile_in_mid_term': [],
                'r_reward': [],
                'r_r_shaping': [],
                'r_count_12km': [], 'r_count_4km': [],
                'b_count_12km': [], 'b_count_4km': [],
            }

            fire_time = -120
            r_cat_entropy = 0.0
            r_bern_fire_prob = 0.0
            r_cat_conf = 0.0
            r_warning = 0.0
            r_missile_in_mid_term = 0.0

            # 回合仿真循环
            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                r_state_check = env.unscale_state(r_check_obs)
                r_warning = float(r_state_check['warning'])
                r_missile_in_mid_term = float(r_state_check['missile_in_mid_term'])

                # 决策
                if count % action_cycle_multiplier == 0:
                    # --- 红方 (RL 智能体) ---
                    with torch.no_grad():
                        r_action_exec, _, _, r_action_check = actor_wrapper.get_action(
                            r_obs, explore={'cont':0, 'cat':1, 'bern':0}, check_obs=r_check_obs, bern_threshold=0.4,
                            temperature={'cat':0.3, 'bern':0.5}
                            ) # check_obs=r_check_obs, check_obs=None 0.06
                    # print("中制导状态", r_obs[3])
                    r_action_label = r_action_exec['cat'] # [0]
                    r_fire = r_action_exec['bern'][0]
                    last_r_action_label = r_action_label

                    # 计算并记录 cat 熵与 bern 熵
                    eps = 1e-8
                    r_cat_entropy = 0.0
                    cat_top1_list = []
                    if 'cat' in r_action_check and r_action_check['cat'] is not None:
                        for cat_probs in r_action_check['cat']:
                            p = np.asarray(cat_probs).flatten()
                            p = np.clip(p, eps, 1.0)
                            r_cat_entropy += -np.sum(p * np.log(p))
                            cat_top1_list.append(np.max(p))
                    # cat 多头的 top-1 置信度（几何平均）
                    if len(cat_top1_list) > 0:
                        r_cat_conf = np.exp(np.mean(np.log(np.clip(cat_top1_list, eps, 1.0))))
                        print(f"红方(RL) cat 综合置信度: {r_cat_conf:.4f}")

                    r_bern_fire_prob = 0.0
                    if 'bern' in r_action_check and r_action_check['bern'] is not None:
                        p = np.asarray(r_action_check['bern']).flatten()
                        r_bern_fire_prob = float(p[0])
                        print(f"红方(RL) 开火概率: {r_bern_fire_prob:.4f}")

                    if r_fire:
                        env.RUAV.about_to_fire = 1
                        
                        print("开火瞬间状态观测", r_check_obs)
                        print("开火瞬间动作", r_action_label)

                    # --- 蓝方 (规则智能体) ---
                    b_state_check = env.unscale_state(b_check_obs)
                    b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
                    last_b_action_label = b_action_label
                    if b_fire:
                        env.BUAV.about_to_fire = 1

                # 执行机动并步进
                r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                
                # 测试时限制开火后爬升
                if getattr(env.RUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'r', tabu=1, action_label=None) # r_action_label)
                    print("Shoot")
                    print()
                    fire_time = env.t
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', tabu=1, action_label=None) # b_action_label)
                

                if (action_cycle_multiplier-1) * env.dt_maneuver <= env.t-fire_time < 2 * action_cycle_multiplier * env.dt_maneuver:
                    print("开火后瞬间观测", r_check_obs)
                    print("开火后动作", r_action_label)
                    print()

                env.step(r_maneuver, b_maneuver)
                # 统计红方的奖励与状态
                done, b_r1, b_r2, b_r3 = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)
                # [诊断] 补一次蓝方调用（返回值丢弃）：更新蓝方导弹 get_in_12km/get_in_4km 锁存与 BUAV._count 计数，
                # 与训练时双侧每步都调用保持一致；放在 'r' 调用之后，不影响已记录的红方奖励。
                env.combat_terminate_and_reward('b', b_action_label, b_fire, action_cycle_multiplier)

                # if abs(env.t % 5) < 0.1:
                    # print("当前动作", r_action_exec)
                    # print("当前奖励函数", b_r1)
                    # print()

                # --- 记录数据 ---
                history['time'].append(count * dt_maneuver)
                history['r_ny'].append(env.RUAV.Ny)
                history['r_alpha'].append(env.RUAV.alpha_air * 180 / np.pi)
                history['r_alt'].append(env.RUAV.alt)
                history['r_mach'].append(env.RUAV.mach)
                history['b_ny'].append(env.BUAV.Ny)
                history['b_alpha'].append(env.BUAV.alpha_air * 180 / np.pi)
                history['b_alt'].append(env.BUAV.alt)
                history['b_mach'].append(env.BUAV.mach)
                history['r_cat_entropy'].append(r_cat_entropy)
                history['r_bern_fire_prob'].append(r_bern_fire_prob)
                history['r_cat_conf'].append(r_cat_conf)
                history['r_warning'].append(r_warning)
                history['r_missile_in_mid_term'].append(r_missile_in_mid_term)
                history['r_reward'].append(b_r1)
                history['r_r_shaping'].append(env.RUAV.r_shaping)
                history['r_count_12km'].append(getattr(env.RUAV, '_count_12km', 0))
                history['r_count_4km'].append(getattr(env.RUAV, '_count_4km', 0))
                history['b_count_12km'].append(getattr(env.BUAV, '_count_12km', 0))
                history['b_count_4km'].append(getattr(env.BUAV, '_count_4km', 0))

                env.render(t_bias=t_bias)

            # 报告结果
            result = "Draw"
            if env.win: result = "Win"
            elif env.lose: result = "Lose"
            print(f"\n--- Test Finished. Result for Red (Loaded Agent): {result} ---")
            
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            
            # --- 保存作战记录到 CSV ---
            try:
                df_history = pd.DataFrame(history)
                save_name = f"CombatLog_vs_Rule_2609_{rule_num}.csv" #_{datetime.datetime.now().strftime('%H%M%S')}.csv"
                save_path = os.path.join(project_root, "logs", save_name)
                df_history.to_csv(save_path, index=False)
                print(f"Combat data for Rule {rule_num} saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

            # --- 绘制曲线 ---
            plt.figure(figsize=(10, 10))
            plt.subplot(4, 1, 1)
            plt.plot(history['time'], history['r_ny'], label='Red Ny', color='crimson')
            plt.plot(history['time'], history['b_ny'], label='Blue Ny', color='royalblue', linestyle='--')
            plt.axhline(y=-3, color='black', linestyle=':', alpha=0.7, label='Ny limit (-3g)')
            plt.axhline(y=9, color='black', linestyle=':', alpha=0.7, label='Ny limit (9g)')
            plt.ylabel('Ny (g)')
            plt.title(f'Test vs Rule {rule_num}: Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 2)
            plt.plot(history['time'], history['r_alpha'], label='Red Alpha', color='crimson')
            plt.plot(history['time'], history['b_alpha'], label='Blue Alpha', color='royalblue', linestyle='--')
            plt.axhline(y=-8, color='black', linestyle=':', alpha=0.7, label='Alpha limit (-8°)')
            plt.axhline(y=26, color='black', linestyle=':', alpha=0.7, label='Alpha limit (26°)')
            plt.ylabel('Alpha (deg)')
            plt.title('Angle of Attack (Alpha)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 3)
            plt.plot(history['time'], history['r_mach'], label='Red Mach', color='crimson')
            plt.plot(history['time'], history['b_mach'], label='Blue Mach', color='royalblue', linestyle='--')
            plt.ylabel('Mach')
            plt.title('Flight Mach Number')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 4)
            plt.plot(history['time'], history['r_alt'], label='Red Alt', color='crimson')
            plt.plot(history['time'], history['b_alt'], label='Blue Alt', color='royalblue', linestyle='--')
            plt.ylabel('Alt (m)')
            plt.xlabel('Time (s)')
            plt.title('Altitude (Height)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()

            # --- Figure 4: r_shaping 与其余部分在累积蒙特卡洛回报中的占比 ---
            plt.figure(2, figsize=(10, 4))
            plt.clf()
            t = np.array(history['time'])
            r_total = np.array(history['r_reward'])
            r_shaping = np.array(history['r_r_shaping'])
            r_rest = r_total - r_shaping

            # 累积蒙特卡洛回报（从当前步到回合结束的折扣和）
            G_shaping = np.zeros_like(r_shaping)
            G_rest = np.zeros_like(r_rest)
            running_shaping = 0.0
            running_rest = 0.0
            for i in range(len(r_total) - 1, -1, -1):
                running_shaping = r_shaping[i] + gamma * running_shaping
                running_rest = r_rest[i] + gamma * running_rest
                G_shaping[i] = running_shaping
                G_rest[i] = running_rest

            denom = np.abs(G_shaping) + np.abs(G_rest)
            denom = np.where(denom == 0, 1, denom)
            shaping_ratio = np.abs(G_shaping) / denom
            rest_ratio = np.abs(G_rest) / denom

            plt.plot(t, shaping_ratio, label='r_shaping 占比', color='crimson')
            plt.plot(t, rest_ratio, label='其余部分占比', color='royalblue', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('占比')
            plt.ylim([0, 1])
            plt.title(f'Test vs Rule {rule_num}: r_shaping 与其余部分在累积蒙特卡洛回报中的占比')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- Figure 3: 红蓝双方 _count_12km / _count_4km 变化（分子图避免曲线重叠遮挡） ---
            fig3 = plt.figure(3, figsize=(10, 6))
            fig3.clf()
            t3 = np.array(history['time'])
            ax_r = fig3.add_subplot(211)
            ax_b = fig3.add_subplot(212, sharex=ax_r)

            ax_r.step(t3, history['r_count_12km'], where='post', label='Red _count_12km', color='crimson', linestyle='--')
            ax_r.step(t3, history['r_count_4km'], where='post', label='Red _count_4km', color='crimson')
            ax_r.set_ylabel('Red count')
            ax_r.set_title(f'Test vs Rule {rule_num}: 双方导弹威胁穿越计数 (_count_12km / _count_4km)')
            ax_r.legend(loc='upper left')
            ax_r.grid(True, alpha=0.3)

            ax_b.step(t3, history['b_count_12km'], where='post', label='Blue _count_12km', color='royalblue', linestyle='--')
            ax_b.step(t3, history['b_count_4km'], where='post', label='Blue _count_4km', color='royalblue')
            ax_b.set_xlabel('Time (s)')
            ax_b.set_ylabel('Blue count')
            ax_b.legend(loc='upper left')
            ax_b.grid(True, alpha=0.3)

            fig3.tight_layout()

            plt.show()

            # input("Press Enter to continue to the next test...")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

