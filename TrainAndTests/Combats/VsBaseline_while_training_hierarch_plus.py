"额外输出双杀率"

import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time  # 确保引入 time 模块
from datetime import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new_hierarchical import *
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper, infer_mask_cfg_from_state_dict
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger

# ==========================================
# [修改] 并行测试 Worker 函数 (增加了 dt_maneuver_val 参数)
# ==========================================
def test_worker(model_state_dict, rule_num,
                env_args, state_dim, hidden_dim,
                action_dims_dict, dt_maneuver_val,
                device_name='cpu', num_runs=1, action_cycle_multiplier=10,
                no_out=0, deterministic=False, restrict_fire=False, vertices=None, auto_regressive=0, Temperature=None,
                red_init_ammo=6, blue_init_ammo=6):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    在独立进程中运行一场对战。
    """
    device = torch.device(device_name)

    env_args.R_cage = 62.00e3
    env_args.max_episode_len = 15*60
    
    # 1. 局部初始化环境 (必须在子进程内创建)
    # 关闭渲染以节省资源
    test_env = ChooseStrategyEnv(env_args, tacview_show=0, vertices=vertices)
    test_env.shielded = 1
    test_env.no_out = no_out
    test_env.dt_move = 0.07 # 0.04
    test_env.dt_maneuver = dt_maneuver_val # 使用传入的值，不依赖全局变量
    
    # 2. 局部初始化网络并加载权重
    # [新增] 根据传入的 state_dict 自动推断训练时的 ver/hor 配置
    mask_cfg = infer_mask_cfg_from_state_dict(model_state_dict)
    net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device)
    actor = HybridActorWrapper(net, action_dims_dict, None, device).to(device)
    actor.load_state_dict(model_state_dict, strict=False)
    actor.eval() # 设置为评估模式
    action_cycle = 10 # 30 action_cycle_multiplier 锁死测试回合的动作间隔，便于做课程学习
    # 间隔2s输出一次，这是课程学习的最终决策步长。有没有真功夫就看2s决策一次能不能做好了，双方都会是2s一次决策机会
    
    # 3. 运行对战逻辑
    result = 0
    result2 = 0
    wins = 0
    loses = 0
    draws = 0
    BVR_perish_togethers = 0

    for _ in range(num_runs):
        test_env.reset(red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo, pomdp=1) # 0
        
        steps = 0
        done = False
        
        b_action_label = 0
        episode_return = 0

        while not done and test_env.running:
            # 获取观测
            r_obs, r_check = test_env.obs_1v1('r', pomdp=1)
            b_obs, b_check = test_env.obs_1v1('b', pomdp=1)
            
            # 决策点
            if steps % action_cycle == 0:
                # 红方 (规则)
                r_state_check = test_env.unscale_state(r_check)
                r_action_label, r_fire = basic_rules(r_state_check, rule_num)
                if r_fire: test_env.RUAV.about_to_fire = 1
                
                # 蓝方 (神经网络)
                with torch.no_grad():
                    # 如果 deterministic 为 True，则机动(cat)采用确定性决策，开火(bern)仍保持随机(1)
                    if deterministic:
                        if Temperature is None:
                            Temperature = {'cat':0.5, 'bern':1}
                        explore_dict = {'cont': 0, 'cat': 1, 'bern': 1} # 'cat': 0
                        # [漏网点1已解决]: 测试一律传入 check_obs=b_check，激活 _deploy_can_fire 规则校验
                        b_act_exec, _, _, _ = actor.get_action(b_obs, explore=explore_dict, temperature=Temperature, check_obs=b_check)
                    else:
                        if Temperature is None:
                            Temperature = {'cat':1, 'bern':1}
                        explore_dict = {'cont': 1, 'cat': 1, 'bern': 1}
                        # [漏网点1已解决]: 随机测试也一律传入 check_obs=b_check，消除开火硬规则差异
                        b_act_exec, _, _, _ = actor.get_action(b_obs, explore=explore_dict, temperature=Temperature, check_obs=b_check)

                    b_action_label = b_act_exec['cat'] # [0]
                    if b_act_exec['bern'][0]:
                        # [漏网点2 解决]: 物理层禁忌校验与开火一致性
                        # 两种测试方式在物理层均统一加上 tabu=1 (强制要求 target_locked, weapon>=0.1, ATA<=max_radar_angle_rad)。
                        # 无论网络意图如何，都必须通过统一的底层物理雷达锁定条件校验才允许射出导弹。
                        test_env.BUAV.about_to_fire = 1

            # 尝试发射 (两种测试方式均强制启用 tabu=1)
            tabu_fire = 1
            if getattr(test_env.RUAV, 'about_to_fire', 0):
                # 如果 restrict_fire 为 True，则限制动作次序（传入 r_action_label）
                r_act_label_to_pass = r_action_label if (restrict_fire or auto_regressive) else None
                launch_missile_immediately(test_env, 'r', action_label=r_act_label_to_pass, tabu=tabu_fire)
            b_m_id = None
            if getattr(test_env.BUAV, 'about_to_fire', 0):
                # 如果 restrict_fire 为 True，则限制动作次序（传入 b_action_label）
                b_act_label_to_pass = b_action_label if (restrict_fire or auto_regressive) else None
                b_m_id = launch_missile_immediately(test_env, 'b', action_label=b_act_label_to_pass, tabu=tabu_fire)

            # 物理步
            r_maneuver = test_env.maneuver14LR(test_env.RUAV, r_action_label)
            b_maneuver = test_env.maneuver14LR(test_env.BUAV, b_action_label)
            test_env.step(r_maneuver, b_maneuver)
            
            # 判定
            done, b_reward1, b_reward2, b_reward3 = test_env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle)
            steps += 1

            if steps % action_cycle == 0:
                episode_return += b_reward1

            if steps * dt_maneuver_val > env_args.max_episode_len: break

        WVR = test_env.close_range_kill()
        # 双杀必须要
        BVR_perish_together = (not WVR) and test_env.draw and test_env.t <= env_args.max_episode_len - 10

        # 返回结果：1 赢, 0 输, 0.5 平
        if test_env.win:
            result += 1/num_runs
            wins += 1/num_runs
        elif test_env.lose:
            result += 0/num_runs
            loses += 1/num_runs
        else:
            result += 0.5/num_runs
            draws += 1/num_runs
        result2 += episode_return /num_runs

        BVR_perish_togethers += BVR_perish_together/num_runs

        
        
    return rule_num, result, result2, wins, loses, draws, BVR_perish_togethers

