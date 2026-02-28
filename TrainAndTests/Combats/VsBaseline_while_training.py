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
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger

# ==========================================
# [修改] 并行测试 Worker 函数 (增加了 dt_maneuver_val 参数)
# ==========================================
def test_worker(model_state_dict, rule_num, 
                env_args, state_dim, hidden_dim, 
                action_dims_dict, dt_maneuver_val, 
                device_name='cpu', num_runs=1, action_cycle_multiplier=30):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    在独立进程中运行一场对战。
    """
    device = torch.device(device_name)
    
    # 1. 局部初始化环境 (必须在子进程内创建)
    # 关闭渲染以节省资源
    test_env = ChooseStrategyEnv(env_args, tacview_show=0)
    test_env.shielded = 1
    test_env.dt_move = 0.05
    test_env.dt_maneuver = dt_maneuver_val # 使用传入的值，不依赖全局变量
    
    # 2. 局部初始化网络并加载权重
    net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(net, action_dims_dict, None, device).to(device)
    actor.load_state_dict(model_state_dict)
    actor.eval() # 设置为评估模式
    action_cycle = action_cycle_multiplier
    
    # 3. 运行对战逻辑
    result = 0
    for _ in range(num_runs):
        test_env.reset(red_init_ammo=6, blue_init_ammo=6)
        
        steps = 0
        done = False
        
        b_action_label = 0
        
        while not done and test_env.running:
            # 获取观测
            r_obs, r_check = test_env.obs_1v1('r', pomdp=1)
            b_obs, _ = test_env.obs_1v1('b', pomdp=1)
            
            # 决策点
            if steps % action_cycle == 0:
                # 红方 (规则)
                r_state_check = test_env.unscale_state(r_check)
                r_action_label, r_fire = basic_rules(r_state_check, rule_num)
                if r_fire: launch_missile_immediately(test_env, 'r')
                
                # 蓝方 (神经网络 - 无法使用确定性决策，会导致测试回合与训练回合呈现巨大的性能差别)
                with torch.no_grad():
                    # [修复] 调用 actor.get_action 而不是 take_action
                    # get_action 返回 4 个值: actions_exec, actions_raw, h_state, actions_dist_check
                    b_act_exec, _, _, _ = actor.get_action(b_obs, explore={'cont':1, 'cat':1, 'bern':1})
                    b_action_label = b_act_exec['cat'][0]
                    if b_act_exec['bern'][0]: launch_missile_immediately(test_env, 'b')

            # 物理步
            r_maneuver = test_env.maneuver14LR(test_env.RUAV, r_action_label)
            b_maneuver = test_env.maneuver14LR(test_env.BUAV, b_action_label)
            test_env.step(r_maneuver, b_maneuver)
            
            # 判定
            done, _, _, _ = test_env.combat_terminate_and_reward('b', b_action_label, False, action_cycle)
            steps += 1
            
            if steps * dt_maneuver_val > env_args.max_episode_len: break

        # 返回结果：1 赢, 0 输, 0.5 平
        if test_env.win:
            result += 1/num_runs
        elif test_env.lose:
            result += 0/num_runs
        else:
            result += 0.5/num_runs

    return rule_num, result

