"""读取环境观测维度并存 env_obs_dims.json：
初始化网络时统一读取该 json（_context.load_obs_dims），
不再每次初始化一个小环境去探测状态维度。
环境配置（边界/机型数量）变化后需重跑本脚本刷新 json。
"""

import os
import sys
import json
import argparse
import numpy as np
from math import pi

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]
    env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    env.reset(red_init_num=2, blue_init_num=2, red_init_ammo=2, blue_init_ammo=2)

    # 各机观测维度一致，用第一架红机的观测确定网络维度
    red_id = [i for i in env.UAV_ids if env.find_uav_with_id(i).side == 'r'][0]
    agent_input, _, _ = env.get_obs(red_id, pomdp=1)

    dims = {
        'fixed_dim': len(agent_input['fixed']),
        'enm_shape': list(agent_input['enm'].shape),
        'ally_shape': list(agent_input['ally'].shape),
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env_obs_dims.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dims, f, indent=2)
    print(f"观测维度已保存: {json_path}")
    print(dims)
