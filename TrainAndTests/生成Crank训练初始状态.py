'''
训练左crank策略：
红方纯追踪蓝方，蓝方一开始就发射一枚导弹，然后练习crank机动

1、目标初始化前首先计算导弹可发射区范围，然后将目标置于可发射区内、不可逃逸区外，对我机纯追踪
2、目标速度和高度为随机数,与我机同高度
3、蓝方只有一枚导弹，开始就发射导弹，后续需保持雷达照射

'''


import argparse
import time
import sys
import os
sys.path.append(project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from Envs.battle6dof1v1_missile0919 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
import matplotlib
import json
import glob
import copy
import socket
import threading
from send2tacview import *
from Algorithms.Rules import decision_rule
from Math_calculates.CartesianOnEarth import *
from Math_calculates.sub_of_angles import *
from torch.distributions import Normal
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Envs.UAVmodel6d import UAVModel
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Visualize.tacview_visualize import *
from Visualize.tensorboard_visualize import *
from Algorithms.Wasted.PPOcontinues_dual_a_out import *
from tqdm import tqdm
from LaunchZone.calc_DLZ import *

use_tacview = 0  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=130,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# # 超参数
# actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
# critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# num_episodes = 3 # 1000
# hidden_dim = [128, 128, 128]  # 128
# gamma = 0.9
# lmbda = 0.9
# epochs = 10  # 10
# eps = 0.2
# pre_train_rate = 0 # 0.25 # 0.25
# k_entropy = 0.01 # 熵系数
mission_name = 'LCrank'

env = Battle(args, tacview_show=use_tacview)
# r_obs_spaces = env.get_obs_spaces('r') # todo 子策略的训练不要用这个
# b_obs_spaces = env.get_obs_spaces('b')
# r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
# action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])

# state_dim = 35 # len(b_obs_spaces)
# action_dim = b_action_spaces[0].shape[0]

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 整一下高度-攻击区离散表（使用有限的选择）
# 水平距离20km~80km, 我机高度 env.min_alt_safe 到 env.max_alt_safe 按 2e3 间隔划分
# 目标高度 = 我机高度-2e3, 0, 2e3, 双方速度均为1.2Ma，最后构建一个[我机高度, 目标高度, 不可逃逸区边界，最大边界]的查询表

init_states = []
start_time = time.time()
for ego_height in np.arange(env.min_alt_safe, env.max_alt_safe+1, 2e3):
    enm_heights = [ego_height-2e3, ego_height, ego_height+2e3]
    for enm_height in enm_heights:
        if enm_height<3e3 or enm_height>13e3:
            enm_heights.remove(enm_height)
    for enm_height in enm_heights:
        far_edges, _ = crank_pre_calculation(5e3, 100e3, 1e3, ego_height, enm_height, v_carrier=1.2*340, v_target=1.2*340)
        init_states.append([ego_height, enm_height, min(far_edges), max(far_edges)])

initial_states = np.array(init_states)
print("原始数据\n")
print(initial_states)

# ------------------------------

data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)
save_path = os.path.join(data_dir, "Crankinitial_states.npy")


# 保存（无损，推荐）
np.save(save_path, initial_states)               # 保存单个数组（精确可还原）
# 读取：
loaded = np.load(save_path)
print("读取的数据\n", loaded)
# loaded 与 initial_states 相同（dtype, shape, values）

# 如需保存多个数组
# np.savez("data/all_inits.npz", initial_states=initial_states, other=other_array)
# zipped = np.load("data/all_inits.npz"); arr = zipped['initial_states']

# 压缩版（节省磁盘）
# np.savez_compressed("data/initial_states_comp.npz", initial_states=initial_states)

# ------------------------------
# 可直接查看的文本格式（CSV），适合二维数值数据
csv_path = os.path.join(data_dir, "Crankinitial_states.csv")
import pandas as pd
pd.DataFrame(initial_states, columns=["ego_h","enm_h","far_min","far_max"][:initial_states.shape[1]]).to_csv("data/Crankinitial_states.csv", index=False)

# 读取 CSV
df = pd.read_csv(csv_path)
loaded_from_csv = df.values  # 转回 numpy array（注意 dtype 可能为 float64）
print("读取的数据\n", loaded_from_csv)
# 或者用 numpy.savetxt（更简单，但无列名）
# np.savetxt("data/initial_states.txt", initial_states, delimiter=",", fmt="%.6f")
# loaded_txt = np.loadtxt("data/initial_states.txt", delimiter=",")

# ------------------------------
# 注意：
# - .npy / .npz 精确、快速、推荐用于程序间传输或中间文件。
# - CSV/ TXT 便于人工查看和调试，但可能改变精度/类型信息。
# - 若数组包含 Python 对象或嵌套结构，需要 allow_pickle=True 加载（security risk）。
# ------------------------------

