'''
遍历机动样式，划分进程计算5个机动样式
'''

import numpy as np
from math import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Envs.MissileModel1112 import * # MissileModel1


g = 9.81

dt = 0.2
t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180


# 提取最后一行
def latest(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[-1]  # 直接返回最后一行

from LaunchZone.calc_hit_points_maneuver_from_RWR import *

if __name__ == '__main__':
    p_carrier_ = np.array([0, 10000, 0], dtype='float64')
    v_carrier_ = np.array([320*1.2, 0, 0], dtype='float64')
    p_target_ = np.array([75e3, 10000, 0e3], dtype='float64')
    v_target_ = np.array([-320*1.2, 0, 0], dtype='float64')

    move_patterns = np.array(range(5)) + 1

    # 变步长解算导弹和目标的运动直到目标被命中或导弹超时/速度过低/错过目标

    import time
    # 串行计算时间
    start_time = time.time()
    for run in range(1):  # 算100次需要多久
        print("run", run + 1)
        hit_counts = 0
        for move_pattern in move_patterns:
            # print("pattern", move_pattern)
            hit = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                          show=0)
            if hit:
                hit_counts += 1

        output = hit_counts / len(move_patterns)  # 可命中比例
        print(output)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"串行程序运行时长: {execution_time} 秒")

    # 并行计算
    from joblib import Parallel, delayed
    t0 = time.time()
    move_patterns = move_patterns.tolist()
    move_patterns*=1
    # 定义并行任务
    parallel_tasks = [
        delayed(sim_hit)(
            p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1, show=0
        )
        for move_pattern in move_patterns
    ]

    # 执行并行任务
    hits = Parallel(n_jobs=1)(parallel_tasks)

    t1 = time.time()
    print(sum(hits)/len(hits))
    print(f"并行程序运行时长: {t1-t0} 秒")
