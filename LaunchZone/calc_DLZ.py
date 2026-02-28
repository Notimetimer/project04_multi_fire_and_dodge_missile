'''
根据二分法计算动态可发射距离
'''
import numpy as np
from math import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LaunchZone.calc_hit_points_maneuver_from_RWR import *
from Envs.MissileModel1112 import *
from joblib import Parallel, delayed

# 继续沿用导弹模型的命中/miss计算
g = 9.81
theta_limit = 85 * pi / 180

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

def horizontal_shoot_range(a, b, epsilon, height_ego, height_enm, v_carrier_, v_target_, move_pattern):
    # a,b,epsilon 现在是水平距离，斜距还得另外算

    p_carrier_ = np.array([0, height_ego, 0], dtype='float64')

    calc_times = 0
    break_flag = 0
    next_b_hit = None
    while b - a >= epsilon and not break_flag:
        calc_times+=1
        lambda1 = (a+b)/2
        p_target_ = np.array([lambda1, height_enm, 0], dtype='float64')
        hit1 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                        show=0) # lambda
        if next_b_hit is None:
            calc_times+=1
            p_target_ = np.array([b, height_enm, 0], dtype='float64')
            hit2 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                        show=0) # 运行b处计算
        else:
            hit2 = next_b_hit

        if hit2:
            break_flag = 1
        elif hit1:
            a = lambda1
            next_b_hit = hit2
        else:
            b = lambda1
            next_b_hit = hit1
    far_edge = (a+b)/2 if not break_flag else b
    return far_edge, calc_times


def crank_pre_calculation(a, b, epsilon, height_ego, height_enm, v_carrier, v_target):
    v_carrier_ = np.array([v_carrier, 0, 0], dtype='float64')
    v_target_ = np.array([-v_target, 0, 0], dtype='float64')
    # 并行计算
    move_patterns = np.array(range(5)) + 1
    move_patterns = move_patterns.tolist()
    parallel_tasks = [
        delayed(horizontal_shoot_range)(
            a, b, epsilon, height_ego, height_enm, v_carrier_, v_target_, move_pattern)
        for move_pattern in move_patterns
    ]
    # 执行并行任务
    far_edges_and_times = Parallel(n_jobs=5)(parallel_tasks)
    far_edges = []
    times = []
    for far_edge, calc_time in far_edges_and_times:
        far_edges.append(far_edge)
        times.append(calc_time)
    return far_edges, times


if __name__ == '__main__':
    # 假设我机和目标机都处在
    height_ego = 10e3
    height_enm = 10e3
    v_carrier = 320*1.2
    v_target = 320*1.2
    b = 80e3
    a = 5e3
    epsilon = 1e3
    import time
    t0 = time.time()

    # p_carrier_ = np.array([0, height_ego, 0], dtype='float64')
    # v_carrier_ = np.array([v_carrier, 0, 0], dtype='float64')
    # v_target_ = np.array([-v_target, 0, 0], dtype='float64')
    # move_patterns = np.array(range(5)) + 1
    
    # # 并行计算  
    # move_patterns = move_patterns.tolist()
    # parallel_tasks = [
    #     delayed(horizontal_shoot_range)(
    #         a, b, epsilon, height_ego, height_enm, v_carrier_, v_target_, move_pattern)
    #     for move_pattern in move_patterns
    # ]
    # # 执行并行任务
    # far_edges_and_times = Parallel(n_jobs=5)(parallel_tasks)
    # far_edges = []
    # times = []
    # for far_edge, calc_time in far_edges_and_times:
    #     far_edges.append(far_edge)
    #     times.append(calc_time)
    
    far_edges, times = crank_pre_calculation(a, b, epsilon, height_ego, height_enm, v_carrier, v_target)
    
    t1 = time.time()
    print(far_edges)
    print(times)
    print(f"并行程序运行时长: {t1-t0} 秒")

    # # 串行计算攻击区
    # start_time = time.time()
    # far_edge = np.zeros_like(move_patterns)
    # calc_times = np.zeros_like(move_patterns)
    # height_ego = 10e3
    # height_enm = 10e3
    # for i, move_pattern in enumerate(move_patterns):
    #     b = 80e3
    #     a = 5e3
    #     epsilon = 1e3
    #     break_flag = 0

    #     next_b_hit = None
    #     while b - a >= epsilon and not break_flag:
    #         calc_times[i]+=1
    #         lambda1 = (a+b)/2
    #         p_target_ = np.array([lambda1, height_enm, 0], dtype='float64')
    #         hit1 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
    #                       show=0) # lambda
    #         if next_b_hit is None:
    #             calc_times[i]+=1
    #             p_target_ = np.array([b, height_enm, 0], dtype='float64')
    #             hit2 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
    #                       show=0) # 运行b处计算
    #         else:
    #             hit2 = next_b_hit

    #         if hit2:
    #             break_flag = 1
    #         elif hit1:
    #             a = lambda1
    #             next_b_hit = hit2
    #         else:
    #             b = lambda1
    #             next_b_hit = hit1
        
    #     far_edge[i] = (a+b)/2 if not break_flag else b

    # print(far_edge)
    # print(calc_times)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"串行程序运行时长: {execution_time} 秒") 

