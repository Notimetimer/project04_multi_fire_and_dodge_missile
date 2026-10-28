
from typing import final
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
import traceback # [新增]
import random
import collections

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None,
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta',
                               compete_old_rate=0.5, deltaFSP_epsilon=0.5,):
    """
    优化后的对手采样逻辑：
    1. 优先判定是否进入“规则复习”分支。
    2. 若未进入，则根据 SP_type 执行具体的采样策略。
    """
    # 【核心修改】在函数内部合并出一个临时的全集字典用于查询分数
    # 这样 keys 里的任何元素都能在这里找到对应的 ELO
    
    if hall_of_fame is not None:
        candidate_pool = hall_of_fame.copy()
        candidate_pool.update(elite_elo_ratings)
    else:
        candidate_pool = elite_elo_ratings
    keys = list(candidate_pool.keys())
    
    if not keys: return np.array([]), []

    # --- 第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护) ---
    # 只要 compete_old_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < compete_old_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    # --- 第二层判断：进入核心采样逻辑 ---
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = min(np.max(elos), float(target_elo) + 300)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
            # # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            # actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
        diffs = elos - actual_target
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
        return probs, keys

    # 2. 处理 FSP (全样本均匀分布)
    elif SP_type == 'FSP':
        probs = np.ones(len(keys)) / len(keys)
        return probs, keys

    # 3. 处理 deltaFSP (新旧池切分)
    elif SP_type == 'deltaFSP':
        n = len(keys)
        new_count = max(1, int(np.ceil(n * 0.2)))
        new_keys = keys[-new_count:]
        old_keys = keys[:-new_count]
        
        # 这里的 deltaFSP_epsilon 建议直接作为参数传入或使用全局变量
        if np.random.rand() < float(deltaFSP_epsilon) or not old_keys:
            target_keys = new_keys
        else:
            target_keys = old_keys
            
        probs = np.ones(len(target_keys)) / len(target_keys)
        return probs, target_keys

    # 4. 处理 SP (最新历史版本)
    elif SP_type == 'SP':
        # rein_keys = [k for k in keys if k.startswith('actor_rein') and '_step_' not in k]
        # 严格匹配 actor_rein + 数字
        rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            # try: return int(k.replace('actor_rein', ''))
            # except: return -1
            try: return int(re.search(r'actor_rein(\d+)', k).group(1))
            except: return -1
            
        best_key = max(rein_keys, key=extract_number)
        return np.array([1.0]), [best_key]

    # 5. 兜底逻辑: Rule 均匀采样 (None)
    else:
        if not rule_keys: return np.array([]), []
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
