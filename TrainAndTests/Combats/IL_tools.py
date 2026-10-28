
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

# ==========================================
# 核心修复：数据结构重组函数
# ==========================================
def restructure_actions(actions_data):
    """
    将 list of dicts [{'fly': 1, 'fire': 0}, ...] 
    转换为 dict of arrays {'cat': array([[1],...]), 'bern': array([[0],...])}
    并确保维度是 (N, 1) 以适配 PPOHybrid2 的索引操作
    """
    # 如果已经是字典格式，直接返回
    if isinstance(actions_data, dict):
        return actions_data
    
    # 如果是列表且包含字典，进行转换
    if isinstance(actions_data, list) and len(actions_data) > 0:
        # print("Restructuring actions from List[Dict] to Dict[Array]...") # 频繁调用可注释掉以减少刷屏
        
        # 初始化容器
        new_actions = {'cat': [], 'bern': []}
        
        for item in actions_data:
            # 兼容处理：item 可能是 dict，也可能是包含 dict 的 numpy array
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item() # 提取 numpy 里的 dict
            
            # 映射 'fly' -> 'cat' (离散机动)
            # 映射 'fire' -> 'bern' (开关开火)
            if isinstance(act, dict):
                # 优先找 'fly'，找不到找 'cat'
                val_cat = act.get('fly', act.get('cat'))
                if val_cat is not None:
                    new_actions['cat'].append(val_cat)
                
                # 优先找 'fire'，找不到找 'bern'
                val_bern = act.get('fire', act.get('bern'))
                if val_bern is not None:
                    new_actions['bern'].append(val_bern)

            # 备用：如果数据意外变成了 list/tuple
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                 new_actions['cat'].append(act[0])
                 new_actions['bern'].append(act[1])

        # 转换为 Numpy Array 并调整形状为 (Batch, 1)
        # 这一点至关重要：PPOHybrid2 里的 expert_cat[:, i] 需要 expert_cat 是二维的
        
        # 1. 'cat': 离散动作，转为 int64，Reshape 为 (N, 1)
        cat_arr = np.array(new_actions['cat'], dtype=np.int64)
        if cat_arr.ndim == 1:
            cat_arr = cat_arr.reshape(-1, 1)
        
        # 2. 'bern': 伯努利动作，转为 float32 (BCE Loss需要)，Reshape 为 (N, 1)
        bern_arr = np.array(new_actions['bern'], dtype=np.float32)
        if bern_arr.ndim == 1:
            bern_arr = bern_arr.reshape(-1, 1)

        result = {
            'cat': cat_arr,
            'bern': bern_arr
        }
        
        print(f"Structure fixed: 'cat' shape={cat_arr.shape}, 'bern' shape={bern_arr.shape}")
        return result

    return actions_data


# ==========================================
# 新增：混合缓冲区类
# ==========================================
class IL_transition_buffer:
    def __init__(self, init_dict, max_size=10000):
        """
        内部存储全部采用 List，确保状态和动作在最外层长度绝对对等。
        """
        self.max_size = max_size
        self.addon_dict = {}
        
        # 无论 init_dict 是否为空，都显式初始化所有键，确保 'obs' 一定存在
        src = init_dict if init_dict is not None else {}
        
        # 强制转换成 list 存储。如果 src 里没有 'obs'，就用 'states' 代替
        self.addon_dict['obs'] = list(src.get('obs', src.get('states', [])))
        self.addon_dict['states'] = list(src.get('states', []))
        self.addon_dict['returns'] = list(src.get('returns', []))
        self.addon_dict['actions'] = list(src.get('actions', []))
        
        # 特殊处理：如果 returns 是 torch.tensor (如你截图中所示)，转为 list 存储
        if torch.is_tensor(src.get('returns')):
            self.addon_dict['returns'] = src['returns'].tolist()

        # 打印初始长度以供检查
        print(f"[IL_transition_buffer] Initialized. Size: {len(self.addon_dict['states'])}")
        
    def add(self, data):
        """
        data: 包含 'obs', 'states', 'actions', 'returns' 的字典，值应为 List。
        """
        # 1. 提取新数据并确保格式为 list (防止 data 缺失 'obs'，逻辑同 init)
        new_obs = list(data.get('obs', data.get('states', [])))
        new_states = list(data.get('states', []))
        new_returns = list(data.get('returns', []))
        new_actions = list(data.get('actions', []))
        
        # 处理可能传入的 tensor
        if torch.is_tensor(data.get('returns')):
            new_returns = data['returns'].tolist()
        
        # 使用 .extend() 拼接列表元素
        self.addon_dict['obs'].extend(new_obs)
        self.addon_dict['states'].extend(new_states)
        self.addon_dict['returns'].extend(new_returns)
        self.addon_dict['actions'].extend(new_actions)
        
        # 2. 基于添加顺序的剪裁（保留最后/最新的 max_size 条）
        current_len = len(self.addon_dict['states'])
        max_size_int = int(self.max_size)  # 确保 max_size 是整数
        if current_len > max_size_int:
            keep_from = current_len - max_size_int
            self.addon_dict['obs'] = self.addon_dict['obs'][keep_from:]
            self.addon_dict['states'] = self.addon_dict['states'][keep_from:]
            self.addon_dict['returns'] = self.addon_dict['returns'][keep_from:]
            self.addon_dict['actions'] = self.addon_dict['actions'][keep_from:]
    

    
    def read(self, batch_size):
        """
        随机采样并进行格式转换。
        """
        total_len = len(self.addon_dict['states'])
        if total_len == 0:
            raise ValueError("IL_transition_buffer is empty.")
            
        indices = np.random.randint(0, total_len, size=min(int(batch_size), total_len))
        
        # 采样（列表推导式，保持原始元素格式）
        sampled_obs = [self.addon_dict['obs'][i] for i in indices]
        sampled_states = [self.addon_dict['states'][i] for i in indices]
        sampled_returns = [self.addon_dict['returns'][i] for i in indices]
        sampled_actions_raw = [self.addon_dict['actions'][i] for i in indices]
        
        # 此时才将采样的动作列表转换为算法需要的 dict-of-arrays 格式
        return {
            'obs': np.array(sampled_obs, dtype=np.float32),
            'states': np.array(sampled_states, dtype=np.float32),
            'returns': np.array(sampled_returns, dtype=np.float32),
            'actions': restructure_actions(sampled_actions_raw) 
        }

    def clear(self):
        for k in self.addon_dict:
            self.addon_dict[k] = []
        print("[IL_transition_buffer] Buffer cleared.")

    def save(self, path):
        """直接序列化整个 buffer 实例"""
        torch.save(self, path)
        print(f"[IL_transition_buffer] Saved to {path}. Size: {len(self.addon_dict['states'])}")

    @staticmethod # 静态方法调用,不需要实例化
    def load(path):
        """从磁盘加载 buffer 实例"""
        if not os.path.exists(path):
            return None
        buffer = torch.load(path, map_location='cpu')
        print(f"[IL_transition_buffer] Loaded from {path}. Size: {len(buffer.addon_dict['states'])}")
        return buffer
