import os, sys
import json
import copy
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


# rule_input 只保留 RWR/weapon/enm 三块，统一后续堆叠为 dict-of-array
def take_some_of_rule_inputs(rule_input):
    # return rule_input
    return copy.deepcopy({
        "RWR_infos":    rule_input["RWR_infos"],
        "weapon_infos": rule_input["weapon_infos"],
        "enm_infos":    rule_input["enm_infos"],
        "ammo":         rule_input["ego_ammo"],
    })


# 递归堆叠各字段为 dict-of-array 或 array（单样本时用 [单步值] 堆出 B=1）
def stack_field(field_list):
    """递归堆叠：list[单步值] -> dict-of-array 或 array。
    None 自动跳过；dict 对每个 key 递归；
    list-of-dict 按字段堆叠为 [B, N]；
    list-of-scalar / scalar 堆叠为 array。"""
    sample = next((v for v in field_list if v is not None), None)
    if sample is None:
        return None
    if isinstance(sample, dict):
        all_keys = set()
        for item in field_list:
            if item is not None:
                all_keys.update(item.keys())
        return {k: stack_field([item.get(k) if item is not None else None
                                for item in field_list])
                for k in all_keys}
    elif isinstance(sample, list):
        if sample and isinstance(sample[0], dict):
            keys = sample[0].keys()
            return {k: np.array([[d.get(k) for d in step] for step in field_list]) for k in keys}
        else:
            return np.array(field_list)
    else:
        return np.array(field_list)


# 网络初始化统一从 json 读观测维度（由 读取环境参数并存json.py 生成），
# 不再每次初始化一个小环境去探测状态维度
def load_obs_dims(json_path=None):
    path = json_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env_obs_dims.json')
    with open(path, 'r', encoding='utf-8') as f:
        dims = json.load(f)
    return dims['fixed_dim'], tuple(dims['enm_shape']), tuple(dims['ally_shape'])


# 初始飞机数量和导弹挂载数统一从 init_states.json 读取（各训练/测试脚本共用一份配置）
def load_init_states(json_path=None):
    path = json_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_states.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_experience(td, obs, state, action, reward, next_state, done, active_mask):
    """
    统一把经验追加到 transition_dict。

    """
    td['obs'].append(obs) # 新增：存储Actor用的局部观测
    td['states'].append(state) # 修改：这里存储Critic用的全局状态
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    td['active_masks'].append(active_mask) # 【新增】active_mask，转入多智能体
    return td


# 递归合并两个 dict-of-arrays 经验树（worker 端已堆叠为 numpy）：空槽位按源结构初始化，
# 非空叶子沿 batch 维（axis=0）拼接。注意 dict 字段不能用 list.extend——extend 会误存键名
def merge_transition_tree(dst, src):
    if isinstance(src, dict):
        merged = dst if isinstance(dst, dict) else {}
        for k, sv in src.items():
            merged[k] = merge_transition_tree(merged.get(k), sv)
        return merged
    arr = np.asarray(src)
    if dst is None or (isinstance(dst, list) and not dst):
        return arr.copy()
    return np.concatenate([np.asarray(dst), arr], axis=0)

if __name__=='__main__':
    sys.path.append(project_root)
