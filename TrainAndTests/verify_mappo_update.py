import os, sys, pickle
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Algorithms'))  # Utils.py 里 from _context import * 需要

from Algorithms.ActorWithPointingNet import ActorWrapper
from Algorithms.CriticWrapper import critic_wrapper
from Algorithms.MAPPOHybrid import MAPPOHybrid


def index_tree(tree, idx):
    if isinstance(tree, dict):
        return {k: index_tree(v, idx) for k, v in tree.items()}
    if isinstance(tree, np.ndarray):
        return tree[idx]
    return tree


if __name__ == '__main__':
    with open(os.path.join(project_root, 'acmi_records', 'agent1_local_trans.pkl'), 'rb') as f:
        data = pickle.load(f)

    # pkl 保存时已完成堆叠，本身就是 dict-of-arrays，直接用
    N = 128  # 截前 128 条做冒烟测试
    transition = {k: index_tree(v, np.arange(N)) for k, v in data.items()}

    ai = transition['obs']['agent_input']
    fixed_dim = ai['fixed'].shape[1]
    enm_shape = ai['enm'].shape[1:]
    ally_shape = ai['ally'].shape[1:]
    print(f'fixed_dim={fixed_dim}, enm_shape={enm_shape}, ally_shape={ally_shape}')

    device = torch.device('cpu')
    actor = ActorWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)
    critic = critic_wrapper(fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device)

    agent = MAPPOHybrid(actor, critic, actor_lr=1e-4, critic_lr=1e-3,
                        lmbda=0.95, epochs=2, eps=0.2, gamma=0.99, device=device)

    metrics = agent.update(transition, adv_normed=True, mini_batch_size=64)
    print('update 完成，指标：')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    # 再验证外部预计算 advantages/td_targets 分支
    with torch.no_grad():
        vals = critic.get_value(transition['states'])['value'].numpy()
    transition2 = dict(transition)
    transition2['advantages'] = np.zeros(N, dtype=np.float32)
    transition2['td_targets'] = vals.astype(np.float32)
    metrics2 = agent.update(transition2, mini_batch_size=64)
    print('预计算 advantage 分支 update 完成：')
    for k, v in metrics2.items():
        print(f'  {k}: {v}')
    print('ALL OK')
