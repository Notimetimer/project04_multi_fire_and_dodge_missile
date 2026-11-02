"""ActorWrapper 计算图可视化：同时导出 TensorBoard 事件和 Netron 可打开的 ONNX。

用法：
    python scratch_vis.py            # 同时生成 runs/ 和 actor.onnx
    python scratch_vis.py --tb       # 仅 TensorBoard
    python scratch_vis.py --onnx     # 仅 ONNX (Netron)
然后：
    tensorboard --logdir runs/actor_graph
    netron actor.onnx   或浏览器打开 https://netron.app
"""
import argparse
import sys

import numpy as np
import torch

from Algorithms.ActorWithATTN import ActorWrapper


# ONNX / TensorBoard 都只接受位置参数(tuple)，且不能有动态控制流，
# 所以用一个 trace 专用的薄包装：把 tuple 还原成 dict，并固定走"评估模式"
# （传入已知 enm/ally 索引与 hor/vert/fire 动作，跳过 sample() 中的随机算子）。
class TraceWrapper(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, fixed, enm, ally, enm_mask, ally_mask,
                enm_idx, hor, vert, fire):
        # 直接调用 selector / pilot，绕过 pack_inputs 中的 numpy 转换
        # （trace/onnx 要求纯 tensor 路径）
        packed_agent = {'fixed': fixed, 'enm': enm, 'ally': ally}
        packed_mask = {'enm': enm_mask, 'ally': ally_mask}
        selected_info, enm_idx_o, mini_rule_inputs, log_prob_enm, ent_enm = \
            self.actor.selector(packed_agent, packed_mask, {},
                                enm_idx=enm_idx)
        # pilot 只输出 logits，不含随机算子，可直接 trace
        pilot_logits = self.actor.pilot(selected_info, mini_rule_inputs)
        return (
            pilot_logits['hor'], pilot_logits['vert'], pilot_logits['fire'],
            enm_idx_o, log_prob_enm,
        )


def make_dummy(batch=2, enm_shape=(3, 10), ally_shape=(2, 9), fixed_dim=28):
    return {
        'fixed': torch.randn(batch, fixed_dim),
        'enm': torch.randn(batch, *enm_shape),
        'ally': torch.randn(batch, *ally_shape),
    }, {
        'enm': torch.ones(batch, enm_shape[0], dtype=torch.bool),
        'ally': torch.ones(batch, ally_shape[0], dtype=torch.bool),
    }, {
        'enm_idx': torch.zeros(batch, dtype=torch.long),
        'hor': torch.zeros(batch, dtype=torch.long),
        'vert': torch.zeros(batch, dtype=torch.long),
        'fire': torch.zeros(batch, dtype=torch.long),
    }


def export_tensorboard(trace_model, agent, mask, actions, log_dir='runs/actor_graph'):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    dummy = (
        agent['fixed'], agent['enm'], agent['ally'],
        mask['enm'], mask['ally'],
        actions['enm_idx'],
        actions['hor'], actions['vert'], actions['fire'],
    )
    # add_graph 对 dict 输入/动态图支持有限，用 trace wrapper 规避
    writer.add_graph(trace_model, dummy)
    writer.close()
    print(f'[TensorBoard] 计算图已写入 {log_dir}/')
    print(f'  查看: tensorboard --logdir {log_dir}')


def export_onnx(trace_model, agent, mask, actions, path='actor.onnx'):
    trace_model.eval()
    dummy = (
        agent['fixed'], agent['enm'], agent['ally'],
        mask['enm'], mask['ally'],
        actions['enm_idx'],
        actions['hor'], actions['vert'], actions['fire'],
    )
    input_names = [
        'fixed', 'enm', 'ally', 'enm_mask', 'ally_mask',
        'enm_idx', 'hor', 'vert', 'fire',
    ]
    output_names = [
        'hor_logits', 'vert_logits', 'fire_logits',
        'enm_idx', 'log_prob_enm',
    ]
    dynamic_axes = {name: {0: 'batch'} for name in input_names + output_names}
    # dynamo=False 使用 legacy torch.jit.trace，能处理 torch.distributions
    torch.onnx.export(
        trace_model, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17,
        do_constant_folding=False, dynamo=False,
    )
    print(f'[ONNX] 已导出 {path}')
    print(f'  查看: netron {path}  或  https://netron.app')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb', action='store_true', help='仅导出 TensorBoard')
    parser.add_argument('--onnx', action='store_true', help='仅导出 ONNX')
    args = parser.parse_args()
    do_tb = args.tb or not args.onnx
    do_onnx = args.onnx or not args.tb

    actor = ActorWrapper()
    actor.eval()
    trace_model = TraceWrapper(actor)
    agent, mask, actions = make_dummy()

    if do_tb:
        export_tensorboard(trace_model, agent, mask, actions)
    if do_onnx:
        export_onnx(trace_model, agent, mask, actions)


if __name__ == '__main__':
    main()
