"""监控 HAN-SAC 训练曲线：从 run 目录的 tensorboard 事件提取
WinRate（train/2 win、train_plus/batch_score、test/avg_score）与
buffer_ended / buffer_size，绘制并保存 PNG 到该 run 目录。

用法（conda 环境 38）：
  python monitor_hansac_training.py                 # 单次生成，自动找最新 HanSAC run
  python monitor_hansac_training.py <run_dir>       # 指定 run 目录
  python monitor_hansac_training.py --watch 300     # 每 300 秒自动刷新
"""
import os
import sys
import glob
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'logs', 'combat')

TAGS = {
    'train_win': 'train/2 win',
    'batch_score': 'train_plus/batch_score',
    'test_score': 'test/avg_score',
    'buffer_ended': 'agent/buffer_ended',
    'buffer_size': 'agent/buffer_size',
}


def find_latest_run(prefix='HanSAC'):
    runs = [d for d in glob.glob(os.path.join(LOGS_ROOT, f'{prefix}*-run-*'))
            if os.path.isdir(d)]
    return max(runs, key=os.path.getmtime) if runs else None


def load_scalars(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    out = {}
    tags = ea.Tags().get('scalars', [])
    for key, tag in TAGS.items():
        if tag in tags:
            ev = ea.Scalars(tag)
            out[key] = ([e.step for e in ev], [e.value for e in ev])
    return out


def plot(run_dir, out_png=None):
    data = load_scalars(run_dir)
    if not data:
        print(f'[monitor] no scalars yet in {run_dir}')
        return
    if out_png is None:
        out_png = os.path.join(run_dir, 'monitor_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key, color in (('train_win', 'tab:green'), ('batch_score', 'tab:orange'),
                       ('test_score', 'tab:purple')):
        if key in data:
            s, v = data[key]
            ax.plot(s, v, marker='.', ms=3, lw=1, label=key, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('WinRate')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if 'buffer_size' in data:
        s, v = data['buffer_size']
        ax2.plot(s, v, lw=1, label='buffer_size', color='tab:blue')
    if 'buffer_ended' in data:
        s, v = data['buffer_ended']
        ax2.plot(s, v, lw=1, label='buffer_ended', color='tab:red')
    ax2.set_title('Replay buffer (ended = terminated+truncated)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    for a in axes:
        a.set_xlabel('decision steps')
    fig.suptitle(os.path.basename(run_dir))
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'[monitor] saved {out_png}')


if __name__ == '__main__':
    args = sys.argv[1:]
    watch_sec = None
    run_dir = None
    if '--watch' in args:
        i = args.index('--watch')
        watch_sec = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        run_dir = args[0]

    while True:
        target = run_dir or find_latest_run()
        if target is None:
            print(f'[monitor] no HanSAC run found under {LOGS_ROOT}')
        else:
            plot(target)
        if watch_sec is None:
            break
        time.sleep(watch_sec)
