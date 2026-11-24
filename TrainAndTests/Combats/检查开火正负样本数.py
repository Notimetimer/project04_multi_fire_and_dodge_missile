'''
临时检查脚本：统计某个训练目录下 fire_records_buffer.pkl 中 hit_record 为 0 和为 1 的样本数。

用法：
    python check_fire_buffer.py [run_dir]
不传 run_dir 时检查脚本顶部 DEFAULT_RUN_DIR 指向的试次目录。
'''
import os
import sys
import pickle

DEFAULT_RUN_DIR = os.path.join(r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat",
    r"2APPO_homo_SLA_1_blackboard_1-run-20261122-212938")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    pkl_path = os.path.join(run_dir, 'fire_records_buffer.pkl')
    print(f'检查文件: {pkl_path}')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        records = obj.get('records', [])
        maxlen = obj.get('maxlen')
    else:
        records, maxlen = obj, None

    print(f'当前记录数 = {len(records)}')
    if not records:
        print('缓冲为空。')
        return

    pos = sum(1 for r in records if r.get('hit_record') == 1)
    neg = sum(1 for r in records if r.get('hit_record') == 0)
    print(f'hit_record=1: {pos}')
    print(f'hit_record=0: {neg}')


if __name__ == '__main__':
    main()
