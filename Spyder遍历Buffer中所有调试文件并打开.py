import os
import pickle
import traceback
import numpy as np

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:
    _HAS_PD = False

try:
    import torch
except ImportError:
    torch = None

# 相对本脚本所在目录定位 Buffer 文件夹
Buffer_dir = os.path.join(os.path.dirname(__file__), 'acmi_records')


# 全局变量：Buffer 中所有 .pkl 文件路径
pkl_files = []

# 全局变量：所有已打开的 pkl 数据，键为相对路径
pkl_data = {}

# 全局变量：Buffer 中所有 .npz 文件路径
npz_files = []

# 全局变量：所有已打开的 npz 数据，键为相对路径
npz_data = {}

# 全局变量：Buffer 中所有 .csv 文件路径
csv_files = []

# 全局变量：所有已打开的 csv 数据，键为相对路径
csv_data = {}

# 全局变量：Buffer 中所有 .pt 文件路径
pt_files = []

# 全局变量：所有已打开的 pt 数据，键为相对路径
pt_data = {}

if os.path.isdir(Buffer_dir):
    for root, _, files in os.walk(Buffer_dir):
        for f in files:
            if f.endswith('.pkl'):
                pkl_files.append(os.path.join(root, f))
            elif f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))
            elif f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
            elif f.endswith('.pt'):
                pt_files.append(os.path.join(root, f))

    for pkl_path in pkl_files:
        rel_path = os.path.relpath(pkl_path, os.path.dirname(__file__))
        if os.path.getsize(pkl_path) == 0:
            print(f"跳过空文件: {rel_path}")
            continue
        try:
            with open(pkl_path, 'rb') as f:
                pkl_data[rel_path] = pickle.load(f)
            print(f"已打开: {rel_path}")
        except Exception as e:
            print(f"打开失败: {rel_path}")
            print(f"  错误: {e}")
            traceback.print_exc()

    # 解包 pkl_data：把文件名（不含扩展名）作为全局变量名
    for pkl_path in pkl_files:
        rel_path = os.path.relpath(pkl_path, os.path.dirname(__file__))
        if rel_path not in pkl_data:
            continue
        var_name = os.path.splitext(os.path.basename(rel_path))[0]
        var_name = var_name.replace('-', '_').replace(' ', '_')
        if not var_name or not (var_name[0].isalpha() or var_name[0] == '_'):
            var_name = '_' + var_name
        globals()[var_name] = pkl_data[rel_path]
        print(f"  已解包为全局变量: {var_name}")

    # 读取 npz：转为普通 dict（键为字段名，值为 numpy 数组），便于变量浏览器查看
    for npz_path in npz_files:
        rel_path = os.path.relpath(npz_path, os.path.dirname(__file__))
        if os.path.getsize(npz_path) == 0:
            print(f"跳过空文件: {rel_path}")
            continue
        try:
            with np.load(npz_path) as data:
                npz_data[rel_path] = {k: data[k] for k in data.files}
            print(f"已打开: {rel_path}")
        except Exception as e:
            print(f"打开失败: {rel_path}")
            print(f"  错误: {e}")
            traceback.print_exc()

    # 解包 npz_data：把文件名（不含扩展名）作为全局变量名
    for npz_path in npz_files:
        rel_path = os.path.relpath(npz_path, os.path.dirname(__file__))
        if rel_path not in npz_data:
            continue
        var_name = os.path.splitext(os.path.basename(rel_path))[0]
        var_name = var_name.replace('-', '_').replace(' ', '_')
        if not var_name or not (var_name[0].isalpha() or var_name[0] == '_'):
            var_name = '_' + var_name
        globals()[var_name] = npz_data[rel_path]
        print(f"  已解包为全局变量: {var_name}")

    # 读取 csv：统一用 pandas
    if not _HAS_PD and csv_files:
        print("未安装 pandas，跳过所有 .csv 文件")
    for csv_path in csv_files if _HAS_PD else []:
        rel_path = os.path.relpath(csv_path, os.path.dirname(__file__))
        if os.path.getsize(csv_path) == 0:
            print(f"跳过空文件: {rel_path}")
            continue
        try:
            csv_data[rel_path] = pd.read_csv(csv_path)
            print(f"已打开: {rel_path}")
        except Exception as e:
            print(f"打开失败: {rel_path}")
            print(f"  错误: {e}")
            traceback.print_exc()

    # 解包 csv_data：把文件名（不含扩展名）作为全局变量名
    for csv_path in csv_files:
        rel_path = os.path.relpath(csv_path, os.path.dirname(__file__))
        if rel_path not in csv_data:
            continue
        var_name = os.path.splitext(os.path.basename(rel_path))[0]
        var_name = var_name.replace('-', '_').replace(' ', '_')
        if not var_name or not (var_name[0].isalpha() or var_name[0] == '_'):
            var_name = '_' + var_name
        globals()[var_name] = csv_data[rel_path]
        print(f"  已解包为全局变量: {var_name}")

    # 读取 pt：torch.load 到 cpu（weights_only=False 以兼容含 numpy 对象的存档）
    if torch is None and pt_files:
        print("未安装 torch，跳过所有 .pt 文件")
    for pt_path in pt_files if torch is not None else []:
        rel_path = os.path.relpath(pt_path, os.path.dirname(__file__))
        if os.path.getsize(pt_path) == 0:
            print(f"跳过空文件: {rel_path}")
            continue
        try:
            pt_data[rel_path] = torch.load(pt_path, map_location='cpu', weights_only=False)
            print(f"已打开: {rel_path}")
        except Exception as e:
            print(f"打开失败: {rel_path}")
            print(f"  错误: {e}")
            traceback.print_exc()

    # 解包 pt_data：把文件名（不含扩展名）作为全局变量名
    for pt_path in pt_files:
        rel_path = os.path.relpath(pt_path, os.path.dirname(__file__))
        if rel_path not in pt_data:
            continue
        var_name = os.path.splitext(os.path.basename(rel_path))[0]
        var_name = var_name.replace('-', '_').replace(' ', '_')
        if not var_name or not (var_name[0].isalpha() or var_name[0] == '_'):
            var_name = '_' + var_name
        globals()[var_name] = pt_data[rel_path]
        print(f"  已解包为全局变量: {var_name}")
else:
    print(f"Buffer 目录不存在: {Buffer_dir}")

"""
obs1 = obs_1['obs']
rounds1 = obs_1['round']
frame1 = obs_1['frame']
alive1 = obs_1['alive']
done1 = obs_1['done']
"""
