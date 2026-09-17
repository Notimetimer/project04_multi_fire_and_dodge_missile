import os
import pickle
import traceback

# 相对本脚本所在目录定位 Buffer 文件夹
Buffer_dir = os.path.join(os.path.dirname(__file__), 'acmi_records')


# 全局变量：Buffer 中所有 .pkl 文件路径
pkl_files = []

# 全局变量：所有已打开的 pkl 数据，键为相对路径
pkl_data = {}

if os.path.isdir(Buffer_dir):
    for root, _, files in os.walk(Buffer_dir):
        for f in files:
            if f.endswith('.pkl'):
                pkl_files.append(os.path.join(root, f))

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
else:
    print(f"Buffer 目录不存在: {Buffer_dir}")

"""
obs1 = obs_1['obs']
rounds1 = obs_1['round']
frame1 = obs_1['frame']
alive1 = obs_1['alive']
done1 = obs_1['done']
"""
