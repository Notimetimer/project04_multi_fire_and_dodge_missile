from datetime import datetime
import os

alg_name = 'PPO'
current_time = datetime.now().strftime('%Y%m%d_%H%M')  # 格式为: 年-月-日 时:分
'''
时间格式：
'%Y-%m-%d %H:%M' → "2024-03-15 14:30"
'%Y%m%d_%H%M' → "20240315_1430"
'%m/%d/%Y %I:%M %p' → "03/15/2024 02:30 PM"
'''
print(f"当前时间: {current_time}")
print(type(current_time))

# 拼接目录名并创建目录
dir_name = f"{alg_name}_{current_time}"
restore_path = os.path.join("Restore", dir_name)
os.makedirs(restore_path, exist_ok=True)

# 创建并写入txt文件
for i in range(10):
    file_path = os.path.join(restore_path, f"{i}.txt")
    with open(file_path, 'w') as f:
        f.write("这是存储的内容")

    print(f"已创建目录和文件: {file_path}")