import datetime
import numpy as np
from math import cos, pi

'''根据时间创建txt文件'''
# 获取当前时间并格式化为字符串
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 创建文件名
file_name = f"{current_time}.txt"

# 创建文件并写入文件头
with open(file_name, "w") as file:  # 'w'重新写入
    file.write("FileType=text/acmi/tacview\n")  #
    file.write("FileVersion=2.2\n")
    file.write("0,ReferenceTime=2025-01-02T05:00:00Z\n")

'''定义编号'''
label1 = '001'
Name1 = "AIM-120C"  # "F-16"
CallSign1 = "red1"  # "soyo"
Color1 = "Red"

label2 = '002'
Name2 = "F-16"  # "F-16"
CallSign2 = "blue1"  # "saki"
Color2 = "Blue"

# 地理原点
Longitude0 = 144 + 43 / 60
Latitude0 = 13 + 26 / 60
Height0 = 0
mark = np.array([Longitude0, Latitude0, Height0])  # 地理原点


# # 使用pandas读取csv，跳过第一行
# data = pd.read_csv("testR.csv", header=None, skiprows=1)

# 或者使用numpy读取csv，跳过第一行
data1 = np.loadtxt("testR.csv", delimiter=",", skiprows=1)

data2 = np.loadtxt("testB.csv", delimiter=",", skiprows=1)

# tacview角度顺序是经纬高滚俯偏
# 东北天转经纬高(警告: 将xyz弯曲到地球表面,不考虑南北极的死锁问题)
def ENU2LLH(mark, NUE):
    # 东北天单位为m，经纬度单位是角度
    N, U, E = NUE
    # E, N, U = ENU
    longit0, latit0, height0 = mark
    R_earth = 6371004  # ???
    dlatit = N / R_earth * 180 / pi
    dlongit = E / (R_earth * cos(latit0 * pi / 180)) * 180 / pi
    dheight = U
    out = np.array([longit0 + dlongit, latit0 + dlatit, height0 + dheight])
    return out


# 假想一段运动+翻转
heading = np.array([0, 0, 0])
from Envs.battle3dof1v1_proportion import dt

t_list = np.arange(0, np.shape(data1)[0] * dt, dt)

# Pos_=pos0_
# p1_ = pos0_
# p2_ = pos0_ + np.array([0, 100, 0])
for i, t in enumerate(t_list):
    # v_ = np.array([1000, 10, 0])
    p1_ = data1[i, 0:3]
    loc1_ = ENU2LLH(mark, p1_)

    heading1 = data1[i, 3:6]
    # heading1 = np.zeros(3)
    p2_ = data2[i, 0:3]
    loc2_ = ENU2LLH(mark, p2_)
    heading2 = data2[i, 3:6]

    t_write = t // 0.001 / 1000  # 整除运算符，保留前三位小数
    # 挨个来写txt
    with open(file_name, "a") as file:  # 'a'追加写入
        file.write(f"#{t_write}\n")
        file.write(f"{label1},T={loc1_[0]}|{loc1_[1]}|{loc1_[2]}|{heading1[0]}|{heading1[1]}|{heading1[2]}")
        file.write(f",Name={Name1},CallSign={CallSign1},Color={Color1}\n")

        file.write(f"{label2},T={loc2_[0]}|{loc2_[1]}|{loc2_[2]}|{heading2[0]}|{heading2[1]}|{heading2[2]}")
        file.write(f",Name={Name2},CallSign={CallSign2},Color={Color2}\n")
