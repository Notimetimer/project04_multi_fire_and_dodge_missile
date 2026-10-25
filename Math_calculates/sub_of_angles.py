from math import pi
from math import pi, cos, sin, atan2

def sub_of_radian(input1, input2=0):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


def sub_of_degree(input1, input2=0):
    # 计算两个角度的差值，范围为[-180, 180]
    diff = input1 - input2
    diff = (diff + 180) % 360 - 180
    return diff

def rel2custom_radian(input1):
    # 将[-pi,pi]的弧度转换为[0,2pi]的弧度
    diff = input1
    diff = diff % (2 * pi)
    return diff

def rel2custom_degree(input1):
    # 将[-180,180]的角度转换为[0,360]的角度
    diff = input1
    diff = diff % 360
    return diff

def mean_of_radians_minor_arc(input_list):
    # 输出按劣弧平均的角度，不加权
    cos_sum = 0
    sin_sum = 0
    for radian in input_list:
        cos_sum += cos(radian)
        sin_sum += sin(radian)
    if cos_sum ** 2 + sin_sum ** 2 < 1e-6:
        cos_sum = 1e-6
    return atan2(sin_sum, cos_sum)
