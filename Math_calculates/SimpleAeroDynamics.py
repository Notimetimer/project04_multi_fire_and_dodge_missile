import numpy as np

# 定义大气计算等通用部分
# 大气密度计算
def rho(h):
    # 输入高度(m)
    answer = 1.225 * np.exp(-h / 9300)
    return answer

# 马赫数计算：
def calc_mach(v, height):
    sound_speed = 20.0463 * np.sqrt(288.15 - 0.00651122 * height) if height <= 11000 else 295.069
    return v / sound_speed, sound_speed

# 大气模型
# 声速
def soundspeed(height):
    if height <= 11000:
        soundspeed1 = 20.05 * np.sqrt(288.15 - 0.00651122 * height)
    else:
        soundspeed1 = 295.069
    return soundspeed1


# 密度
def air_density(height):
    return 1.225 * np.exp(height / 9300)
