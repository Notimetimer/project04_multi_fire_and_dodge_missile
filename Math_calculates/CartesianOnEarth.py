from math import cos, sin, pi
def LLH2NUE(lon, lat, h, lon_o=118, lat_o=30, h_o=0):
    x = (lat - lat_o) * 111000  # 纬度差转米（近似）
    y = h - h_o
    z = (lon - lon_o) * (111413 * cos(lat_o * pi / 180) - 94 * cos(3 * lat_o * pi / 180))  # 经度差转米（近似）
    return x, y, z
def NUE2LLH(N, U, E, lon_o=118, lat_o=30, h_o=0):
    lon = lon_o + E / (111413 * cos(lat_o * pi / 180) - 94 * cos(3 * lat_o * pi / 180))
    lat = lat_o + N / 111000
    h = U + h_o
    return lon, lat, h