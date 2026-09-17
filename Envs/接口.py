"""
一、id范围:

红方飞机 1,2,3,...
红方导弹 101,102,103,...
蓝方飞机 201,202,203,...
蓝方导弹 301,302,303,...


二、个体观测空间：

1、本机必要信息
ego_main: [ # 向量
        0 v_ego
        1 h_ego
        2 cos(theta)
        3 sin(theta)
        4 cos(phi)
        5 sin(phi)
]

ego_ammo:  ego.ammo

RWR_infos: [ # 向量
        0 warning
        1 cos(threat_delta_psi)
        2 sin(threat_delta_psi)
        3 threat_theta
        4 threat_distance
]

border

        删去Radar_info

2、制导武器信息
weapon_infos: 

3、队友信息
ally_infos:

4、敌机信息
enm_infos:


三、动作空间：



"""