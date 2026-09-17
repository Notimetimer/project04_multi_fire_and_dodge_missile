"""
一、id范围:

红方飞机 1,2,3,...
红方导弹 101,102,103,...
蓝方飞机 201,202,203,...
蓝方导弹 301,302,303,...


二、个体观测空间：
ego_main: [
        'v_ego'
        'h_ego'
        'cos_theta'
        'sin_theta'
        'cos_phi'
        'sin_phi'
]
ego_control: [
        'p', 
        'q', 
        'r', 
        'theta_v', 
        'psi_v', 
        'alpha_air', 
        'beta_air', 
        'Ny'
]
radar_info: [
        ego.radar_on
        0
        0
]
border :[
        'd_hor2border'
        'border_LR'
        'dist_from_center'
]
ego_ammo:  ego.ammo
RWR_infos: [[
        'warning'
        'cos_threat_delta_psi'
        'sin_threat_delta_psi'
        'threat_theta'
        'threat_distance'
]]
weapon_infos: [[
        'missile_t_fly'
        'cos_msl_delta_psi'
        'sin_msl_delta_psi'
        'mid_term'
]]
ally_infos: [
        'radar_on'
        'cos_ally_delta_psi'
        'sin_ally_delta_psi'
        'ally_delta_theta'
        'ally_dist'
        'v_ally'
        'cos_ally_rel_heading'
        'sin_ally_rel_heading'
        'theta_ally'
]
enm_infos: [[
        'enm_observable'
        'enm_locked'
        'locked_by_enm'
        'cos_enm_delta_psi'
        'sin_enm_delta_psi'
        'enm_delta_theta'
        'enm_dist'
        'enm_ATA'
        'v_enm'
        'enm_AA'
]]
enm_info_supplement:[[
        'enm_q_beta'
        'enm_q_epsilon'
]]


三、动作空间：



"""