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
actor内部区分：

A.关于RWR（RWR响起时强制进入这个）: 
        A0 尾后机动且下高
                环境动作 maneuver_for_1_enm: action_v = 2
                环境动作 maneuver_for_1_enm: action_h = 3
        A1 三九线机动且下高
                环境动作 maneuver_for_1_enm: action_v = 2
                环境动作 maneuver_for_1_enm: action_h = 2 if threat_delta_psi > 0 else 4

B.关于队友: 
        暂无启动条件

C.关于敌机: 
        敌机选取，输出目标条件
                关于敌机的机动
                action_v:
                        C1-0 爬升          -->  环境动作 maneuver_for_1_enm: action_v=0
                        C1-1 指向敌机      -->  环境动作 maneuver_for_1_enm: action_v=1
                        C1-2 下降          -->  环境动作 maneuver_for_1_enm: action_v=2
                action_h:
                        C2-0 指向          -->  环境动作 maneuver_for_1_enm: action_h=0
                        C2-1 右crank       -->  环境动作 maneuver_for_1_enm: action_h=1
                        C2-2 尾后          -->  环境动作 maneuver_for_1_enm: action_h=3
                        C2-3 左crank       -->  环境动作 maneuver_for_1_enm: action_h=5
                action_fire:
                        C3-1 开火
                        C3-0 不开火

D.无敌时刻（敌机全被消灭，且我机没有被雷达照射）：
        暂无启动条件




"""