def guided_control(selected_uav, theta_req, psi_req, v_req, k_thrust=10, k_lift=10):
    from math import pi
    import numpy as np
    theta_req = np.clip(theta_req, -pi / 2, pi / 2)
    # k_thrust = 10
    # k_lift = 10
    m = selected_uav.m
    v_ = selected_uav.vel_
    # # 航向飞行
    L_ = 10e3 * np.array(
        [np.cos(theta_req) * np.cos(psi_req), np.sin(theta_req), np.cos(theta_req) * np.sin(psi_req)])

    # # test 纯瞄准飞行
    # target_pos = env.BUAV.pos_
    # current_pos = ruav.pos_
    # L_target_ = target_pos - current_pos  # 实际目标线
    # L_ = L_target_  # 如果要跟住目标的话

    # 切向控制
    ax_required = k_thrust * (v_req - selected_uav.speed)
    ax_required = np.clip(ax_required, -1, 1)
    action_x_required = ax_required / selected_uav.nx_limit[1]

    # 俯仰控制
    offaxis_angle = np.arccos(np.dot(L_, v_) / (np.linalg.norm(L_) * np.linalg.norm(v_)))
    action_y_required = k_lift * (offaxis_angle / pi)
    action_y_required = np.clip(action_y_required, -1, 1)

    # 滚转控制
    # y轴位于速度和目标线矢量围成的平面内并垂直于速度
    temp = np.cross(v_, L_)
    if np.linalg.norm(temp) == 0:
        action_z_required = 0
    else:
        x_b_ = v_ / np.linalg.norm(v_)
        y_b_ = np.cross(temp, v_)
        y_b_ = y_b_ / np.linalg.norm(y_b_)
        # 把铅垂线往z轴的方向旋转theta从而和y_b_计算phi
        temp_ = temp / np.linalg.norm(temp) * selected_uav.theta  # 旋转矢量
        alpha = np.linalg.norm(temp_)
        up_ = np.array([0, 1, 0])
        up_temp_ = (1 - np.cos(alpha)) * np.dot(temp_, up_) * temp_ + np.cos(alpha) * up_ + np.sin(
            alpha) * np.cross(
            temp_, up_)
        up_temp_ = up_temp_ / np.linalg.norm(up_temp_)
        abs_phi_req = np.arccos(np.dot(up_temp_, y_b_))
        if np.dot(np.cross(up_temp_, y_b_), x_b_) > 0:
            phi_req = abs_phi_req
        else:
            phi_req = -abs_phi_req
        action_z_required = phi_req / pi

    r_action_n = [[action_x_required, action_y_required, action_z_required]]
    return r_action_n
