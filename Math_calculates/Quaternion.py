import numpy as np
# # %matplotlib qt
# # %matplotlib notebook # 为了让图能够旋转
import matplotlib.pyplot as plt
from scipy.linalg import expm

# 欧拉角转四元数
def eu2quat(eu):
    psi, theta, phi = eu
    e0 = np.cos(psi/2)*np.cos(theta/2)*np.cos(phi/2)+np.sin(psi/2)*np.sin(theta/2)*np.sin(phi/2)
    e1 = np.cos(psi/2)*np.cos(theta/2)*np.sin(phi/2)-np.sin(psi/2)*np.sin(theta/2)*np.cos(phi/2)
    e2 = np.cos(psi/2)*np.sin(theta/2)*np.cos(phi/2)+np.sin(psi/2)*np.cos(theta/2)*np.sin(phi/2)
    e3 = np.sin(psi/2)*np.cos(theta/2)*np.cos(phi/2)-np.cos(psi/2)*np.sin(theta/2)*np.sin(phi/2)
    return np.array([e0, e1, e2, e3])
# 四元数乘法
def quaternionMultiplication(q1, q2):
    # q1, q2: [q0, q1, q2, q3]
    result = np.zeros(4)
    result[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    result[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    result[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    result[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return result

def quaternionInv(q):
    # q: [q0, q1, q2, q3]
    # 共轭
    return np.array([q[0], -q[1], -q[2], -q[3]])

def handwritten_turn(q, v0_):
    # q和v0_都是列向量，q是四元数，v0_是[4x1]形式的向量，[0; vector]
    # 在Python中v0_可以传入np.array([0, vx, vy, vz])
    q_inv = quaternionInv(q)
    temp1 = quaternionMultiplication(q, v0_)
    v1_1 = quaternionMultiplication(temp1, q_inv)
    return v1_1

def draw_axis(ax, o, T, N, B, name1, name2, name3):
    # 在3D图上绘制坐标轴箭头
    # ax.quiver(o[0], o[1], o[2], T[0]-o[0], T[1]-o[1], T[2]-o[2], color='r', linewidth=2)
    # ax.quiver(o[0], o[1], o[2], N[0]-o[0], N[1]-o[1], N[2]-o[2], color='g', linewidth=2)
    # ax.quiver(o[0], o[1], o[2], B[0]-o[0], B[1]-o[1], B[2]-o[2], color='b', linewidth=2)
    ax.plot([o[0],T[0]], [o[1],T[1]],[o[2],T[2]], color='r', linewidth=2)
    ax.plot([o[0],N[0]], [o[1],N[1]],[o[2],N[2]], color='g', linewidth=2)
    ax.plot([o[0],B[0]], [o[1],B[1]],[o[2],B[2]], color='b', linewidth=2)

    ax.text(T[0], T[1], T[2], name1)
    ax.text(N[0], N[1], N[2], name2)
    ax.text(B[0], B[1], B[2], name3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    ax.grid(True)

def quat_mult(q1, q2):
    # 简化版本四元数乘法，q1,q2=[q0,q1,q2,q3]
    q1s = q1[0]
    q1v = q1[1:]
    q2s = q2[0]
    q2v = q2[1:]
    qs = q1s*q2s - np.dot(q1v,q2v)
    qv = q1s*q2v + q2s*q1v + np.cross(q1v,q2v)
    return np.concatenate(([qs], qv))

# 直接使用的旋转公式
def rotate_with_Rod(vector0_, axis0_, alpha):
    "被转向量，转轴，转角(弧度)"
    alpha_hat_ = axis0_/(np.linalg.norm(axis0_)+1e-9)
    alpha_ = alpha_hat_*alpha
    vector1_ = (1-np.cos(alpha))*np.dot(alpha_hat_,vector0_)*alpha_hat_+\
                np.cos(alpha)*vector0_+\
                np.sin(alpha)*np.cross(alpha_hat_,vector0_)
    return vector1_


# 开始翻译主代码
if __name__ == "__main__" :
    # 初始化参数
    psi = 0*np.pi/180
    theta = 0*np.pi/180
    phi = 0*np.pi/180

    # R_zyx 矩阵
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])
    R_x = np.array([
        [1,0,0],
        [0,np.cos(phi), -np.sin(phi)],
        [0,np.sin(phi),  np.cos(phi)]
    ])
    R_zyx = R_z @ R_y @ R_x

    T = R_zyx @ np.array([1,0,0])
    N = R_zyx @ np.array([0,1,0])
    B = R_zyx @ np.array([0,0,1])

    v=3.0
    vector0_=T*v
    v0_=np.concatenate(([0],vector0_))
    delta_t=1
    wx=0*np.pi/180
    wy=0*np.pi/180
    wz=60*np.pi/180

    # 原方向
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    o = np.array([0,0,0])
    draw_axis(ax, o, T, N, B, 'T_0','N_0','B_0')
    ax.set_box_aspect([1,1,1])  # 使 x、y、z 轴看起来比例相等
    ax.set_xlim([-1, 1])  # 设置 x 轴范围
    ax.set_ylim([-1, 1])  # 设置 y 轴范围
    ax.set_zlim([-1, 1])  # 设置 z 轴范围

    # 方法1
    alpha_x=wx*delta_t
    alpha_y=wy*delta_t
    alpha_z=wz*delta_t
    alpha_=alpha_x*T+alpha_y*N+alpha_z*B
    alpha=np.linalg.norm(alpha_)
    if alpha>0:
        q=np.concatenate(([np.cos(alpha/2)], np.sin(alpha/2)*alpha_/alpha))
    else:
        q=np.array([1,0,0,0])
    # v1_1计算
    v0_array = v0_   # [0,vx,vy,vz]
    # 需要用写好的handwritten_turn, 注意这里手写turn用的q是[q0,q1,q2,q3], v0_是四元数形式[0; v]
    # 但原函数是将q和v0_都当成四元数，这里需要保持一致：
    v0_quat = v0_array
    q_quat = q
    # handwritten_turn是MATLAB版，这里手动套用
    def handwritten_turn_py(q,v):
        q_inv=quaternionInv(q)
        temp1=quaternionMultiplication(q,v)
        return quaternionMultiplication(temp1,q_inv)
    v1_1=handwritten_turn_py(q_quat, v0_quat)

    # 旋转可视化
    T1_1=handwritten_turn_py(q_quat, np.concatenate(([0],T)))[1:]
    N1_1=handwritten_turn_py(q_quat, np.concatenate(([0],N)))[1:]
    B1_1=handwritten_turn_py(q_quat, np.concatenate(([0],B)))[1:]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    draw_axis(ax1,o,T,N,B,'T_0','N_0','B_0')
    draw_axis(ax1,o,T1_1,N1_1,B1_1,'T_{1-1}','N_{1-1}','B_{1-1}')
    ax1.set_box_aspect([1,1,1])  # 使 x、y、z 轴看起来比例相等
    ax1.set_xlim([-1, 1])  # 设置 x 轴范围
    ax1.set_ylim([-1, 1])  # 设置 y 轴范围
    ax1.set_zlim([-1, 1])  # 设置 z 轴范围

    # 方法2 (使用Rodrigues公式形式)
    alpha_x=wx*delta_t
    alpha_y=wy*delta_t
    alpha_z=wz*delta_t
    alpha_=alpha_x*T+alpha_y*N+alpha_z*B
    alpha=np.linalg.norm(alpha_)
    if alpha>0:
        alpha_hat_=alpha_/alpha
    else:
        alpha_hat_=np.array([0,0,0])
    v1_2 = np.concatenate((
        [0],
        (1-np.cos(alpha))*np.dot(alpha_hat_,vector0_)*alpha_hat_+np.cos(alpha)*vector0_+np.sin(alpha)*np.cross(alpha_hat_,vector0_)
    ))

    T1_2 = np.concatenate((
        [0],
        (1-np.cos(alpha))*np.dot(alpha_hat_,T)*alpha_hat_+np.cos(alpha)*T+np.sin(alpha)*np.cross(alpha_hat_,T)
    ))
    N1_2 = np.concatenate((
        [0],
        (1-np.cos(alpha))*np.dot(alpha_hat_,N)*alpha_hat_+np.cos(alpha)*N+np.sin(alpha)*np.cross(alpha_hat_,N)
    ))
    B1_2 = np.concatenate((
        [0],
        (1-np.cos(alpha))*np.dot(alpha_hat_,B)*alpha_hat_+np.cos(alpha)*B+np.sin(alpha)*np.cross(alpha_hat_,B)
    ))

    T1_2=T1_2[1:]
    N1_2=N1_2[1:]
    B1_2=B1_2[1:]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    draw_axis(ax2,o,T,N,B,'T_0','N_0','B_0')
    draw_axis(ax2,o,T1_2,N1_2,B1_2,'T_{1-2}','N_{1-2}','B_{1-2}')
    ax2.set_box_aspect([1,1,1])  # 使 x、y、z 轴看起来比例相等
    ax2.set_xlim([-1, 1])  # 设置 x 轴范围
    ax2.set_ylim([-1, 1])  # 设置 y 轴范围
    ax2.set_zlim([-1, 1])  # 设置 z 轴范围

    # 方法3：通过微分方程积分四元数
    dt=0.02
    q3=np.array([1,0,0,0])
    omega=wx*T+wy*N+wz*B
    omegax=omega[0]
    omegay=omega[1]
    omegaz=omega[2]
    q_dot_matrix=np.array([
        [0,-omegax,-omegay,-omegaz],
        [omegax,0,omegaz,-omegay],
        [omegay,-omegaz,0,omegax],
        [omegaz,omegay,-omegax,0]
    ])

    t_=0
    while t_<=delta_t:
        q_dot=0.5*q_dot_matrix@q3
        q3=q3+q_dot*dt
        q3=q3/np.linalg.norm(q3)
        t_+=dt

    e0,e1,e2,e3 = q3
    R=np.array([
        [e0**2+e1**2-e2**2-e3**2, 2*(e1*e2 - e0*e3), 2*(e1*e3+e0*e2)],
        [2*(e1*e2+e0*e3), e0**2 - e1**2 + e2**2 - e3**2, 2*(e2*e3 - e0*e1)],
        [2*(e1*e3 - e0*e2), 2*(e2*e3+e0*e1), e0**2 - e1**2 - e2**2 + e3**2]
    ])
    T1_3=R@T
    N1_3=R@N
    B1_3=R@B
    v1_3=R@vector0_

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    draw_axis(ax3,o,T,N,B,'T_0','N_0','B_0')
    draw_axis(ax3,o,T1_3,N1_3,B1_3,'T_{1-3}','N_{1-3}','B_{1-3}')
    # ax3.quiver(o[0], o[1], o[2], v1_3[0]-o[0], v1_3[1]-o[1], v1_3[2]-o[2], color='black')
    ax3.set_box_aspect([1,1,1])  # 使 x、y、z 轴看起来比例相等
    ax3.set_xlim([-1, 1])  # 设置 x 轴范围
    ax3.set_ylim([-1, 1])  # 设置 y 轴范围
    ax3.set_zlim([-1, 1])  # 设置 z 轴范围

    # 方法5 解析计算四元数
    alpha_x=wx*delta_t
    alpha_y=wy*delta_t
    alpha_z=wz*delta_t
    alpha_=alpha_x*T+alpha_y*N+alpha_z*B
    alpha=np.linalg.norm(alpha_)
    if alpha>0:
        q_rot=np.concatenate(([np.cos(alpha/2)], np.sin(alpha/2)*alpha_/alpha))
    else:
        q_rot=np.array([1,0,0,0])
    q0=np.array([1,0,0,0])
    q5=quat_mult(q_rot, q0)
    e0,e1,e2,e3=q5
    R=np.array([
        [e0**2+e1**2-e2**2-e3**2, 2*(e1*e2 - e0*e3), 2*(e1*e3+e0*e2)],
        [2*(e1*e2+e0*e3), e0**2 - e1**2 + e2**2 - e3**2, 2*(e2*e3 - e0*e1)],
        [2*(e1*e3 - e0*e2), 2*(e2*e3+e0*e1), e0**2 - e1**2 - e2**2 + e3**2]
    ])
    T1_5=R@T
    N1_5=R@N
    B1_5=R@B
    v1_5=R@vector0_

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    draw_axis(ax5,o,T,N,B,'T_0','N_0','B_0')
    draw_axis(ax5,o,T1_3,N1_3,B1_3,'T_{1-3}','N_{1-3}','B_{1-3}')
    # ax5.quiver(o[0], o[1], o[2], v1_4[0]-o[0], v1_4[1]-o[1], v1_4[2]-o[2], color='black')
    ax5.set_box_aspect([1,1,1])  # 使 x、y、z 轴看起来比例相等
    ax5.set_xlim([-1, 1])  # 设置 x 轴范围
    ax5.set_ylim([-1, 1])  # 设置 y 轴范围
    ax5.set_zlim([-1, 1])  # 设置 z 轴范围

    plt.show()
