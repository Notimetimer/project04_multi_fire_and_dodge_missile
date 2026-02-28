from random import random
import random
from gym import spaces
import copy
# from MissileModel2_2 import *  # from MissileModel2 import missile_class
from Envs.MissileModel1 import *  # test

'''
坐标系：
统一用北天东，前上右顺序
'''

g = 9.81
dt = 0.8  # 0.02
# t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180

# battle_field = np.array([10000, 1e4, 10000])  # 东西、南北方向各10km，高度范围1000m,以0,0,500为中心

d1 = 80e3
d2 = d1  # 100e3
d3 = 15e3

min_north = -d1 / 2
max_north = d1 / 2

min_east = -d2 / 2
max_east = d2 / 2

min_height = 0
max_height = d3

battle_field = np.array([[min_north, max_north], [min_height, max_height], [min_east, max_east]])
birthpointr = np.array([min_north + 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])
birthpointb = np.array([max_north - 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])


# random_array = np.clip(np.random.randn(3),-1,1)
# "笼子"的范围：x、y方向+-各5000m, z方向0~1000m

def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


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


# 坐标转换
def passive_rotation(vector, psi, theta, gamma):
    # vector是行向量，根据psi，theta，gamma的顺序旋转坐标系，最后输出行向量
    # 注意：北天东坐标
    R1 = np.array([
        [cos(-psi), 0, -sin(-psi)],
        [0, 1, 0],
        [sin(-psi), 0, cos(-psi)]
    ])
    R2 = np.array([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cos(gamma), sin(gamma)],
        [0, -sin(gamma), cos(gamma)]
    ])
    # (R3*R2*R1*vector).T=vector.T*R1.T*R2.T*R3.T
    return vector @ R1.T @ R2.T @ R3.T


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


# 弧度压缩
def radian_lim(input1):
    if input1 > pi:
        return input1 - 2 * pi
    if input1 <= -pi:
        return input1 + 2 * pi
    else:
        return input1


# def calculate_attack_angle(self):
#     """计算攻击角度"""
#     target_vector = self.BUAV.pos_ - self.RUAV.pos_
#     velocity_vector = self.RUAV.vel_
#
#     # 计算夹角
#     dot_product = np.dot(target_vector, velocity_vector)
#     norms = np.linalg.norm(target_vector) * np.linalg.norm(velocity_vector)
#
#     if norms == 0:
#         return 180
#
#     cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
#     angle = np.arccos(cos_angle) * 180 / np.pi
#
#     return angle


# 无人机模型
class UAVModel(object):
    def __init__(self):
        super(UAVModel, self).__init__()
        # 无人机的标识
        self.id = None
        self.red = False
        self.blue = False
        self.side = None  # 阵营
        self.size = 5  # meter
        self.area = 27.87  # F-16机翼面积
        self.m = 12000  # F-16 正常起飞重量
        self.color = None
        # 无人机的状态
        self.pos_ = np.zeros(3)
        self.speed = None  # 速率
        self.vel_ = None  # 速度矢量
        self.mach = None
        self.psi = None  # 航向角
        self.theta = None  # 俯仰角
        self.gamma = None  # 滚转角
        self.nx = None
        self.ny = None
        self.nz = None
        # 无人机飞行约束
        self.speed_max = 2 * 340
        self.speed_min = 120
        # 无人机轨迹
        self.trajectory = np.empty((0, 3))  # 新增轨迹列表
        self.vellist = np.empty((0, 3))

        # 导弹相关属性
        self.missiles = []  # 已发射的导弹列表
        self.missile_count = 0  # 已发射导弹数量
        self.max_missiles = 6  # 增加最大可携带导弹数量
        self.missile_launch_interval = 9.1  # 发射间隔
        self.last_launch_time = -10  # 上次发射时间
        self.missile_detect_range = 20e3  # 探测范围
        self.missile_launch_angle = 45  # 发射视角范围
        self.missile_min_range = 5e3  # 最小发射距离
        self.missile_optimal_range = 25e3  # 最佳发射距离
        self.missile_max_range = 50e3  # 最大发射距离
        self.missile_launch_speed_threshold = 100  # 300  # 最小发射速度要求

        # 过载量控制
        self.nx_limit = [-1, 1.5]
        self.ny_limit = [-1.5, 6]
        self.gamma_max = 170 * pi / 180

        # 无人机对抗相关
        self.got_hit = False
        self.crash = False
        self.attacking = False
        self.dead = False

    # todo 阻力系数：应该是和马赫数和迎角有关的，但是先借用下导弹的阻力系数函数了
    def Cd(self, mach):
        if 0 < mach <= 0.9:
            cd = 0.16
        if 0.9 < mach <= 1.1:
            cd = 0.16 + 0.29 * (mach - 0.9) / 0.2
        if 1.1 < mach <= 3:
            cd = 0.45 - 0.25 * (mach - 1.1) / 1.9
        else:
            cd = 0.2
        return cd / 10

    # def reset(self):
    #     # self.got_hit = False
    #     # self.crash = False
    #     # self.attacking = False
    #     # 随机生成位置
    #     if self.blue == False:
    #         self.pos_ = birthpointr
    #     else:
    #         self.pos_ = birthpointb

    # todo 补充无人机的运动方程和动作逻辑
    def move(self, nx, ny, gamma, record=False):
        # 动力学运动学输出在上述基础上进行更新
        height = self.pos_[1]
        rho = air_density(height)
        speed = self.speed
        mach, _ = calc_mach(speed, height)
        Drag = -1 / 2 * rho * speed ** 2 * self.area * self.Cd(mach)
        # 加速度更新速度
        self.speed += g * (nx - sin(self.theta)) + Drag / self.m

        v = np.clip(self.speed, self.speed_min, self.speed_max)
        self.speed = v

        self.theta += g / v * ny * cos(gamma) - g / v * cos(self.theta)
        self.theta = np.clip(self.theta, -85 * pi / 180, 85 * pi / 180)  # theta限幅
        self.psi += g * ny * sin(gamma) / (v * cos(self.theta))

        if self.psi > pi:
            self.psi -= 2 * pi
        if self.psi <= -pi:
            self.psi += 2 * pi

        v_ = np.array([v * cos(self.theta) * cos(self.psi),
                       v * sin(self.theta),
                       v * cos(self.theta) * sin(self.psi)])

        self.vel_ = v_ * v / np.linalg.norm(v_)
        # 速度更新位置
        self.pos_ += self.vel_ * dt_refer
        if record:
            self.trajectory = np.vstack((self.trajectory, self.pos_))
            self.vellist = np.vstack((self.vellist, self.vel_))

    def short_range_kill(self, target):
        # 近距杀，不需要导弹的模型
        pt_ = target.pos_
        L_ = pt_ - self.pos_
        distance = np.linalg.norm(L_)
        speed = np.linalg.norm(self.vel_)
        attack_angle = acos(np.dot(self.vel_, L_) / (distance * speed))
        if distance < 8e3 and attack_angle * 180 / pi < 30:
            if 0:  # target.side == "red":  # test 红方开无敌
                return False
            else:
                return True  # "绝对杀伤锥"

    def can_launch_missile(self, target, current_time):
        can_shoot = True
        """判断是否可以发射导弹"""
        # 检查导弹数量限制 fixme
        if self.missile_count > self.max_missiles:
            print('导弹射完')
            can_shoot = False

        if len(self.missiles) >= 2:  # 一次最多打两枚
            can_shoot = False

        # 检查发射间隔
        if current_time - self.last_launch_time < self.missile_launch_interval:
            can_shoot = False

        # 检查目标是否在探测范围内
        distance = np.linalg.norm(target.pos_ - self.pos_)
        if distance > self.missile_max_range or distance < self.missile_min_range:
            can_shoot = False

        # 检查目标是否在视角范围内
        target_vector = target.pos_ - self.pos_
        velocity_vector = self.vel_
        dot_product = np.dot(target_vector, velocity_vector)
        norms = np.linalg.norm(target_vector) * np.linalg.norm(velocity_vector)
        if norms == 0:
            can_shoot = False
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / pi
        if angle > self.missile_launch_angle:
            can_shoot = False

        return can_shoot

    def launch_missile(self, target, current_time):
        """发射导弹"""
        # 创建新导弹
        # new_missile = missile_class(self.pos_, self.vel_, current_time)
        new_missile = missile_class(self.pos_, self.vel_, target.pos_, target.vel_, current_time)  # test
        new_missile.CallNumber = 111 + self.missile_count  # len(self.missiles)

        # 根据距离调整导弹参数
        distance = np.linalg.norm(target.pos_ - self.pos_)
        if distance < self.missile_optimal_range:
            # fixme 导弹模型这块可以不调整
            new_missile.max_g = 40  # 最大过载
            new_missile.sight_angle_max = pi / 2  # 导引头视角
        else:

            new_missile.max_g = 40  # 最大过载
            new_missile.sight_angle_max = pi / 2  # 导引头视角

        self.missiles.append(new_missile)
        print('导弹已发射')
        print(current_time)
        self.missile_count += 1
        print(self.missile_count)
        self.last_launch_time = current_time
        return True

    def update_missiles(self, target, dt):
        target_hit = False
        """更新所有已发射导弹的状态"""
        for missile in self.missiles[:]:  # 使用切片创建副本以允许删除
            # 计算前导弹和目标位速
            last_pmt_ = missile.pos_
            last_vmt_ = missile.vel_
            last_ptt_ = target.pos_
            last_vtt_ = target.vel_
            # 对每一枚导弹小步走
            if not missile.dead:
                for j1 in range(int(plane_missile_time_rate)):
                    # 插值计算目标位置
                    ptt1_ = last_ptt_ + last_vtt_ * dt / plane_missile_time_rate * j1
                    # 获取目标信息
                    target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, ptt1_)
                    # 更新导弹状态
                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt=dt / plane_missile_time_rate)

                    # 毁伤判定
                    # 判断命中情况并终止运行
                    vmt1 = norm(last_vmt_)
                    if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                        missile.dead = True
                    if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                        missile.dead = True
                    if missile.t > missile.t_max:  # 超时自爆
                        missile.dead = True
                    if missile.t >= 0 + dt:
                        hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, ptt1_, last_vtt_,
                                                           dt=dt / plane_missile_time_rate)
                        if hit:
                            print('Target hit')
                            missile.dead = True
                            missile.hit = True
                            target_hit = True

                            missile.pos_ = point_m
                            missile.vel_ = last_vmt_
                            target.pos_ = point_t
                            target.vel_ = last_vtt_
                            # print(norm(point_m-point_t))

        return target_hit

        #     hit, _, _ = hit_target(pmt_1, vmt_1, ptt_1, vtt_1, dt=dt/plane_missile_time_rate)
        #     if hit:
        #         print('target hit')
        #         missile.dead = True
        #         target_hit = True
        #         break
        #     if missile.dead:
        #         # self.missiles.remove(missile)
        #         continue
        #
        #     # 获取目标信息
        #     target_info = missile.observe(missile.vel_, target.vel_, missile.pos_, target.pos_)
        #     # 更新导弹状态
        #     missile.step(target_info, dt)
        #
        # return target_hit


class Battle(object):
    def __init__(self, args):
        super(Battle, self).__init__()
        self.args = args
        self.UAVs = None
        self.dt = None
        self.t = None
        self.game_time_limit = None
        self.running = None
        self.action_space = []
        self.reset()  # 重置位置和状态
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        # 智能体的观察空间
        r_obs_n, b_obs_n = self.get_obs()
        self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             r_obs_n]
        self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             b_obs_n]
        self.missile_trajectories = []
        self.RUAV.trajectory = []
        self.BUAV.trajectory = []

    def reset(self):  # 重置位置和状态
        self.dt = dt_refer  # simulation interval，1 second
        self.t = 0
        self.game_time_limit = self.args.max_episode_len
        self.UAVs = [UAVModel() for _ in range(2)]
        for i, UAV in enumerate(self.UAVs):
            UAV.id = i
            if i < 1:
                UAV.red = True
                UAV.blue = False
                UAV.blue = False
                UAV.side = "red"
                UAV.color = np.array([1, 0, 0])
                self.RUAV = UAV
            else:
                UAV.red = False
                UAV.blue = True
                UAV.side = "blue"
                UAV.color = np.array([0, 0, 1])
                self.BUAV = UAV

        self.running = True
        for i, UAV in enumerate(self.UAVs):
            # UAV.trajectory = np.empty((0, 3))  # 新增轨迹列表
            # UAV.vellist = np.empty((0, 3))
            # outrange
            UAV.got_hit = False
            UAV.dead = False
            # UAV.reset()
            # 出生点
            if UAV.red:
                # 红方初始化
                UAV.pos_ = birthpointr.copy() + \
                           np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
                # UAV.pos_ = np.array([-38841.96119795,   9290.02131746,  -1686.95469864])
                UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
                UAV.psi = (0 + (random.randint(0, 1)-0.5)*2 * random.uniform(50, 60)) * pi / 180
                UAV.theta = 0 * pi / 180
                UAV.gamma = 0 * pi / 180
                UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                                 sin(UAV.theta),
                                                 cos(UAV.theta) * sin(UAV.psi)])
                # UAV.vel_ = np.array([59.41381164,  0.34337249, -0.22103594])/dt
                self.RUAV = UAV  # 同步名称（reset）
            if UAV.blue:
                # 蓝方初始化
                UAV.pos_ = birthpointb.copy() + \
                           np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
                # UAV.pos_ = np.array([38005.14540582, 6373.80721704, -1734.42509136])
                UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
                UAV.psi = (180 + (random.randint(0, 1)-0.5)*2 * random.uniform(50, 60)) * pi / 180
                UAV.theta = 0 * pi / 180
                UAV.gamma = 0 * pi / 180
                UAV.psi = sub_of_radian(UAV.psi, 0)
                UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                                 sin(UAV.theta),
                                                 cos(UAV.theta) * sin(UAV.psi)])
                # UAV.vel_ = np.array([-56.67453876, 0., 12.5974088]) / dt
                self.BUAV = UAV
        # 不考虑弹药数量，只考虑发射间隔
        self.missile_trajectories = []
        self.RUAV.trajectory = []
        self.BUAV.trajectory = []
        return r_obs_n, b_obs_n

    def step(self, r_actions, b_actions, assume=False, record=False):
        # 输入动作（范围为[-1,1],含义依次为nx,ny,gamma）
        if not assume:
            self.t += dt_refer
            self.t = round(self.t, 2)  # 保留两位小数
        else:
            self.t = round(self.t, 2)
            UAVs_ = copy.deepcopy(self.UAVs)
            pass

        actions = r_actions + b_actions
        if not assume:
            # 更新导弹状态
            for UAV in self.UAVs:
                adv = self.UAVs[1 - UAV.id]
                if UAV.red:  # 红方无人机
                    # 更新已发射导弹
                    hit = UAV.update_missiles(self.BUAV, dt_refer)
                    if hit:
                        self.BUAV.got_hit = True
                        self.running = False
                    if not (UAV.dead or adv.dead):
                        # 发射新导弹 test
                        if UAV.can_launch_missile(self.BUAV, self.t):
                            UAV.launch_missile(self.BUAV, self.t)

        # 无人机运动更新
        for UAV, action in zip(self.UAVs, actions):
            if UAV.dead:
                continue
            # 输入动作与动力运动学状态
            action = np.clip(action, -1, 1)
            vel_ = UAV.vel_
            pos_ = UAV.pos_
            vel_hor = np.linalg.norm([vel_[0], vel_[2]])
            theta = np.arctan2(vel_[1], vel_hor)
            gamma = action[2] * pi / 1
            UAV.gamma = gamma
            g_x = -g * sin(theta)
            g_y = -g * cos(theta) / cos(gamma) if abs(
                gamma) != pi / 2 else 0
            height = pos_[1]
            rho = air_density(height)
            speed = UAV.speed
            mach, _ = calc_mach(speed, height)
            Drag = -1 / 2 * rho * speed ** 2 * UAV.area * UAV.Cd(mach)

            nx_compensate = -(g_x + Drag / UAV.m) / g
            ny_compensate = -g_y / g

            nx_min = UAV.nx_limit[0]
            nx_max = UAV.nx_limit[1]
            ny_min = UAV.ny_limit[0]
            ny_max = UAV.ny_limit[1]

            nx = action[0] * nx_max + nx_compensate if action[0] >= 0 else -action[0] * nx_min + nx_compensate
            ny = action[1] * ny_max + ny_compensate if action[1] >= 0 else -action[1] * ny_min + ny_compensate

            gamma = 180 * (pi / 180) * action[2]

            nx = np.clip(nx, nx_min, nx_max)
            ny = np.clip(ny, ny_min, ny_max)
            gamma = radian_lim(gamma)

            UAV.move(nx, ny, gamma, record)

        # 毁伤判断
        for index, UAV in enumerate(self.UAVs):
            adv = self.UAVs[1 - index]
            # 近距杀
            short_range_killed = UAV.short_range_kill(adv)
            if short_range_killed:
                self.running = False
                adv.got_hit = True

        # 撞击判断
        for index, UAV in enumerate(self.UAVs):
            adv = self.UAVs[1 - index]
            # 相撞判别
            pt_ = adv.pos_
            L_ = pt_ - UAV.pos_
            distance = np.linalg.norm(L_)
            if distance <= 100:
                self.running = False
                UAV.crash = True
                adv.crash = True
            # 出界判别
            if self.out_range(UAV):
                self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
                UAV.dead = True
                self.running = False

        r_dones = False
        b_dones = False
        if self.RUAV.dead:
            r_dones = True
        if self.BUAV.dead:
            b_dones = True

        if assume:
            terminate = False
            self.running = True
            self.UAVs = UAVs_

        self.RUAV = self.UAVs[0]
        self.BUAV = self.UAVs[1]

        # 记录轨迹
        self.RUAV.trajectory.append(self.RUAV.pos_.copy())
        self.BUAV.trajectory.append(self.BUAV.pos_.copy())
        
        if record:
            # 记录导弹轨迹
            for missile in self.UAVs[0].missiles:
                if not missile.dead:
                    self.missile_trajectories.append({
                        'traj': missile.trajectory.copy(),
                        'launch_time': missile.launch_time
                    })

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def get_state(self, side):
        '''
        每个智能体动作输入为nx, ny, gamma
        观测结构:
        自身状态信息：
            012 自身位置（东北天）
            345 v, θ, ψ
        敌方目标共7个位，分别为：
            0、可观测标志位
            1、相对本机速度矢量的俯仰角
            2、相对本机速度矢量的航向角
            3、同本机的距离
            4、视径向速度
            5、视周向角速率
            6、空置
        '''
        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
        own_situations = np.zeros((6,), dtype=np.float32)
        adv_situations = np.zeros((7,), dtype=np.float32)
        # todo 观测空间欠缺归一化！暂不考虑局部观测的问题，
        own_situations[0:3] = own.pos_
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        # print(vv_)
        theta = atan2(vv_, np.linalg.norm(vh_))
        psi = atan2(vh_[2], vh_[0])
        own_situations[3] = np.linalg.norm(v_)
        own_situations[4] = theta
        own_situations[5] = psi
        # 目标观测信息
        # 补充:能否观测标记
        L_ = adv.pos_ - own.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        # 6↓
        adv_situations[0] = 1
        # 7↓
        adv_situations[1] = q_epsilon  # sub_of_radian(q_epsilon, theta)
        # 8↓
        adv_situations[2] = sub_of_radian(q_beta, psi)
        # 9↓
        adv_situations[3] = np.linalg.norm(L_)
        # 相对速度
        vT_ = adv.vel_
        vr_ = vT_ - v_
        vr_radial = np.dot(vr_, v_) / np.linalg.norm(v_)  # 径向速度
        temp = np.cross(v_, vr_) / np.linalg.norm(v_)
        vr_tangent = np.linalg.norm(temp)  # 周向速度
        omega = vr_tangent / np.linalg.norm(L_)
        # 10↓
        adv_situations[4] = vr_radial
        # 11↓
        adv_situations[5] = omega
        # 12↓
        adv_situations[6] = 0  # 空置，我也不知道能写啥

        one_side_obs_one_drone = np.concatenate([
            own_situations.flatten(),
            adv_situations.flatten(),
        ])
        one_side_obs = one_side_obs_one_drone
        return one_side_obs

    def get_obs(self):
        r_obs_n = self.get_state('r')
        # v_ = self.RUAV.vel_
        # L_ = self.BUAV.pos_ - self.RUAV.pos_
        # off_axis = acos(np.dot(L_, v_) / (np.linalg.norm(v_) * np.linalg.norm(L_))) * 180 / pi
        # if off_axis > 100:  # 部分观测 test
        #     r_obs_n[6:] = 0
        b_obs_n = self.get_state('b')
        # v_ = self.BUAV.vel_
        # L_ = self.RUAV.pos_ - self.BUAV.pos_
        # off_axis = acos(np.dot(L_, v_) / (np.linalg.norm(v_) * np.linalg.norm(L_))) * 180 / pi
        # if off_axis > 100:  # 部分观测 test
        #     b_obs_n[6:] = 0

        return np.stack(r_obs_n), np.stack(b_obs_n)

    # 碰边或撞地
    def out_range(self, UAV):
        position = UAV.pos_
        out = True
        if min_north <= position[0] <= max_north and \
                min_height <= position[1] <= max_height and \
                min_east <= position[2] <= max_east:
            out = False
        # if min_north <= position[0] <= max_north and \
        #         min_height <= position[1] <= max_height and \
        #         min_east <= position[2] <= max_east:
        # if min_north <= position[0] <= max_north and \
        #         min_height <= position[1] and \
        #         min_east <= position[2] <= max_east:
        #     out = False
        # # 对升限的处理有限，目前只能强制按平
        # if position[1] >= max_height + 2000:
        #     UAV.vel_[1] = 0
        #     UAV.pos_[1] = max_height + 2000
        #     UAV.theta = 0
        return out

    def get_reward(self, missiled_combat='Flase'):
        if missiled_combat == True:
            # 添加导弹命中相关的奖励和惩罚
            pass
        '结果奖励部分'
        RUAV = self.RUAV
        BUAV = self.BUAV
        UAVs = [RUAV, BUAV]
        A = [0, 0]  # R, B
        rewards = [0, 0]  # R, B
        R_h = 0   #高度优势奖励函数
        R_v = 0   #速度
        R_fai =0   #角度
        R_d = 0    #距离

        # 获取红方和蓝方的速度
        red_speed = RUAV.speed
        blue_speed = BUAV.speed


        for i, UAV in enumerate(UAVs):  # UAVs[0]为红方，UAVs[1]为蓝方
            adv = UAVs[1 - i]
            # # 出界剩余时间惩罚：
            t_last_max = 60
            t_last = np.ones(6)
            t_last[0] = (max_height - UAV.pos_[1]) / (UAV.vel_[1]) if UAV.vel_[1] > 0 else t_last_max
            t_last[1] = (UAV.pos_[1] - min_height) / (-UAV.vel_[1]) if UAV.vel_[1] < 0 else t_last_max
            t_last[2] = (max_east - UAV.pos_[2]) / (UAV.vel_[2]) if UAV.vel_[2] > 0 else t_last_max
            t_last[3] = (UAV.pos_[2] - min_east) / (-UAV.vel_[2]) if UAV.vel_[2] < 0 else t_last_max
            t_last[4] = (max_north - UAV.pos_[0]) / (UAV.vel_[0]) if UAV.vel_[0] > 0 else t_last_max
            t_last[5] = (UAV.pos_[0] - min_north) / (-UAV.vel_[0]) if UAV.vel_[0] < 0 else t_last_max
            #
            min_t_last = min(t_last)
            rewards[i] -= (1 - min_t_last / t_last_max) * 100

            # 态势优势奖励
            # 高度优势函数
            if UAV.pos_[1] <= 13e3 and UAV.pos_[1] >= 1e3:     # 其余条件按照out_ranges算
                R_h += 40 / (1 + np.exp(0.1*(UAV.pos_[1]-14e3))) - 40 / (1 + np.exp(0.1*(UAV.pos_[1]-1e3))) - 40

            # 速度优势函数
            # 根据速度关系调整 R_v
            if red_speed > 1.5 * blue_speed:
                R_v += 40
            elif 0.8 * blue_speed < red_speed <= 1.5 * blue_speed:
                R_v += 40 * (-1 + red_speed/ blue_speed)
            elif red_speed <= 0.8 * blue_speed:
                R_v -= 40

            # 角度优势函数
            # 计算蓝方相对于红方的位置向量
            relative_pos = BUAV.pos_ - RUAV.pos_
            # 计算水平距离
            horizontal_distance = np.linalg.norm(relative_pos[[0, 2]])
            # 计算垂直距离
            vertical_distance = relative_pos[1]
            # 计算目视倾角 fai（弧度）
            fai_rad = np.abs(atan2(vertical_distance, horizontal_distance))
            # 将弧度转换为角度
            fai_deg = fai_rad * 180 / pi
            # 根据 fai 调整 R_fai
            if fai_deg < 45:
                R_fai += 40 * (1-fai_deg/45)
            elif fai_deg > 45:
                R_fai -= 40 * (fai_deg/45-1)

            #距离优势函数
            # 获取导弹的最大和最小发射距离
            Dmin = RUAV.missile_min_range
            Dmax = RUAV.missile_max_range
            # 计算红蓝双方相对距离 D
            D = np.linalg.norm(BUAV.pos_ - RUAV.pos_)

            # 根据相对距离 D 调整 R_d
            if D > Dmax:
                R_d -= 40  # 距离过远，给予负奖励
            elif Dmin < D <= Dmax:
                R_d += 40  # 在有效射程内，给予正奖励
            elif D <= Dmin:
                R_d -= 40  # 距离过近，给予负奖励
            # 将 R_v 添加到红方奖励中
            rewards[i] += R_v + R_h + R_fai + R_d


            # 靠近边界惩罚
            if UAV.pos_[1] < 800:
                rewards[i] -= 80 * (1 - UAV.pos_[1] / 800) * -UAV.theta / (pi / 2)
            if max_height - UAV.pos_[1] < 800:
                rewards[i] -= 80 * (1 - (max_height - UAV.pos_[1]) / 800) * UAV.theta / (pi / 2)
            ns_range = max_north - min_north
            ew_range = max_east - min_east
            trigger = 2e3
            if min(max_north - UAV.pos_[0], UAV.pos_[0] - min_north) <= trigger:
                rewards[i] -= 30 * (trigger - min(max_north - UAV.pos_[0], UAV.pos_[0] - min_north)) / trigger
            if min(max_east - UAV.pos_[2], UAV.pos_[2] - min_east) <= trigger:
                rewards[i] -= 30 * (trigger - min(max_east - UAV.pos_[2], UAV.pos_[2] - min_east)) / trigger

            # 出界惩罚
            if self.out_range(UAV):
                rewards[i] -= 100
            # 超出限高惩罚
            # if UAV.pos_[1] >= max_height:
            #     rewards[i] -= 80  # 重复
            # 对手出界
            if self.out_range(adv):
                rewards[i] += 100
            # 被命中(不管是导弹还是"扫描枪")
            if UAV.got_hit:
                rewards[i] -= 100
            # 命中对手
            if adv.got_hit:
                rewards[i] += 100  # 增加命中奖励
            # if BUAV.got_hit:
            #     rewards[i] += 200  # 增加命中奖励
            # 撞机
            if UAV.crash:
                rewards[i] -= 70
            # 平局
            if self.t >= self.game_time_limit and not any([UAV.dead, adv.dead]):
                rewards[i] -= 10

            # test 目标追赶
            r_obs_n, b_obs_n = self.get_obs()
            # rewards[0] = (1 - np.linalg.norm(r_obs_n[7] / pi * 2)) * 100
            rewards[0] += (1 - np.linalg.norm((r_obs_n[7] - RUAV.theta) / pi * 2 * 2)) * 40  # test
            rewards[1] += (  1 - np.linalg.norm((b_obs_n[7] - BUAV.theta) / pi * 2 * 2)) * 40  # test
            # rewards[1] = (1-np.linalg.norm(BUAV.theta/pi*2)) * 100
            rewards[0] += (1 - np.linalg.norm(r_obs_n[8] / pi)) * 40
            rewards[1] += (1 - np.linalg.norm(b_obs_n[8] / pi)) * 40

        # return rewards[0], rewards[1]
        # todo 奖励改成元组形式，第一项喂给经验池，第二项用作episode_return
        return (rewards[0], rewards[0] - 100), (rewards[1], rewards[1] - 100)

    # def take_hit(self):
    #     # r_alive = np.array([UAV.got_hit for UAV in self.RUAVs], dtype=np.float32)
    #     # b_alive = np.array([UAV.got_hit for UAV in self.BUAVs], dtype=np.float32)
    #     r_alive = np.array([self.RUAV.got_hit], dtype=np.float32)
    #     b_alive = np.array([self.BUAV.got_hit], dtype=np.float32)
    #
    #     return r_alive, b_alive

    def get_terminate(self):  # 历史遗留，别动
        r_dead = [self.RUAV.got_hit]
        b_dead = [self.BUAV.got_hit]
        if self.running == False:
            return True
        if all(r_dead) or all(b_dead):
            return True
        return False

    # def clear_death(self):
    #     for UAV in self.UAVs:
    #         UAV.dead = UAV.got_hit
