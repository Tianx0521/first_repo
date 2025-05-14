import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
# from dateutil.rrule import easter
from gymnasium import spaces
from torch.backends.cudnn import flags

from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy


class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_uav=3, num_lasers=16):
        self.length = length  # 地图宽度
        self.num_obstacle = num_obstacle  # 障碍物数目
        self.num_uav = num_uav  # 无人机的数目
        self.time_step = 0.5  # 时间间隙
        self.v_max = 0.1  # 无人机最大速度
        self.a_max = 0.04  # 无人机的最大加速度
        self.att_r = 0.2598 #无人机的攻击半径
        self.L_sensor = 0.2  # 激光测距的距离
        self.num_lasers = num_lasers  # 激光的个数
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in
                                     range(self.num_uav)]  # 存放激光的测量值
        self.agents = [] #存放智能体的id
        # 动态生成agents的id
        for i in range(self.num_uav):
            self.agents.append("agent_" + str(i))


        #self.info = np.random.get_state()  # 获取时间种子
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]  # 存放障碍物
        self.history_positions = [[] for _ in range(num_uav)]  # 用于存放无人机的路径节点
        self.history_target_positions = []  #存放目标的路径节点
        self.multi_current_pos = [] # 用于存放无人机的当前位置
        self.multi_current_vel = [] # 用于存放无人机的当前速度
        self.target_current_pos = [-1, -1]  # 用于存放无人机的当前位置
        self.multi_current_vel = [0, 0]  # 用于存放无人机的当前速度

        # 设置所有智能体的状态和动作空间维度
        uav_state_dim = 4 + 2 * (self.num_uav - 1) + 16 + 2  #自身状态4个（x, y ,vx, vy） 、其余无人机的位置（x, y）* (num_uav-1)、 激光测距数据16个、 目标信息2个
        uav_action_dim = 2

        self.observation_space = {}   #观测空间大小
        self.action_space = {}  #动作空间大小
        for i in range(self.num_uav):
            self.observation_space["agent_" + str(i)] = spaces.Box(low=-np.inf, high=np.inf, shape=(uav_state_dim,))
            self.action_space["agent_" + str(i)] = spaces.Box(low=-np.inf, high=np.inf, shape=(uav_action_dim,))

        #定义围捕距离(追击距离）和飞向目标的距离
        self.d_capture = self.length / 7
        self.d_limit = 0.1

        #


    def reset(self):
        SEED = random.randint(1, 1000)  #设置训练的时间种子
        random.seed(SEED)
        #根据无人机的个数和攻击半径重置无人机的围捕距离
        self.d_capture = calculate_RoundUp_distances(k=self.num_uav, r=self.att_r)

        #重置无人机相关信息
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_uav)]
        for i in range(self.num_uav):
            self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            self.multi_current_vel.append(np.zeros(2))  # initial velocity = [0,0]

        #重置目标相关信息
        self.target_current_pos = np.random.uniform(0, self.length, 2)  # 用于存放目标的当前位置
        #self.multi_current_vel = np.zeros(2)  # 用于存放目标的当前速度
        self.history_target_positions = []

        # update lasers
        self.update_lasers_isCollied_wrapper()
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):  # action[i]有两个元素，分别表示agenti在x和y方向上的加速度
        last_d2target = []  # 上一时刻无人机与目标之间的距离
        # print(actions)
        # time.sleep(0.1)
        pos_taget = self.target_current_pos
        #更新无人机的运动状态
        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            last_d2target.append(np.linalg.norm(pos - pos_taget))
            #更新无人机的速度
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            # 速度边界处理，速度的值不能查过对应最大的速度
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if vel_magnitude >= self.v_max:
                self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            # 更新无人机和目标的位置
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        #随机更新目标的运动状态
        target_vx = np.random.uniform(-0.02, 0.02)
        target_vy = np.random.uniform(-0.02, 0.02)
        self.target_current_pos[0] += target_vx * self.time_step
        self.target_current_pos[1] += target_vy * self.time_step

        # 更新障碍物的位置    Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rps = generate_points_on_circle(k=self.num_uav, center=self.target_current_pos, radius=self.d_capture)
        rewards, dones = self.cal_rewards_dones(Collided, rps)
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        total_obs = []

        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            # 添加无人机自身的状态S_uavi
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]  # dim 4


            #添加其他无人机的位置
            S_team = []  #添加其他无人机的信息
            for j in range(self.num_uav):
                if j != i:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])

            #添加目标的信息
            S_target = []
            pos_target = self.target_current_pos
            d = np.linalg.norm(pos - pos_target)
            theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
            S_target.extend([d / np.linalg.norm(2 * self.length), theta])

            #添加传感器的测量值
            S_obser = self.multi_current_lasers[i]  # dim 16

            #合成单个智能体的状态
            single_obs = [S_uavi, S_team, S_obser, S_target]

            #将单个智能体的状态拉成一维的
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
        return total_obs



    def cal_rewards_dones(self, IsCollied, rps):
        dones = [False] * self.num_uav
        rewards = np.zeros(self.num_uav)
        mu1 = 0.7#0.7  # r_near
        mu2 = 0.4  # r_safe
        mu3 = 0.5  # r_multi_stage
        mu4 = 5  # r_finish
        # d_capture = 0.3
        # d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        # #计算靠近目标围捕点的奖励
        for i in range(self.num_uav ):
            pos_target = rps[i]
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)  # distance to target
            if d > self.d_capture:
                cos_v_d = np.dot(vel, dire_vec) / (v_i * d + 1e-3)
                r_near = abs(2 * v_i / self.v_max) * cos_v_d
                # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
                rewards[i] += mu1 * r_near  # TODO: if not get nearer then receive negative reward

        ## 2 collision reward for all UAVs:
        #计算避障奖励
        for i in range(self.num_uav):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu2 * r_safe

        ##计算环境结束标志
        flag = False
        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            pos_target = self.target_current_pos
            dt = np.linalg.norm(pos_target - pos)
            dr = np.linalg.norm(rps[i] - pos)
            if dr > self.d_limit or dt > self.d_capture+self.d_limit:
                flag = True
                break
        if flag == False:
            dones = [True] * self.num_uav

        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()

        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_uav):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)

            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.target_current_pos[0], self.target_current_pos[1], c='r', label='Target')
        self.history_target_positions.append(copy.deepcopy(self.target_current_pos))
        trajectory = np.array(self.history_target_positions)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()

        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()

        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_uav):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)

            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        plt.scatter(self.target_current_pos[0], self.target_current_pos[1], c='r', label='Target')
        pos_e = copy.deepcopy(self.target_current_pos)
        self.history_target_positions.append(pos_e)
        trajectory = np.array(self.history_target_positions)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()


class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))  # 障碍物的坐标
        angle = np.random.uniform(0, 2 * np.pi)  # 障碍物的运动方向
        # speed = 0.03
        speed = 0.00  # 障碍物的速度，为0表示静止
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])  # 障碍物的速度，包含x和y方向的速度
        self.radius = np.random.uniform(0.1, 0.15)  # 障碍物的半径