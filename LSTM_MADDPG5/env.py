'''
状态空间：agent: 当前位置(2)，当前速度(2)，team位置(2*num_uav)，雷达（16），目标距离(1)，目标角度(1)  = 26
动作空间：x、y轴的加速度分量a(2)
self.num_uav：最后一个为环境中的目标，不属于agent
'''
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
import matplotlib.animation as animation
from PIL import Image
import random
import copy

class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_UAVs=3):
        self.length = length  # length of boundary 边界长度
        self.num_obstacle = num_obstacle  # number of obstacles
        self.num_UAV = num_UAVs #需要训练的无人机数量
        self.num_agents = num_UAVs + 1 # self.num_uav：最后一个为环境的目标，不属于agent
        self.time_step = 0.5  # update time step 时间步长
        self.v_max = 0.1  # 无人机最大速度
        self.a_max = 0.04  # 无人机的最大加速度
        self.L_sensor = 0.4  # 无人机的传感器（如激光测距仪）的探测距离。
        self.num_lasers = 16  # num of laserbeams #无人机上的激光束数量，用于探测周围环境。
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in
                                     range(self.num_agents)]  # 一个列表，包含每个无人机的当前激光束测量值，初始化为传感器距离。
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(self.num_agents)]  # 一个列表，用于记录每个无人机的历史位置。
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,)) # action represents [a_x,a_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,))
        self.v_target = 0.05  # 目标的速度，比无人机的最大速度慢。
        self.target_velocity = np.array([self.v_target, 0.])  # 物体的速度（向右匀速直线）

    def reset(self):
        SEED = random.randint(1, 1000)
        random.seed(SEED)
        self.multi_current_pos = []  # 所有代理当前位置的列表（包含target）
        self.multi_current_vel = []  # 所有代理当前速度的列表（包含target）
        self.history_positions = [[] for _ in range(self.num_agents)]  # 每个代理的历史位置将作为一个子列表存储（包含target）
        for i in range(self.num_agents):
            if i != self.num_agents - 1:  # if not target
                '''无人机的位置被随机初始化在二维空间中的一个点上，这个点的坐标x和y分别均匀分布在0.1到0.4之间。'''
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:  # for target
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                '''目标的位置被设置为一个特定的点（在代码中，这个位置被设置为[0.5, 1.75]'''
                self.multi_current_pos.append(np.array([0.2, 1.75]))
            self.multi_current_vel.append(np.zeros(2))  # initial velocity = [0,0] #初始速度都被设置为[0, 0]

        # update lasers
        # 更新环境中每个agent的激光传感器状态，并检测是否有代理与障碍物发生碰撞
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):
        last_d2target = []  # 存储每个智能体到目标的距离
        '''循环遍历列表中每一个agent'''
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]  # 获取当前智能体的位置。
            if i != self.num_agents - 1:  # 如果当前智能体不是最后一个target
                pos_taget = self.multi_current_pos[-1]  # 目标位置为最后一个位置
                last_d2target.append(np.linalg.norm(pos - pos_taget))  # 计算并存储当前智能体到目标位置（最后一个智能体的位置）的欧几里得距离。

            #更新速度
            if i != self.num_agents - 1:  # 如果当前智能体是UAV
                # 根据动作（加速度）更新智能体的速度
                self.multi_current_vel[i][0] += actions[i][0] * self.time_step
                self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            else: #target
                #向右匀速直线运动，加速度为0
                self.multi_current_vel[i][0] = self.target_velocity[0]
                self.multi_current_vel[i][1] = self.target_velocity[1]

            '''速度向量归一化'''
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1: #环境目标速度恒定，不需要归一化
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max

            # 根据更新后的速度更新智能体的位置
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions
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

        Collided = self.update_lasers_isCollied_wrapper()  # 检查智能体是否与障碍物或其他智能体碰撞
        '''保留agent到目标的距离信息，设置奖励值'''
        rewards, dones = self.cal_rewards_dones(Collided, last_d2target)  # 根据碰撞情况和智能体到目标的距离计算奖励和是否结束的信息
        multi_next_obs = self.get_multi_obs()  # 获取所有智能体的下一步观察

        return multi_next_obs, rewards, dones

    def get_multi_obs(self):
        '''
        收集并处理多个智能体（可能是无人机或其他自主实体）的观测信息。
        这些智能体在一个环境中移动，并且每个智能体都有自己的位置、速度以及激光传感器数据。
        函数的目标是构建一个包含所有智能体观测信息的列表 total_obs
        :return:
        '''
        total_obs = [] #用于存储所有UAV的观测信息
        single_obs = []  # 临时存储单个UAV的观测信息
        for i in range(self.num_agents-1):  #不包含环境目标
            #pos 和 vel 分别是当前智能体的位置和速度。
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            # S_uavi 是一个四维列表，包含当前智能体的归一化位置和速度
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]  # dim 4

            # 用于存储其他智能体的位置信息。
            S_team = []  # dim 4 for 3 agents 1 target
            # 用于存储目标的相对距离和角度信息。
            S_target = []  # dim 2

            '''
            如果 j 不是当前智能体的索引且不是目标的索引，将j智能体的归一化位置添加到 S_team。
            如果 j 是目标的索引，计算当前智能体到目标智能体的距离 d 和角度 theta，并将归一化距离和角度添加到 S_target。
            '''
            for j in range(self.num_agents):  #包含环境目标
                if j != i and j != self.num_agents - 1: #其它UAV
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                elif j == self.num_agents - 1: #环境目标
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
                    S_target.extend([d / np.linalg.norm(2 * self.length), theta])

            # 获取激光传感器数据
            S_obser = self.multi_current_lasers[i]  # dim 16

            #观测状态合集
            single_obs = [S_uavi, S_team, S_obser, S_target]
            _single_obs = list(itertools.chain(*single_obs))  # 所有子列表展平成一个一维列表
            total_obs.append(_single_obs)
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        '''
        计算多智能体系统中的奖励（rewards）和结束状态（dones）
        :param IsCollied:每个智能体是否与障碍物或墙壁发生碰撞
        :param last_d:存储上一步每个智能体到目标智能体的距离
        :return 1 :一个数组，包含每个无人机的奖励
        :return 2 :一个布尔数组，指示每个无人机是否达到了结束状态
        '''
        dones = [False] * (self.num_UAV)  # 指示每个无人机是否达到了结束状态。
        rewards = np.zeros(self.num_UAV)  # 每个无人机的奖励。


        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        '''
        更新环境中每个代理的激光传感器状态，并检测是否有代理与障碍物或墙壁发生碰撞
        :return: 每个agent是否发生碰撞的状态列表
        '''
        self.multi_current_lasers = [] #存储每个agent当前的激光传感器读数。
        dones = []  # 每个agent是否发生碰撞的状态。
        for i in range(self.num_UAV): #只包含无人机
            pos = self.multi_current_pos[i] #当前agent位置
            current_lasers = [self.L_sensor] * self.num_lasers  # 初始化当前代理的激光读数为预设的激光传感器最大距离
            done_obs = [] #记录每个障碍物是否被碰撞
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                #雷达信息 ；并检测是否发生碰撞
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                #将更新后的激光读数_current_lasers与原始激光读数current_lasers进行比较，取两者中的最小值，以更新current_lasers
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs) #只要存在至少与某一个障碍物碰撞
            if done:
                self.multi_current_vel[i] = np.zeros(2) #如果发生碰撞（done为True），则将当前速度设置为零（np.zeros(2)）
            self.multi_current_lasers.append(current_lasers) #保存所有无人机的观测值
            dones.append(done)
        return dones

    #渲染环境并保存图像
    def render(self):
        plt.clf()
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')

        # plot round-up-UAVs
        for i in range(self.num_UAV):
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
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # plot obs
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

    #动画展示
    def render_anime(self, frame_num):
        plt.clf()
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_UAV):
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

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
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
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)


if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # record magnitude of vel
    velocities_x = [[] for _ in range(env.num_agents)]  # record vel_x
    velocities_y = [[] for _ in range(env.num_agents)]  # record vel_y

    obs = env.reset()

    def update(frame):
        global obs,velocities_magnitude,velocities_x,velocities_y

        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)

            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        actions = [[0.02,0.02],[0.02,0.02],[0.02,0.02]]
        obs_, rewards, dones = env.step(actions)
        print(rewards)
        env.render_anime(frame)
        obs = obs_
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in",frame,"steps.")

        return []

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()