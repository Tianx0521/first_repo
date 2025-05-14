import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
#from dateutil.rrule import easter
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVEnv:
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        self.length = length # 地图宽度
        self.num_obstacle = num_obstacle # 障碍物数目
        self.num_agents = num_agents  #智能体数目（包括无人机和目标）
        self.time_step = 0.5 # 时间间隙
        self.v_max = 0.1  #无人机最大速度
        self.v_max_e = 0.12  #目标的最大速度
        self.a_max = 0.04   #无人机的最大加速度
        self.a_max_e = 0.05  #目标的最大加速度
        self.L_sensor = 0.2   #激光测距的距离
        self.num_lasers = 16 # 激光的个数
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]  #存放激光的测量值
        #self.agents = ['agent_0','agent_1','agent_2','target']  #存放智能体的id
        self.agents = []
        #动态生成agents的id
        for i in range(self.num_agents):
            if i != self.num_agents-1:
                self.agents.append("agent_"+str(i))
            else:
                self.agents.append("target")

        self.info = np.random.get_state() # 获取时间种子
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]  #存放障碍物
        self.history_positions = [[] for _ in range(num_agents)] #用于存放智能体的路径节点

        #设置所有智能体的状态和动作空间维度
        uav_state_dim = 4 + 2 * (self.num_agents-2) + 16 + 2
        target_state_dim = 4 + 16 + (self.num_agents-1)
        self.observation_space = {}
        self.action_space = {}
        for i in range(self.num_agents):
            if i != self.num_agents-1:
                self.observation_space["agent_"+str(i)]=spaces.Box(low=-np.inf, high=np.inf, shape=(uav_state_dim,))
                self.action_space["agent_"+str(i)]=spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            else:
                self.observation_space["target"]=spaces.Box(low=-np.inf, high=np.inf, shape=(target_state_dim,))
            self.action_space["target"]=spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):  #action[i]有两个元素，分别表示agenti在x和y方向上的加速度
        last_d2target = []  #上一时刻无人机与目标之间的距离
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents-1):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step

            #速度边界处理，速度的值不能查过对应最大的速度
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e
            #更新无人机的位置
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

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
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            S_obser = self.multi_current_lasers[i] # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                single_obs = [S_uavi,S_obser,S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(self.num_agents-1):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe


        ## 3 multi-stage's reward for rounding-up-UAVs
        p = []  #存放所有无人机的坐标点
        for i in range(self.num_agents):
            p.append(self.multi_current_pos[i])

        #计算所有连接目标点的三角形面积
        S = [] #存放三角形面积  共无人机的数目个三角形，num_uav-1
        for i in range(self.num_agents-1):
            if i != self.num_agents-2:
                S.append(cal_triangle_S(p[i],p[i+1], p[-1]))
            else:
                S.append(cal_triangle_S(p[i],p[0], p[-1]))

        #计算所有无人机形成的多边形面积
        S4 = 0 #存放多边形的面积，三角形数目为无人机数目-2， 即num_agent-3
        for i in range(self.num_agents-3):
            S4 = S4 + cal_triangle_S(p[i],p[i+1], p[i+2])

        #计算各无人机到目标的距离
        dl = []
        for i in range(self.num_agents-1):
            dl.append(np.linalg.norm(p[i]-p[-1]))

        Sum_S = sum(S)
        Sum_d = sum(dl)
        # p0 = self.multi_current_pos[0]
        # p1 = self.multi_current_pos[1]
        # p2 = self.multi_current_pos[2]
        # pe = self.multi_current_pos[-1]
        # S1 = cal_triangle_S(p0,p1,pe)
        # S2 = cal_triangle_S(p1,p2,pe)
        # S3 = cal_triangle_S(p2,p0,pe)
        # S4 = cal_triangle_S(p0,p1,p2)
        # d1 = np.linalg.norm(p0-pe)
        # d2 = np.linalg.norm(p1-pe)
        # d3 = np.linalg.norm(p2-pe)
        # Sum_S = S1 + S2 + S3
        # Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in dl):
            r_track = - Sum_d/max(dl)
            rewards[0:self.num_agents-2] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in dl)):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:self.num_agents-2] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in dl):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:self.num_agents-2] += mu3*r_capture
        
        ## 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in dl):
            rewards[0:self.num_agents-2] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
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
        for i in range(self.num_agents - 1):
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
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

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

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
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

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

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
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))  #障碍物的坐标
        angle = np.random.uniform(0, 2 * np.pi)  #障碍物的运动方向
        # speed = 0.03 
        speed = 0.00 # 障碍物的速度，为0表示静止
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)]) #障碍物的速度，包含x和y方向的速度
        self.radius = np.random.uniform(0.1, 0.15)   #障碍物的半径