import numpy as np
import math

# 1 simulate lidar
def update_lasers(pos, obs_pos, r, L, num_lasers, bound):

    distance_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    isInObs = distance_to_obs < r \
                or pos[0] < 0 \
                or pos[0] > bound \
                or pos[1] < 0 \
                or pos[1] > bound
    
    if isInObs:
        return [0.0] * num_lasers, isInObs
    
    angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
    laser_lengths = [L] * num_lasers
    
    for i, angle in enumerate(angles):
        intersection_dist = check_obs_intersection(pos, angle, obs_pos, r, L)
        if laser_lengths[i] > intersection_dist:
            laser_lengths[i] = intersection_dist
    
    for i, angle in enumerate(angles):
        wall_dist = check_wall_intersection(pos, angle, bound, L)
        if laser_lengths[i] > wall_dist:
            laser_lengths[i] = wall_dist
    
    return laser_lengths, isInObs

def check_obs_intersection(start_pos, angle, obs_pos,r,max_distance):
    ox = obs_pos[0]
    oy = obs_pos[1]

    end_x = start_pos[0] + max_distance * np.cos(angle)
    end_y = start_pos[1] + max_distance * np.sin(angle)

    dx = end_x - start_pos[0]
    dy = end_y - start_pos[1]
    fx = start_pos[0] - ox
    fy = start_pos[1] - oy

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        if 0 <= t1 <= 1:
            return t1 * max_distance
        if 0 <= t2 <= 1:
            return t2 * max_distance

    return max_distance

def check_wall_intersection(start_pos, angle, bound, L):

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    L_ = L
    #  (y = bound)
    if sin_theta > 0:  
        L_ = min(L_, abs((bound - start_pos[1]) / sin_theta))
    
    #  (y = 0)
    if sin_theta < 0:  
        L_ = min(L_, abs(start_pos[1] / -sin_theta))

    #  (x = bound)
    if cos_theta > 0: 
        L_ = min(L_, abs((bound - start_pos[0]) / cos_theta))
    
    #  (x = 0)
    if cos_theta < 0: 
        L_ = min(L_, abs(start_pos[0] / -cos_theta))

    return L_

def cal_triangle_S(p1, p2, p3):
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S

#计算k架无人机的包围距离
def calculate_RoundUp_distances(k, r):
    if k < 3:
        raise ValueError("k must be greater than 3")

    # 计算圆心到中心点的距离
    d = r / math.sin(math.pi / k)

    # 计算每个圆心的坐标
    angles = [2 * math.pi * i / k for i in range(k)]
    centers = [(d * math.cos(angle), d * math.sin(angle)) for angle in angles]

    # 计算每个圆心到中心点的距离
    distances = [math.hypot(x, y) for (x, y) in centers]

    return distances[0]


#计算围捕点
def generate_points_on_circle(k, center, radius):
    """
    生成均匀分布在圆上的k个点的坐标。

    参数:
    k (int): 需要生成的点的数量。
    center (list): 圆心的坐标 [x, y]。
    radius (float): 圆的半径。

    返回:
    list: 包含k个点的二维列表，每个点的坐标为 [x, y]。
    """
    points = []
    angle_step = 2 * math.pi / k  # 计算每个点之间的角度间隔

    for i in range(k):
        angle = i * angle_step  # 当前点的角度
        x = center[0] + radius * math.cos(angle)  # 计算x坐标
        y = center[1] + radius * math.sin(angle)  # 计算y坐标
        if math.fabs(x) < 1e-12:
            x = 0
        if math.fabs(y) < 1e-12:
            y = 0
        points.append([x, y])  # 将点添加到列表中

    return points


def sort_row_indices_desc(two_d_list):
    """
    对二维列表的每一行按从大到小排序，并返回排序后的列索引列表。

    参数:
    two_d_list (list of lists): 需要排序的二维列表。

    返回:
    list of lists: 每行按从大到小排序后的列索引列表。
    """
    # 初始化结果列表
    sorted_indices = []

    # 遍历每一行
    for row in two_d_list:
        # 使用enumerate获取元素及其索引
        enumerated_row = list(enumerate(row))
        # 根据元素的值进行降序排序，key=lambda x: x[1]表示按值排序
        sorted_enumerated = sorted(enumerated_row, key=lambda x: x[1], reverse=True)
        # 提取排序后的索引
        indices = [index for index, value in sorted_enumerated]
        # 将索引列表添加到结果中
        sorted_indices.append(indices)

    return sorted_indices

#根据所有观测值，计算所有智能体的state和other_state
def compute_state_and_other_state(obs, batchsize=1):
    if batchsize == 1:
        states = []
        other_states = []
        agent_num = len(obs)
        for agent_id in range(agent_num):
            statei = obs[agent_id] #自身位置， 速度（4维）、目标自己的位置（2维）、16个测距传感数据（16维）
            other_statei = []  #其智能体的绝对位置
            for i in range(agent_num):
                if i == agent_id:
                    continue
                other_statei.append([obs[i][0], obs[i][1]])
            states.append(statei)
            other_states.append(other_statei)
        return states, other_states
    else:
        states = []
        other_states = []
        agent_num = len(obs[0])
        for agent_id in range(agent_num):
            statei = obs[:, agent_id, :]  # 自身位置， 速度（4维）、目标自己的位置（2维）、16个测距传感数据（16维）
            states.append(statei)
            obsA = obs.copy()
            obsAi = np.delete(obsA, agent_id, axis=1)
            obsAi = obsAi[:, :, [0, 1]]
            other_states.append(obsAi)
        return states, other_states

#根据所有动作，计算每个智能体的other_actions
def compute_other_actions(actions, batchsize):
    other_actions = []
    agent_num = len(actions[0])
    for agent_id in range(agent_num):
        acA = actions.copy()
        acAi = np.delete(acA, agent_id, axis=1)
        other_actions.append(acAi)
    return other_actions


#根据所有观测值，计算所有智能体的state和other_state
def compute_state_and_other_state_XD(obs, batchsize=1):
    if batchsize == 1:
        states = []
        other_states = []
        agent_num = len(obs)
        for agent_id in range(agent_num):
            statei = obs[agent_id][2:] #自身位置， 速度（4维）、目标自己的位置（2维）、16个测距传感数据（16维）
            other_statei = []  #其智能体的绝对位置
            for i in range(agent_num):
                if i == agent_id:
                    continue
                other_statei.append([obs[i][0], obs[i][1]])
            states.append(statei)
            other_states.append(other_statei)
        return states, other_states
    else:
        states = []
        other_states = []
        agent_num = len(obs[0])
        for agent_id in range(agent_num):
            statei = obs[:, agent_id, 2:]  # 自身位置， 速度（4维）、目标自己的位置（2维）、16个测距传感数据（16维）
            states.append(statei)
            obsA = obs.copy()
            obsAi = np.delete(obsA, agent_id, axis=1)
            obsAi = obsAi[:, :, [0, 1]]
            obsPos = obs.copy()
            obsPosi = obsPos[:, agent_id, [0, 1]]
            obsPosi = np.expand_dims(obsPosi, axis=1)
            obsAi = obsAi - obsPosi
            other_states.append(obsAi)
        return states, other_states



