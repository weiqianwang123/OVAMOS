import pomdp_py
import math
import random
import numpy as np
from vlfm.policy.OVAMOS.oo_pomdp.domain.state import *
from vlfm.policy.OVAMOS.oo_pomdp.domain.action import *
from vlfm.policy.OVAMOS.oo_pomdp.domain.observation import *
from vlfm.policy.OVAMOS.oo_pomdp.utils.map import Map

class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, map, objid, sensor, dim, sigma=1.0, 
                 alpha=0.5, beta=0.5, gamma=0.5, theta=0.5):
        """
        初始化观测模型
        
        参数:
            map: 地图对象
            objid: 目标ID
            sensor: 传感器对象（包含 robot_id、fov、sensing_region_size 等信息）
            dim: 世界的尺寸 (width, height)
            sigma: 高斯噪声标准差（用于检测时的噪声）
            alpha: 当目标在 fov 内且检测到时的概率因子（事件 A）
            beta: 当目标在 fov 内但漏检时的概率因子（事件 B）
            gamma: 当目标不在 fov 内但发生误检时的概率因子（事件 C）
            theta: 当目标不在 fov 内且正确未检测时的概率因子（事件 D）
        """
        self._map = map
        self._objid = objid
        self._sensor = sensor
        self._dim = dim
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        
    def update_map(self, new_map):
        self._map = new_map
        
    def probability(self, observation, next_state, action, **kwargs):
        """
        计算观测概率 P(o | s', a)
        
        利用混合模型，设 value 为当前 fov 内包含目标的概率（0~1）：
        
          - 若观测为非 NULL（即检测到一个观测值 o）：
              * 当目标在 fov 内（概率为 value）：事件 A，检测到目标，
                观测服从以目标位置为均值、标准差为 sigma 的高斯分布，概率为
                  alpha * Gaussian(o; object_pose, sigma)
              * 当目标不在 fov 内（概率为 1-value）：事件 C，误检，
                认为误检在感知区域内均匀分布，概率为
                  gamma * (1/sensing_area)
              综合：
                P(o|s,a) = value * (alpha * Gaussian) + (1 - value) * (gamma / sensing_area)
          
          - 若观测为 NULL（即未检测到目标）：
              * 当目标在 fov 内：事件 B，漏检，概率为 beta
              * 当目标不在 fov 内：事件 D，正确未检测，概率为 theta
              综合：
                P(NULL|s,a) = value * beta + (1 - value) * theta
        """
        # 获取机器人和物体的位置
        # next_robot_state = kwargs.get("next_robot_state", None)
        # if next_robot_state is not None:
        #     robot_pose = next_robot_state.pose
        #     if isinstance(next_state, ObjectState):
        #         object_pose = next_state.pose
        #     else:
        #         object_pose = next_state.pose(self._objid)
        # else:
        #     robot_pose = next_state.pose(self._sensor.robot_id)
        #     object_pose = next_state.pose(self._objid)
        
        # # 当前 fov 包含目标的概率（value），取值范围 [0,1]
        # value = self._map.compute_best_in_fov(robot_pose, self._sensor.fov)
        
        # # 传感器感知区域面积（用于均匀分布计算误检情况）
        # sensing_area = self._sensor.sensing_region_size()
        # is_in_region = self._map.within_fov(robot_pose, object_pose, self._sensor.fov)
        # if observation.pose != ObjectObservation.NULL:
        #     # 非 NULL 观测（有检测值 o）
        #     # 事件 A：检测到目标（目标在 fov 内）
        #     # gaussian = pomdp_py.Gaussian(
        #     #     list(object_pose),
        #     #     [[self.sigma**2, 0], [0, self.sigma**2]]
        #     # )
        #     # p_A = self.alpha * gaussian[observation.pose]
        #     # # 事件 C：误检（目标不在 fov 内），假设均匀分布在感知区域内
        #     # p_C = self.gamma * (1.0 / sensing_area)
            
        #     # prob = value * p_A + (1 - value) * p_C
        #     if is_in_region:
        #         prob = 1
        #     else:
        #         prob = 0
        # else:
        #     # NULL 观测（未检测到目标）
        #     # 事件 B：目标在 fov 内但漏检
        #     # p_B = self.beta
        #     # # 事件 D：目标不在 fov 内且正确未检测
        #     # p_D = self.theta
            
        #     # prob = value * p_B + (1 - value) * p_D
        #     if is_in_region:
        #         prob = value
        #     else:
        #         prob = 1
        prob = 0.5
        return prob
    
    def sample(self, next_state, action, **kwargs):
        """
        根据混合的观测模型采样一个观测值，不使用阈值判断，而是
        直接计算各事件的概率，然后随机采样。
        """
        # 获取机器人和物体的位置
        # robot_pose = next_state.pose(self._sensor.robot_id)
        # object_pose = next_state.pose(self._objid)
        
        # # 由 VLM 得到当前 fov 内存在目标的置信度
        # value = self._map.compute_best_in_fov(robot_pose, self._sensor.fov)
        # is_in_region = self._map.within_fov(robot_pose, object_pose, self._sensor.fov)
        # alpha = 0
        # if is_in_region:
        #     alpha = 0.5
        # else:
        #     alpha = 0
        
        # p_A = 0  
        # p_B = 1
        # p_C = 0
        # p_D = 0
        # # 计算四个事件的概率
        # # p_A = value * self.alpha     # 目标在 fov 内且检测到（高斯采样）
        # # p_B = value * self.beta      # 目标在 fov 内但漏检（返回 NULL）
        # # p_C = (1 - value) * self.gamma  # 目标不在 fov 内但误检（从感知区域均匀采样）
        # # p_D = (1 - value) * self.theta  # 目标不在 fov 内且正确未检测（返回 NULL）
        
        # # 使用四个事件的概率直接采样事件
        # event = random.choices(["A", "B", "C", "D"],
        #                     weights=[p_A, p_B, p_C, p_D],
        #                     k=1)[0]
        # # print("event:",event,"value:",value,"action:",action.name)
        # if event == "A":
        #     # 事件 A：检测到目标，使用高斯分布采样观测值
        #     gaussian = pomdp_py.Gaussian(
        #         list(object_pose),
        #         [[self.sigma**2, 0], [0, self.sigma**2]]
        #     )
        #     zi = gaussian.random()
           
        # elif event == "B":
        #     # 事件 B：漏检，返回 NULL
        #     zi = ObjectObservation.NULL
        # elif event == "C":
        #     robot_x, robot_y, robot_theta = robot_pose
        #     # 事件 C：误检，随机在感知区域内采样一个点
        #     fov_angle = self._sensor.fov          # 传感器的视野角度（弧度），例如 60° 对应约 1.047 弧度
        #     min_range = self._sensor.min_range      # 传感器的最小检测距离
        #     max_range = self._sensor.max_range      # 传感器的最大检测距离

        #     # 在 [-fov_angle/2, fov_angle/2] 内采样一个相对于机器人正前方的角度
        #     sampled_angle = random.uniform(-fov_angle/2, fov_angle/2)

        #     # 在 [min_range, max_range] 内采样一个距离
        #     sampled_distance = random.uniform(min_range, max_range)

        #     # 计算采样点在机器人局部坐标系中的坐标
        #     local_x = sampled_distance * math.cos(sampled_angle)
        #     local_y = sampled_distance * math.sin(sampled_angle)

        #     # 将局部坐标转换为全局坐标
        #     global_x = robot_x + (local_x * math.cos(robot_theta) - local_y * math.sin(robot_theta))
        #     global_y = robot_y + (local_x * math.sin(robot_theta) + local_y * math.cos(robot_theta))

        #     # 得到最终的采样观测位置
        #     zi = (int(round(global_x)), int(round(global_y)))
        # else:  # event == "D"
        #     # 事件 D：正确未检测到，返回 NULL
        #     zi = ObjectObservation.NULL
        zi = ObjectObservation.NULL
        return ObjectObservation(self._objid, zi)
    
    def argmax(self, next_state, action, **kwargs):
        """
        返回最可能的观测值（最大后验估计）
        
        策略：
         - 如果 value 较高（目标可能在 fov 内），并且 alpha >= beta，
           则返回事件 A 的 MPE（最大后验估计，即高斯分布的均值附近）；
         - 否则返回 NULL。
        """
        # robot_pose = next_state.pose(self._sensor.robot_id)
        # object_pose = next_state.pose(self._objid)
        
        # value = self._map.compute_best_in_fov(robot_pose, self._sensor.fov)
        # is_in_region = self._map.within_fov(robot_pose, object_pose, self._sensor.fov)
        # if is_in_region:
        #         gaussian = pomdp_py.Gaussian(
        #             list(object_pose),
        #             [[self.sigma**2, 0], [0, self.sigma**2]]
        #         )
        #         zi = gaussian.mpe()  # 最大后验估计
        #         zi = (int(round(zi[0])), int(round(zi[1])))
           
        # else:
        zi = ObjectObservation.NULL
        
        return ObjectObservation(self._objid, zi)