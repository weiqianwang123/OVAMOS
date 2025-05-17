import numpy as np
import math
from vlfm.policy.OVAMOS.oo_pomdp.domain.state import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow, Wedge, Circle
import os
from typing import Optional, Callable, Tuple, List, Union
from sklearn.cluster import DBSCAN
from vlfm.policy.OVAMOS.oo_pomdp.utils.rrt_planner import RRTPlanner
from vlfm.policy.OVAMOS.oo_pomdp.utils.astar_planner import AStarPlanner
from collections import deque
class Map:
    def __init__(self,sensor, width=1000, length=1000,pixel_per_meter=20,value_discount=0.5):
        """
        初始化障碍物检测器
        :param width: 地图宽度
        :param length: 地图长度
        """
        self._sensor = sensor
        self.width = width
        self.length = length
        self.pixel_per_meter = pixel_per_meter
        self.obstacle_map = np.zeros((width, length), dtype=np.uint8)  # 0 代表无障碍物，1 代表有障碍物
        self.value_map = np.zeros((width, length), dtype=np.uint8)
        self.frontiers = []
        self._episode_pixel_origin = np.array([length // 2, width  // 2])
        self._value_discount = value_discount
        self._rrt_planner = None
        self._astar_planner = None
        self.navigate_mode = False
        self.navigate_goal = None
    
    def vis_multi_value_map(self,value_maps):
        # Extract the number of channels from the third dimension
        height, width, num_channels = value_maps.shape

        for c in range(num_channels):
            # Create a new figure for channel c
            plt.figure()
            
            # Display the c-th channel of the value map
            im = plt.imshow(value_maps[:, :, c], cmap='jet', origin='upper')
            
            # Set plot title
            plt.title(f"Value Map Channel {c}")
            
            # Add a colorbar
            plt.colorbar(im, shrink=0.8)
            
            # Save to disk as PNG (e.g. value_map_channel_0.png, value_map_channel_1.png, ...)
            plt.savefig(f"value_map_channel_{c}.png", bbox_inches='tight')
            
            # Close the figure to free memory
            plt.close()


    def set_navigate_goal(self,goal):
        px,py = self._xy_to_px(np.array([[goal[0], goal[1]]]))[0]
        if self.obstacle_map[px, py] == 0 :
            print("Goal Adjusted")
            self._rrt_planner = RRTPlanner(self.obstacle_map)
            self._rrt_planner.build_rrt()
            px,py = self._rrt_planner.find_nearest_navigable_node((px,py))
        x,y = self._px_to_xy(np.array([[px, py]]))[0]

        self.navigate_mode = True
        self.navigate_goal = (x,y)
    def reset_navigate(self):
        self.navigate_mode = False
        self.navigate_goal = None

    def get_candidate_points(self, robot_pose=None):
        """
        获取候选点：
        1. 将 frontiers 和 prior 里的点合并。
        2. 如果提供了 robot_pose，则按照距离进行排序。

        Args:
            robot_pose (tuple, optional): 机器人的当前位置 (x, y)。

        Returns:
            list: 排序后的候选点列表 [(x, y), ...]
        """
        # **确保 frontiers 是列表，并转换为 tuple**
        frontiers_set = {tuple(point) for point in self.frontiers}  # 转换为哈希类型
        prior_set = {tuple(point) for point in self.prior.keys()}  # 转换为哈希类型

        # 1️⃣ 合并 frontiers 和 prior 目标点
        combined_points = list(frontiers_set | prior_set)  # 用集合去重后转换回列表

        if robot_pose is not None:
            # 2️⃣ 按照机器人当前位置计算欧几里得距离，并排序
            robot_xy = np.array(robot_pose[:2])  
            
            filtered_points = [
            p for p in combined_points if np.linalg.norm(np.array(p) - robot_xy) >= 0.001
            ]

            # 3️⃣ 按照距离排序
            filtered_points.sort(key=lambda p: np.linalg.norm(np.array(p) - robot_xy))
        
            return filtered_points

        return combined_points




    def check_navigate_success(self, robot_pose):
        """
        检查机器人是否成功到达导航目标。
        
        Args:
            robot_pose (tuple): 机器人当前位置 (x, y)（单位：米）

        Returns:
            bool: 如果机器人在目标点 1.5m 范围内，则返回 True，否则返回 False
        """
        if self.navigate_goal is None:
            return False  # 没有目标，直接返回 False
        self.set_navigate_goal(self.navigate_goal)
        goal_x, goal_y = self.navigate_goal
        robot_x, robot_y = robot_pose[:2]  # 取前两个值，忽略角度 θ（如果有）

        # 计算欧几里得距离
        distance = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))

        return distance < 1 # 如果距离小于 1m，返回 True

        
    
    def _pixel_value_within_radius(self,value_map: np.ndarray, center_px: Tuple[int, int], radius_px: int) -> float:
        """
        在给定的二维 value_map 中，以 center_px 为中心，计算圆形区域内（半径为 radius_px）的像素值的最大值。

        Args:
            value_map (np.ndarray): 输入的二维数组，表示地图或图像的数值信息。
            center_px (Tuple[int, int]): 中心点的像素坐标，格式为 (row, col)。
            radius_px (int): 圆形区域的半径，单位为像素。

        Returns:
            float: 圆形区域内的最大像素值；若区域为空，则返回 0.0。
        """
        r_center, c_center = center_px

        # 计算候选区域的边界（矩形包围盒），确保不超出图像范围
        r_min = max(r_center - radius_px, 0)
        r_max = min(r_center + radius_px + 1, value_map.shape[0])
        c_min = max(c_center - radius_px, 0)
        c_max = min(c_center + radius_px + 1, value_map.shape[1])
        
        # 截取出矩形区域
        region = value_map[r_min:r_max, c_min:c_max]
        
        # 构造与 region 尺寸相同的坐标网格，计算每个像素与中心点的欧式距离是否在 radius_px 内
        rows = np.arange(r_min, r_max)[:, None]  # 转换为列向量
        cols = np.arange(c_min, c_max)            # 行向量
        # 计算每个点与中心点的距离平方
        dist_sq = (rows - r_center) ** 2 + (cols - c_center) ** 2
        # 构造掩码，判断哪些像素位于圆形区域内
        mask = dist_sq <= radius_px ** 2
        
        # 根据掩码取出圆形区域内的像素值
        if np.any(mask):
            region_in_circle = region[mask]
            return float(np.max(region_in_circle)) if region_in_circle.size > 0 else 0.0
        else:
            return 0.0
    def _get_value(self,point: np.ndarray,radius_px) -> Union[float, Tuple[float, ...]]:
            x, y = point
            point_px =  self._xy_to_px(np.array([[x, y]]))[0]
            # 如果只有1个价值通道，直接采样；如果有多个，则返回所有通道的数值
            all_values = [
                self._pixel_value_within_radius(self.value_map, point_px, radius_px)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)
    def _sort_waypoints(
        self, waypoints: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, List[float]]:
        """
        根据给定的 waypoints（frontier 点）及半径，在地图上采样价值，
        返回按价值降序排序后的 frontier 点及其对应的价值。
        """
        # 将半径从米转换为像素
        radius_px = int(radius * self.pixel_per_meter)

        

        # 对所有 frontier 点计算其价值
        values = [self._get_value(point, radius_px) for point in waypoints]


        # 使用 numpy.argsort 对价值进行降序排序（最高价值对应的索引在最前面）
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values


    def save_maps(self, belief, objid,robot_id,robot_pose=None,detected_pose=None, save_path="map_visualization.png"):
        """
        生成并保存 value_map、obstacle_map 和 belief_map（信念地图）到本地文件。

        Args:
            belief: 机器人对目标位置的信念（`pomdp_py.OOBelief`）。
            robot_pose: (x, y) 机器人位置（单位：米），如果提供则在地图上标记。
            save_path: 保存的文件路径（支持 .png, .jpg, .pdf 等格式）。
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 绘制障碍物地图
        axes[0].imshow(self.obstacle_map, cmap="gray", origin="upper")
        axes[0].set_title("Obstacle Map ")

        # 绘制价值地图
        im1 = axes[1].imshow(self.value_map, cmap="jet", origin="upper")
        axes[1].set_title("Value Map ")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)



         # 在 obstacle_map 上绘制 prior 点，置信度越高颜色越深
        # if self.prior:
        #     max_confidence = max(self.prior.values())  # 归一化到 [0,1] 范围
        #     for (x, y), confidence in self.prior.items():
        #         px = self._xy_to_px(np.array([[x, y]]))[0]
        #         row, col = px
        #         color_intensity = confidence / max_confidence  # 归一化颜色深度
        #         triangle = plt.Polygon(
        #             [(col, row - 5), (col - 4, row + 4), (col + 4, row + 4)],
        #             color=(1,1,1,1),  # 置信度越高，颜色越深的红色
        #         )
        #         axes[1].add_patch(triangle)

      
        # 如果提供了机器人位置，在所有地图上标注
        # 如果提供了 robot_pose 和 fov，则在障碍物地图上绘制机器人的视野
        if robot_pose is not None and len(robot_pose) >= 3:
            # 将机器人位姿转换为像素坐标
            robot_px = self._xyz_to_px(np.array([[robot_pose[0], robot_pose[1],robot_pose[2]]]))[0]
            r_px, c_px,theta= robot_px

            # 计算传感器的外半径和内半径（单位：像素）
            outer_radius = self._sensor.max_range * self.pixel_per_meter
            inner_radius = self._sensor.min_range * self.pixel_per_meter
            wedge_width = outer_radius - inner_radius  # 扇形宽度

            # 将机器人的朝向和 FOV 转换为角度（度数）
            theta_deg = math.degrees(theta)
            half_fov_deg = math.degrees(self._sensor.fov) / 2.0

            # 使用 Wedge 绘制环形扇区
            # 注意：Wedge 的中心在 (x, y) 图像坐标中，x 对应列，y 对应行
            wedge = Wedge(center=(c_px, r_px),
                        r=outer_radius,
                        theta1=theta_deg - half_fov_deg,
                        theta2=theta_deg + half_fov_deg,
                        width=wedge_width,
                        facecolor="none",
                        edgecolor="red",
                        linewidth=1,
                        alpha=0.7)
            axes[0].add_patch(wedge)


            # 同时在障碍物地图上标记机器人的位置（例如用黄色圆点）
            axes[0].scatter(c_px, r_px, c="blue", marker="o", s=10, label="Robot")
            axes[0].legend()
        if self.navigate_goal is not None:
            # detected_pose 格式：(x, y)
            detected_px = self._xy_to_px(np.array([[self.navigate_goal[0], self.navigate_goal[1]]]))[0]
            dr, dc = detected_px
            # 为防止分辨率过高，设置一个半径（例如 10 像素）高亮周围区域
            highlight_radius = 4
            circle = Circle((dc, dr), radius=highlight_radius, edgecolor="lime", facecolor="none", linewidth=3)
            axes[0].add_patch(circle)
            # 在其他图中也可标注
            # for ax in [axes[1], axes[2]]:
            #     ax.scatter(dc, dr, c="lime", marker="x", s=100, label="Detected")
            #     ax.legend()
        
        if self.frontiers:
        # 假设 self.frontiers 为 [(x, y), ...] 形式的列表
            frontier_array = np.array(self.frontiers)
            # 转换到像素坐标，注意返回值每行格式为 [row, col]
            frontier_px = self._xy_to_px(frontier_array)
            axes[0].scatter(frontier_px[:, 1], frontier_px[:, 0], c="green", marker="o", s=5, label="Frontier")
            axes[0].legend()


        for ax in axes:
            ax.invert_yaxis()
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存图像
        plt.savefig(save_path, dpi=300)
        plt.close(fig)  # 关闭图像，避免内存占用

        print(f"地图已保存到 {save_path}")



        

    def update(self, new_obstacle_map,new_value_map,new_frontiers,select_best=True):
        """
        更新障碍物地图
        :param new_map: 1000x1000 的 numpy 数组，包含最新信息
        """
        assert new_obstacle_map.shape == (self.width, self.length), "Obstacle map shape mismatch!"
        assert new_value_map.shape == (self.width, self.length), "Value map shape mismatch!"
        self.obstacle_map =new_obstacle_map
        self.value_map = new_value_map
        self.frontiers = new_frontiers
        # self._rrt_planner = RRTPlanner(self.obstacle_map)
        # self._rrt_planner.build_rrt()
        # self._astar_planner = AStarPlanner(self.obstacle_map)
        if select_best and len(self.frontiers) > 0:
            # 将 frontiers 转换为 numpy 数组（假设每个 frontier 为 [x, y]）
            print("have frontier!")
            waypoints_arr = np.array(self.frontiers)
            sorted_frontiers, sorted_values = self._sort_waypoints(waypoints_arr, radius=0.5)
            best_frontier = sorted_frontiers[0]
            px,py = self._xy_to_px(np.array([[best_frontier[0], best_frontier[1]]]))[0]
            if self.obstacle_map[px, py] == 0 :
                print("Frontier Adjusted")
                self._rrt_planner = RRTPlanner(self.obstacle_map)
                self._rrt_planner.build_rrt()
                px,py = self._rrt_planner.find_nearest_navigable_node((px,py))
            x,y = self._px_to_xy(np.array([[px, py]]))[0]
            self.frontiers = [(x,y)]
        print("map updated in pomdp")

    def get_best_point_in_value(self, num_samples=10):
        threshold = 0.05
        indices = np.argwhere(self.value_map > threshold)
        if len(indices) == 0:
            print("Warning: No valid high-value areas found in value_map.")
            return {}
        
        # 将像素坐标转换为 (x, y) 坐标
        xy_candidates = self._px_to_xy(indices)
        xy_candidates = np.array(xy_candidates)
        
        # DBSCAN 聚类
        clustering = DBSCAN(eps=0.1, min_samples=1).fit(xy_candidates)
        labels = clustering.labels_
        
        prior = {}
        unique_labels = set(labels)
        best_point_in_all = None
        best_value_in_all = -np.inf
        # 对每个聚类，选取 value_map 值最高的点
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            best_point = None
            best_value = -np.inf
            for idx in cluster_indices:
                row, col = indices[idx]
                point_value = float(self.value_map[row, col])
                if point_value > best_value:
                    best_value = point_value
                    best_point = self._px_to_xy(np.array([[row, col]]))[0]
                    best_px, best_py = row, col
            if best_value > best_value_in_all:
                best_point_in_all =  best_point
        px,py = self._xy_to_px(np.array([[best_point_in_all[0],best_point_in_all[1]]]))[0]
        if self.obstacle_map[px, py] == 0 :
            self._rrt_planner = RRTPlanner(self.obstacle_map)
            self._rrt_planner.build_rrt()
            px,py = self._rrt_planner.find_nearest_navigable_node((px,py))
            x,y = self._px_to_xy(np.array([[px, py]]))[0]
            best_point_in_all = (x,y)
        return best_point_in_all
    def get_prior_with_value_map(self, num_samples=10):
        threshold = 0.05
        indices = np.argwhere(self.value_map > threshold)
        if len(indices) == 0:
            print("Warning: No valid high-value areas found in value_map.")
            return {}
        
        # 将像素坐标转换为 (x, y) 坐标
        xy_candidates = self._px_to_xy(indices)
        xy_candidates = np.array(xy_candidates)
        
        # DBSCAN 聚类
        clustering = DBSCAN(eps=0.1, min_samples=1).fit(xy_candidates)
        labels = clustering.labels_
        
        prior = {}
        unique_labels = set(labels)
        
        # 对每个聚类，选取 value_map 值最高的点
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            best_point = None
            best_value = -np.inf
            for idx in cluster_indices:
                row, col = indices[idx]
                
                point_value = float(self.value_map[row, col])
                if point_value > best_value:
                    best_value = point_value
                    best_point = self._px_to_xy(np.array([[row, col]]))[0]
                    best_px, best_py = row, col
            if best_point is not None:
                prior[tuple(best_point)] = best_value
        # if len(prior)==1:
        #     only_key,only_value = list(prior.items())[0]
        #     prior[only_key] = only_value
        #     new_key = (only_key[0]+0.1,only_key[1]+0.1)
        #     prior[new_key] = only_value/2
        self.prior = prior
        print("PRIOR BY VALUE MAP:",prior)
        return prior

    

    def compute_best_in_fov(self,robot_pose,fov=None):
        """
        计算给定 1000x1000 value_map 中位于当前机器人视野内的所有值的平均值。

        机器人位姿 robot_pose 格式为 (r_x, r_y, r_theta)，单位为米和弧度。
        value_map 为以像素为单位的二维数组。

        返回：
            float: 机器人视野内所有点的平均值。如果视野内没有任何点，则返回 np.nan。
        """
        if fov is None:
            fov = self._sensor.fov
        rx, ry, rth = robot_pose

        # 将机器人 (x, y) 坐标（单位：米）转换为像素坐标，注意 _xy_to_px 返回格式为 (row, col)
        robot_px = self._xyz_to_px(np.array([[rx,ry,rth]]))[0]  # 例如 robot_px = [r_px, c_px]
        r_px, c_px ,rth= robot_px  # r_px: 行坐标, c_px: 列坐标

        # 获取 value_map 的尺寸（单位：像素）
        height, width = self.value_map.shape

        # 构建每个像素的坐标网格
        # 注意：此处 x 对应列索引，y 对应行索引
        x_coords = np.arange(width)   # 列索引
        y_coords = np.arange(height)  # 行索引
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')  # X: 每个像素的列坐标, Y: 行坐标

        # 计算每个像素与机器人的差异（以像素为单位）
        dx = X - c_px  # 列方向的差异
        dy = Y - r_px  # 行方向的差异
        distances = np.sqrt(dx**2 + dy**2)  # 像素距离

        # 将传感器的距离范围从米转换为像素
        min_range_px = self._sensor.min_range * self.pixel_per_meter
        max_range_px = self._sensor.max_range * self.pixel_per_meter

        # 生成距离范围掩码
        mask_range = (distances >= min_range_px) & (distances <= max_range_px)

        # 计算每个像素相对于机器人的方位角
        # np.arctan2 返回范围 [-π, π]，减去 rth 后归一化到 [0, 2π)
        angles = (np.arctan2(dy, dx) - rth) % (2 * math.pi)
        half_fov =  fov / 2  # 视野的一半（单位：弧度）
        # 生成视野范围掩码：角度在 [0, half_fov] 或 [2π - half_fov, 2π)
        mask_fov = (angles <= half_fov) | (angles >= (2 * math.pi - half_fov))

        # 综合距离和视野的掩码
        mask = mask_range & mask_fov

        # 如果视野内没有任何点，返回 np.nan
        if not np.any(mask):
            print("no points!")
            return 0

        # 计算视野内所有点的均值
        # best_value = np.mean(self.value_map[mask])
        best_value = np.max(self.value_map[mask])
        return best_value*self._value_discount


    def within_fov(self, robot_pose, target_pose, fov=None):
        """
        判断目标点是否在机器人的视野 (FOV) 内。

        Args:
            robot_pose: (r_x, r_y, r_theta) 机器人位姿，单位为米和弧度。
            target_pose: (t_x, t_y) 目标点的坐标，单位为米。
            fov: 视野角度，单位为弧度。

        Returns:
            bool: 如果目标点在 FOV 内，返回 True，否则返回 False。
        """
        # 机器人位置和朝向
        if fov is None:
            fov = self._sensor.fov
        rx, ry, rth = robot_pose
        tx, ty = target_pose

        # 计算目标点到机器人的向量 (dx, dy)
        dx = tx - rx
        dy = ty - ry

        # 计算目标点的欧几里得距离
        distance = np.sqrt(dx**2 + dy**2)

        # 计算传感器的有效范围
        min_range = self._sensor.min_range
        max_range = self._sensor.max_range

        # 目标点必须在传感器的有效范围内
        if not (0.1 <= distance <= max_range):
            return False

        # 计算目标点相对于机器人的角度（归一化到 [-π, π]）
        angle_to_target = np.arctan2(dy, dx) - rth
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi

        # 目标点的角度需要在 [-fov/2, fov/2] 之间
        return abs(angle_to_target) <= (fov / 2)







    def valid_pose(self, pose, check_collision=True, save_path="/home/yfx/vlfm/output_frames_POMDP/valid_pose.png"):
        """
        检查给定的 pose (x, y) 是否有效，并可视化:
        - 绘制原始 obstacle_map
        - 标注机器人位置和半径区域
        - **先画完，再翻转**
        - 保存图片
        """
        if not self.in_boundary(pose):
            return False

        x, y = pose[:2]
        pts = np.array([[x, y]])
        center_px = self._xy_to_px(pts)[0]  # 转换为像素坐标
        row_center, col_center = center_px

        # 机器人半径（像素）
        radius_pixels = 2

        # # 创建 Matplotlib 图像
        # fig, ax = plt.subplots(figsize=(8, 8))
        
        # # **先画原始地图**
        # ax.imshow(self.obstacle_map, cmap="gray", origin="upper")  # 不翻转，先画完

        # # **画机器人位置（红色 ）**
        # ax.plot(col_center, row_center, 'ro', markersize=6, label="Robot")

        # # **画机器人影响范围（蓝色半透明圆）**
        # circle = plt.Circle((col_center, row_center), radius_pixels, color='blue', alpha=0.3, label="Radius")
        # ax.add_patch(circle)

        # 碰撞检测
        valid = True
        for i in range(-radius_pixels, radius_pixels + 1):
            for j in range(-radius_pixels, radius_pixels + 1):
                if i**2 + j**2 <= radius_pixels**2:  # 仅检查半径范围内的像素
                    r = row_center + i
                    c = col_center + j
                    if r < 0 or r >= self.obstacle_map.shape[0] or c < 0 or c >= self.obstacle_map.shape[1]:
                        valid = False
                        break
                    if self.obstacle_map[r, c] == 0:  # 0 代表障碍物
                        valid = False
                        break

        # **翻转整个图像**
        # ax.invert_yaxis()  # 画完再翻转，确保 (0,0) 在左下角

        # # **标记有效/无效状态**
        # ax.set_title("Valid Pose ✅" if valid else "Invalid Pose ❌ (Collision)")
        # ax.legend()

        # # 保存并关闭
        # plt.savefig(save_path)
        # plt.close()
        # print(f"机器人位置可视化已保存: {save_path}")

        return valid


    def valid_motions(self, robot_pose, all_motion_actions):
        """
        检查从当前机器人位姿出发，所有候选运动动作是否合法。
        robot_pose 为 (x, y, θ)，单位：米和弧度；
        all_motion_actions 中的 forward 以米为单位，angle 单位为弧度。
        返回满足条件的动作集合。
        """
        rx, ry, rth = robot_pose
        valid = set()
        for motion_action in all_motion_actions:
            forward, angle = motion_action.motion()
            new_rth = rth + angle
            new_rx = rx + forward * math.cos(new_rth)
            new_ry = ry + forward * math.sin(new_rth)
            rth = rth % (2 * math.pi)
            next_pose = (new_rx, new_ry, new_rth)
            if next_pose != robot_pose and self.valid_pose(next_pose):
                valid.add(motion_action)
        
        # print("VALID",valid)
        return valid

    def get_distance_heading(self,robot_pose,target_pose):
        robot_px = self._xy_to_px(np.array([[robot_pose[0], robot_pose[1]]]))[0]
        target_px = self._xy_to_px(np.array([[target_pose[0], target_pose[1]]]))[0]
        # path = self._rrt_planner.find_path(robot_px, target_px, save_path="/home/yfx/vlfm/output_frames_POMDP/rrt_path.png")
        start = tuple(map(int, robot_px))  # 变成 (x, y) 形式
        goal = tuple(map(int, target_px))  # 变成 (x, y) 形式

        path = self._astar_planner.find_path(start, goal)
        if path is not None:
            path_length_meters = sum(
                np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) 
                for i in range(len(path) - 1)
            ) / self.pixel_per_meter
        else:
            path_length_meters = 0


        # robot_pxz = self._xyz_to_px(np.array([[robot_pose[0], robot_pose[1],robot_pose[2]]]))[0]
        #  # 计算路径朝向（heading of path）
        # start_px = np.array(path[0])
        # next_px = np.array(path[1])  # 取路径的下一个点
        # path_heading = np.arctan2(next_px[1] - start_px[1], next_px[0] - start_px[0])

        # # 计算路径朝向与机器人朝向的偏差 Δθ
        # robot_heading = robot_pxz[2]  # 机器人当前朝向（弧度）
        # heading_deviation = path_heading - robot_heading


        # # print("robot_heading",robot_heading,"heading_deviation",heading_deviation,"robot_pose",robot_pose)
        # # 归一化到 [-π, π]
        # heading_deviation = (heading_deviation + np.pi) % (2 * np.pi) - np.pi
        return path_length_meters,0
    
    

    def in_boundary(self, pose):
        """
        检查 pose 是否在地图边界内。这里 pose 为 (x, y) 或 (x, y, θ)，单位为米和弧度。
        利用 _xy_to_px 将米坐标转换为像素坐标，然后判断是否在地图尺寸范围内。
        """
        # 先取出前两个坐标 (x, y)（单位：米）
        x, y = pose[:2]
        # 转换为像素坐标
        pts = np.array([[x, y]])
        px = self._xy_to_px(pts)[0]  # 得到 (row, col)
        row, col = px
        # 检查是否在 [0, width) 和 [0, length) 内
        if 0 <= row < self.width and 0 <= col < self.length:
            # 如果有角度信息，检查角度范围
            if len(pose) == 3:
                th = pose[2]
                if not (0 <= th <= 2 * np.pi):
                    return False
            return True
        return False


    def _xy_to_px(self, points: np.ndarray, pixel_per_meter=None, is_belief=False) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        if pixel_per_meter is None:
            pixel_per_meter = self.pixel_per_meter
        if is_belief:
            episode_pixel_origin = (50 // 2, 50 // 2)
            upside = 50
            width = 50  # 假设 belief map 也是 50x50
        else:
            episode_pixel_origin = self._episode_pixel_origin
            upside = self.value_map.shape[0]
            width = self.value_map.shape[1]  # 获取地图的宽度

        # px = np.rint(points[:, ::-1] * pixel_per_meter) + episode_pixel_origin
        px = np.rint(points * pixel_per_meter) + episode_pixel_origin

        # 上下翻转：对 row 坐标进行翻转
        px[:, 1] = upside - px[:, 1]

        return px.astype(int)  # 直接返回 (col, row)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()

        # 获取地图的高度
        upside = self.value_map.shape[0]

        # 上下翻转：恢复被 `_xy_to_px()` 变换的 row 坐标
        px_copy[:, 1] = upside - px_copy[:, 1]

        # 还原像素坐标到实际世界坐标
        points = (px_copy - self._episode_pixel_origin) / self.pixel_per_meter

        return points  # 直接返回 (x, y)


      


    def _xyz_to_px(self, points: np.ndarray) -> np.ndarray:
        """
        Converts an array of (x, y, theta) world coordinates to pixel coordinates (px, py, ptheta).
        
        Args:
            points (np.ndarray): A NumPy array of shape (N, 3), where each row is (x, y, theta).

        Returns:
            np.ndarray: A NumPy array of shape (N, 3), where each row is (px, py, ptheta).
        """
        assert points.shape[1] == 3, "Input points must have shape (N, 3) for (x, y, theta)"

        # 提取 (x, y) 部分并转换为 (px, py)
        xy_points = points[:, :2]
        px_py = self._xy_to_px(xy_points)  # 调用已有的坐标转换方法

        # 处理朝向 theta (角度转换)
        theta = points[:, 2]+math.pi/2  # 提取 theta（弧度制）
        ptheta = theta  # 角度翻转

        # 组合成 (px, py, ptheta)
        pxyz = np.column_stack((px_py, ptheta))
        return pxyz

    def _px_to_xyz(self, px: np.ndarray) -> np.ndarray:
        """
        Converts an array of (px, py, ptheta) pixel coordinates back to world coordinates (x, y, theta).

        Args:
            px (np.ndarray): A NumPy array of shape (N, 3), where each row is (px, py, ptheta).

        Returns:
            np.ndarray: A NumPy array of shape (N, 3), where each row is (x, y, theta).
        """
        assert px.shape[1] == 3, "Input px must have shape (N, 3) for (px, py, ptheta)"

        # 提取 (px, py) 部分并转换回 (x, y)
        px_py = px[:, :2]
        xy_points = self._px_to_xy(px_py)  # 调用已有的方法转换回世界坐标

        # 处理朝向 ptheta (角度转换)
        ptheta = px[:, 2]  # 提取 ptheta（像素坐标中的朝向）
        theta = (2 * np.pi - ptheta) % (2 * np.pi)  # 角度翻转回世界坐标系

        # 组合成 (x, y, theta)
        xyz = np.column_stack((xy_points, theta))
        return xyz

        
    # colors
def lighter(color, percent):
    """assumes color is rgb between (0, 0, 0) and (255, 255, 255)"""
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white - color
    return color + vector * percent