import numpy as np
from heapq import heappush, heappop
from itertools import permutations
import matplotlib.pyplot as plt
from collections import deque
from itertools import product
class MSPL:
    def __init__(self, env, meters_per_pixel=0.05):
        """
        初始化 MSPL 计算类
        Args:
            env: Habitat-Sim 环境
            meters_per_pixel (float): 地图缩放比例，每像素代表的米数
        """
        self.env = env
        self.meters_per_pixel = meters_per_pixel
        self.pathfinder = env._sim.pathfinder

        # 生成可导航地图
        self.navigable_map = self.pathfinder.get_topdown_view(
            meters_per_pixel=self.meters_per_pixel,
            height=env.current_episode.start_position[1] + 0.2
        )
        


    def find_nearest_valid_point(self, point):
        
        """
        在 Habitat-Sim `PathFinder` 中找到 `point` 附近最近的可导航区域。

        Args:
            point (tuple): 目标点 (x, y)
            radius (float): 搜索半径（米）

        Returns:
            (float, float): 最近的可导航点
            """
        x,z,y = point
        if self.env._sim.pathfinder.is_navigable([x,z, y]):  # 检查是否可行
            return (x,z,y)

        # **使用 Habitat 自带 `get_random_navigable_point` 进行修正**
        nearest_valid = self.env._sim.pathfinder.snap_point([x,z, y])
        if nearest_valid is not None:
            return (nearest_valid[0],nearest_valid[1], nearest_valid[2])  # 返回 (x,z, y)

        return None  # 没有找到可行点（不太可能发生）



    def world_to_map(self, x, y, z):
        """
        将 Habitat 世界坐标 (x, y, z) 转换为地图像素坐标 (row, col)，适配矩形 `map`。

        - **`x` 对应 `col`（左右方向）**
        - **`z` 对应 `row`（前后方向）**
        - **`z` 可能为负数，因此 `row` 计算需要偏移**
        - **适配矩形 `map`，`row` 和 `col` 计算不同**

        Args:
            x, y, z (float): 世界坐标系中的位置

        Returns:
            (int, int): 地图像素坐标 (row, col)
        """
        map_height, map_width = self.navigable_map.shape  # 适配矩形地图

        # **获取 Habitat-Sim 的世界坐标范围**
        if not hasattr(self, "world_origin"):
            self.world_origin = self.env._sim.pathfinder.get_bounds()[0]  # `(x_min, y_min, z_min)`

        # **计算世界坐标相对于地图左上角的偏移**
        x_offset = x - self.world_origin[0]  # `x` 方向偏移
        z_offset = z - self.world_origin[2]  # `z` 方向偏移（Habitat 的 `z` 可能是负的）

        # **转换到地图坐标**
        row = int(z_offset / self.meters_per_pixel)  # `z` 方向 → `row`
        col = int(x_offset / self.meters_per_pixel)  # `x` 方向 → `col`

        # **边界裁剪**
        row = np.clip(row, 0, map_height - 1)
        col = np.clip(col, 0, map_width - 1)

        return row, col





    def astar_search(self, start, goal):
        """
        在 navigable_map 上执行 A* 搜索
        """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(node):
            neighbors = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for d in directions:
                nr, nc = node[0] + d[0], node[1] + d[1]
                if 0 <= nr < self.navigable_map.shape[0] and 0 <= nc < self.navigable_map.shape[1] and self.navigable_map[nr, nc] == 1:
                    neighbors.append((nr, nc))
            return neighbors

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        return None

    def compute_astar_distances(self, waypoints):
        """
        计算所有目标点对之间的 A* 最短路径距离
        """
        N = len(waypoints)
        dist_matrix = np.full((N, N), float("inf"))

        for i in range(N):
            for j in range(N):
                if i != j:
                    path = self.astar_search(waypoints[i], waypoints[j])
                    if path:
                        dist_matrix[i, j] = len(path)

        return dist_matrix

    def solve_open_tsp_astar(self, dist_matrix):
        """
        计算基于 A* 距离的最优目标访问顺序（Open TSP，不返回起点）
        Args:
            dist_matrix (np.ndarray): 目标点对之间的 A* 最短路径矩阵
        Returns:
            list: 访问目标点的最优顺序（不返回起点）
        """
        N = dist_matrix.shape[0]
        if N <= 1:
            return list(range(N))

        min_dist = float("inf")
        best_order = None

        for order in permutations(range(1, N)):  # 不让路径返回 0（机器人起点）
            total_dist = dist_matrix[0, order[0]]  # 从机器人出发
            total_dist += sum(dist_matrix[order[i], order[i + 1]] for i in range(len(order) - 1))

            if total_dist < min_dist:
                min_dist = total_dist
                best_order = [0] + list(order)  # 机器人始终在最前，不回到起点

        return best_order

    def visualize_path(self, path, save_path):
        """
        在 `navigable_map` 上绘制完整路径，并保存到本地。
        
        Args:
            path (list): (row, col) A* 生成的路径点序列
            save_path (str): 本地保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制导航地图
        ax.imshow(self.navigable_map, cmap="gray", origin="upper")
        
        # 取出所有路径点的 (row, col)
        rows, cols = zip(*path)
        
        # **绘制完整路径（红色）**
        ax.plot(cols, rows, color='red', linewidth=2, label="A* Path")

        # **标记起点**
        ax.scatter(cols[0], rows[0], color='blue', s=100, label="Robot Start")

        # **标记终点**
        ax.scatter(cols[-1], rows[-1], color='green', s=100, label="Final Goal")

        # **添加图例**
        ax.legend()
        
        # 保存图片
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Path saved to: {save_path}")


    from itertools import product

    def compute_best_length(self, height_threshold=1, save_path="/home/yfx/vlfm/output_frames_POMDP/mspl_path.png"):
        """
        计算 Habitat-Sim Multi-Object 最短路径，并确保：
        - ✅ 每种类别的目标点只访问一次
        - ✅ 选中的 `goal` 是全局最优的
        - ✅ 计算 A* 规划路径，确保遍历顺序最优
        - ✅ 绘制路径并保存

        Args:
            height_threshold (float): 机器人与目标的最大高度差（米）
            save_path (str): 生成的路径图像保存路径
        
        Returns:
            list: 机器人访问所有目标点的最优路径
            float: 以米（m）为单位的总路径长度
        """
        robot_xyz = self.env._sim.get_agent_state().position
        robot_xyz = self.find_nearest_valid_point((robot_xyz[0], robot_xyz[1], robot_xyz[2]))
        robot_pos = self.world_to_map(robot_xyz[0], robot_xyz[1], robot_xyz[2])
        
        print("robo_pose",robot_pos,"real_robo_pose",robot_xyz)
        robot_height = robot_xyz[1]  # 获取机器人的 Y 轴高度

        # **1️⃣ 获取所有可行的 `goal`，并按类别分类**
        goals = self.env.current_episode.goals
        category_to_goals = {}

        for goal in goals:
            category = goal.object_category
            goal_xyz = self.find_nearest_valid_point((goal.position[0], goal.position[1], goal.position[2]))
            print(category,goal_xyz)
            goal_xy = self.world_to_map(goal_xyz[0], goal_xyz[1], goal_xyz[2])
            
            # 过滤掉高度相差过大的 `goal`
            if abs(goal.position[1] - robot_height) > height_threshold:
                continue

            if category == "bed":
                goal_xy = (65,25)

            if category not in category_to_goals:
                category_to_goals[category] = []

            category_to_goals[category].append((goal_xy, category))  # 存储坐标+类别

        print("category_to_goals",category_to_goals)
        if not category_to_goals:
            print("❌ 没有合适的目标点（所有目标都超出了高度限制）")
            return [], 0.0

        # **2️⃣ 生成所有可能的 `goal` 组合（每个类别选一个）**
        possible_combinations = list(product(*category_to_goals.values()))
        print("possible_combinations",possible_combinations)
        min_path_length = float("inf")
        best_goal_positions = None
        best_order = None

        # **3️⃣ 遍历所有 `goal` 组合，找到全局最优**
        for goal_set in possible_combinations:
            print("goal_set",goal_set)
            goal_positions = [g[0] for g in goal_set]  # 只取 (x, y)
            valid_goal_positions = goal_positions  # 目标点已确保可导航

            # **每种类别的 `goal` 只访问一次**
            all_positions = [robot_pos] + valid_goal_positions  # 机器人 + 目标点
            astar_dist_matrix = self.compute_astar_distances(all_positions)
            print("distance_matrix",astar_dist_matrix)
            optimal_order = self.solve_open_tsp_astar(astar_dist_matrix)

            # **计算该 `goal` 组合的总路径长度**
            total_path_length = sum(
                astar_dist_matrix[optimal_order[i], optimal_order[i + 1]] * self.meters_per_pixel
                for i in range(len(optimal_order) - 1)
            )

            # **如果该组合是最优的，就记录**
            if total_path_length < min_path_length:
                min_path_length = total_path_length
                best_goal_positions =  all_positions
                best_order = optimal_order

        # **4️⃣ 计算最终的最短路径**
        final_path = []
        for i in range(len(best_order) - 1):
            start_idx = best_order[i]
            goal_idx = best_order[i + 1]
            
            path_segment = self.astar_search(best_goal_positions[start_idx], best_goal_positions[goal_idx])
            if path_segment:
                final_path.extend(path_segment)

        print(f"✅ Optimal visit order (Per category best goal, No return): {best_order}")
        print(f" Total path length: {min_path_length:.2f} meters")

        # **5️⃣ 绘制路径并保存**
        self.visualize_path(final_path, save_path)

        return  min_path_length
