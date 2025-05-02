from collections import deque
import heapq
import numpy as np
import matplotlib.pyplot as plt
import cv2
class AStarPlanner:
    def __init__(self, grid_map, resolution=20):
        self.map = grid_map
        self.expand_obstacles()
        self.resolution = resolution
        self.rows, self.cols = grid_map.shape
    


 

    def expand_obstacles(self, expansion_radius=5):
        """
        对障碍物进行膨胀，增加安全边界
        :param expansion_radius: 膨胀的半径（单位：像素）
        """
        kernel = np.ones((2 * expansion_radius + 1, 2 * expansion_radius + 1), np.uint8)
        self.map = cv2.dilate(self.map.astype(np.uint8), kernel, iterations=1)


    def heuristic(self, node, goal):
        """启发式函数：曼哈顿距离"""
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def neighbors(self, node):
        """获取可行的相邻节点"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < self.rows and 0 <= y < self.cols and self.map[x, y] == 1:
                result.append((x, y))
        return result

    def find_nearest_valid_point(self, goal):
        """
        如果目标点在障碍物上，寻找最近的可通行点。
        使用 BFS 搜索最近的可行点，保证是最短距离。
        """
        x, y = goal  # 确保 goal 是 (x, y) 坐标
        if self.map[x, y] == 1:
            return (x, y)  # 目标点本身是可通行的，直接返回

        queue = deque([(x, y)])
        visited = set([(x, y)])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols and (nx, ny) not in visited:
                    if self.map[nx, ny] == 1:  # 找到最近的可行点
                        return (nx, ny)
                    queue.append((nx, ny))
                    visited.add((nx, ny))

        return goal  # 如果没有找到可通行点（不太可能）


    def find_path(self, start, goal):
        """
        运行 A* 计算最优路径。
        如果目标点在障碍物上，则使用 `find_nearest_valid_point()` 找最近的可行点。
        """
        goal = self.find_nearest_valid_point(goal)  # 确保目标点是可行的
        if not goal:
            print("无法找到最近的可行目标点！")
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                # self.visualize(start,goal,path)
                return path 
            for neighbor in self.neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        return None  # 没有找到路径

    def reconstruct_path(self, came_from, current):
        """重建最短路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    

    def visualize(self, start, goal, path):
        """可视化路径，地图和路径一起翻转"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # ⬆️ 上下翻转地图
        flipped_map = np.flipud(self.map)  
        ax.imshow(flipped_map, cmap="gray", origin="upper")  # 设定 origin="lower" 保持一致

        # ⬆️ 翻转路径点、起点、终点
        def flip_y(y):
            return self.map.shape[0] - y  # 调整 Y 轴翻转

        # 画 A* 路径（黄色 ）
        if path:
            path_x = [p[1] for p in path]  
            path_y = [flip_y(p[0]) for p in path]  
            ax.plot(path_x, path_y, 'y-', linewidth=2, label="Path")

        # 画起点（红色 ）和目标点（绿色 ）
        ax.plot(start[1], flip_y(start[0]), 'ro', markersize=6, label="Robot")  
        ax.plot(goal[1], flip_y(goal[0]), 'go', markersize=6, label="Goal")  

        ax.legend()
        plt.savefig("/home/yfx/vlfm/output_frames_POMDP/astar_path.png")
        plt.close()
        # print("路径已保存到 /home/yfx/vlfm/output_frames_POMDP/astar_path.png")


