import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class RRTPlanner:
    def __init__(self, navigable_map, step_size=20, max_nodes=100):
        """
        初始化 RRT 规划器。
        
        Args:
            navigable_map (np.array): 1000x1000 的可导航地图，1 为可通行，0 为障碍物
            step_size (int): RRT 扩展步长，默认 20 像素
            max_nodes (int): 生成的最大 RRT 节点数
        """
        self.map = navigable_map
        self.step_size = step_size
        self.max_nodes = max_nodes
        self.graph = nx.Graph()
        self.nodes = []
        self.min_x = 0
        self.min_y = 0
        self.max_x = 1000
        self.max_y = 1000

    def compute_sampling_bbox(self):
        """计算 navigable 且 explored 区域的最小包围盒"""
       

        coords = np.argwhere(self.map)

        if len(coords) == 0:
            raise ValueError("没有可采样的区域！")

        self.min_x, self.min_y = coords.min(axis=0)
        self.max_x, self.max_y = coords.max(axis=0)

        # print(f"采样 bbox: x ∈ [{self.min_x}, {self.max_x}], y ∈ [{self.min_y}, {self.max_y}]")

    def is_valid(self, x, y):
        """检查点是否在地图内且无障碍物"""
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and self.map[x, y] == 1:
            return True
        return False

    def sample_free_space(self):
        """只在 navigable 且 explored 区域的 bbox 里面随机采样"""
        for _ in range(100):  # 最多尝试 100 次
            x = random.randint(self.min_x, self.max_x)
            y = random.randint(self.min_y, self.max_y)
            if self.is_valid(x, y):
                return (x, y)
        raise RuntimeError("无法在采样边界内找到有效点")

    def nearest_neighbor(self, x, y):
        """找到 RRT 树中距离 (x, y) 最近的已有节点"""
        return min(self.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array([x, y])))

    def steer(self, from_node, to_node):
        """从 from_node 朝着 to_node 方向扩展 step_size 的步长"""
        vec = np.array(to_node) - np.array(from_node)
        dist = np.linalg.norm(vec)
        if dist < self.step_size:
            return to_node
        new_node = from_node + (vec / dist) * self.step_size
        return tuple(map(int, new_node))

    def collision_free(self, node1, node2):
        """检查 node1 -> node2 的连线是否无碰撞"""
        
        num_checks = int(np.linalg.norm(np.array(node1) - np.array(node2)) / 2)
        if num_checks == 0:
            return True  # node1 和 node2 相同，无需检查
        for i in range(num_checks + 1):
            point = np.array(node1) + (np.array(node2) - np.array(node1)) * (i / num_checks)
            if not self.is_valid(int(point[0]), int(point[1])):
                return False
        return True

    def build_rrt(self):
        """生成 RRT 树"""
        self.compute_sampling_bbox()
        root = self.sample_free_space()
        self.nodes.append(root)
        self.graph.add_node(root)

        for _ in range(self.max_nodes):
            rand_point = self.sample_free_space()
            nearest = self.nearest_neighbor(*rand_point)
            new_node = self.steer(nearest, rand_point)

            if self.is_valid(*new_node) and self.collision_free(nearest, new_node):
                self.nodes.append(new_node)
                self.graph.add_node(new_node)
                self.graph.add_edge(nearest, new_node, weight=np.linalg.norm(np.array(nearest) - np.array(new_node)))
        # self.visualize_rrt_tree()
        print(f"RRT 生成完成，节点数: {len(self.nodes)}")

    def find_path(self, start, goal, save_path="rrt_path.png"):
        """计算从 start 到 goal 的最短路径，并可视化 & 保存"""
        if not self.is_valid(*start):
            # print("起点不可通行")
            return None

        # 如果 goal 不可通行，找到离它最近的 RRT 采样点
        if not self.is_valid(*goal):
            nearest_goal = min(self.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(goal)))
            # print(f"目标点在障碍物上，使用最近的可行点: {nearest_goal}")
        else:
            nearest_goal = goal

        # 找到 RRT 树中离 start 最近的点
        nearest_start = self.nearest_neighbor(*start)

        # 在 RRT 图中寻找路径
        try:
            path = nx.shortest_path(self.graph, source=nearest_start, target=nearest_goal, weight="weight")
            # print(f"找到路径，长度: {len(path)}")
        except nx.NetworkXNoPath:
            # print("无法找到路径")
            return None
        
        # 可视化并保存路径
        # self.visualize_path(start, goal, path, save_path)
        return path

    def visualize_path(self, start, goal, path, save_path):
        """可视化 RRT 树、路径，并保存"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # 翻转地图（上下翻转）
        flipped_map = np.flipud(self.map)

        ax.imshow(flipped_map, cmap="gray", origin="upper")

        # 画 RRT 采样节点（蓝色）
        for node in self.nodes:
            ax.plot(node[1], self.map.shape[0] - node[0], 'bo', markersize=2)  # 调整y坐标翻转

        # 画 RRT 边（蓝线）
        for edge in self.graph.edges:
            node1, node2 = edge
            ax.plot([node1[1], node2[1]], 
                    [self.map.shape[0] - node1[0], self.map.shape[0] - node2[0]], 
                    'b-', linewidth=0.5)

        # 画路径（黄色）
        if path:
            path_x = [p[1] for p in path]
            path_y = [self.map.shape[0] - p[0] for p in path]  # y 轴翻转
            ax.plot(path_x, path_y, 'y-', linewidth=2)

        # 画起点（红色 ）和目标点（绿色 ）
        ax.plot(start[1], self.map.shape[0] - start[0], 'ro', markersize=6, label="Robot") 
        ax.plot(goal[1], self.map.shape[0] - goal[0], 'go', markersize=6, label="Goal") 

        ax.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"路径已保存到 {save_path}")



    def visualize_rrt_tree(self, save_path="/home/qianwei/vlfm/vlfm/reality_experiment/others/rrt_tree.png"):
        """可视化当前 RRT 树并保存到本地"""
        fig, ax = plt.subplots(figsize=(8, 8))

        flipped_map = np.flipud(self.map)
        ax.imshow(flipped_map, cmap="gray", origin="upper")

        # 画所有节点 (蓝色小点)
        for node in self.nodes:
            ax.plot(node[1], self.map.shape[0] - node[0], 'bo', markersize=2)

        # 画所有边 (蓝线)
        for edge in self.graph.edges:
            node1, node2 = edge
            ax.plot(
                [node1[1], node2[1]], 
                [self.map.shape[0] - node1[0], self.map.shape[0] - node2[0]], 
                'b-', linewidth=0.5
            )

        ax.set_title("RRT Tree")
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"RRT 树已保存到 {save_path}")

    def find_nearest_navigable_node(self, goal_pixel):
        """
        给定一个目标像素，返回最近的 RRT 采样节点。
        
        Args:
            goal_pixel (tuple): (x, y) 目标位置
        Returns:
            nearest_node (tuple): 最近的已采样节点
        """
        if not self.nodes:
            print("当前没有采样节点，请先运行 build_rrt()")
            return None
        
        nearest_node = min(self.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(goal_pixel)))
        return nearest_node