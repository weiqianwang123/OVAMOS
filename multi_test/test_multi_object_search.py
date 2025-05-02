

from habitat.core.registry import registry
from habitat.tasks.nav.object_nav_task import ObjectNavigationTask
from habitat.sims.habitat_simulator.actions import HabitatSimActions
@registry.register_task(name="MultiObjectNav-v0")
class MultiObjectNavTask(ObjectNavigationTask):
    def reset(self, episode):
        super().reset(episode)
        self.target_objects = episode.target_objects  # Define multiple target objects

    def step(self, action):
        observations = super().step(action)
        # Add custom code to handle multiple targets
        return observations


from habitat.tasks.nav.object_nav_task import ObjectGoal
from habitat.core.dataset import Episode
from typing import List
import attr
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectEpisode(ObjectGoalNavEpisode):
    """
    An episode that includes multiple object goals.
    """
    goals: List[ObjectGoal] = attr.ib(factory=list)

# Example of creating an episode with multiple objects to search for
multi_object_episode = {
    "episode_id": "episode_1",
    "scene_id": "/home/yfx/vlfm/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
    "start_position": [0, 0, 0],
    "start_rotation": [0, 0, 0, 1],
    "goals": [
        ObjectGoal(object_id="0", position=[1, 0, 1]),  # Example of object goals
        ObjectGoal(object_id="1", position=[2, 0, 3])
    ]
}

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

@registry.register_dataset(name="multi_obj_dataset")
class MOVONDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[MultiObjectEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = MultiObjectEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = MultiObjectEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]
        

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]
       # 按顺序两两合并
        # for i in range(0, len(self.episodes)-1, 2):
        #     ep1 = self.episodes[i]
        #     ep2 = self.episodes[i + 1]
        #     self.merge_episodes(ep1, ep2)
            
    def merge_episodes(self, episode1: MultiObjectEpisode, episode2: MultiObjectEpisode) -> None:
        """
        融合两个 episode 的目标，互相追加，避免重复。
        :param episode1: 第一个 episode 实例
        :param episode2: 第二个 episode 实例
        """
        if not isinstance(episode1.goals, list):
            episode1.goals = list(episode1.goals)
        if not isinstance(episode2.goals, list):
            episode2.goals = list(episode2.goals)

        # 去重追加目标
        ep1_goal_ids = {goal.object_id for goal in episode1.goals}
        ep2_goal_ids = {goal.object_id for goal in episode2.goals}

        # 追加新的目标
        episode1.goals.extend(goal for goal in episode2.goals if goal.object_id not in ep1_goal_ids)
        episode2.goals.extend(goal for goal in episode1.goals if goal.object_id not in ep2_goal_ids)

import habitat
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
import random
import numpy as np
import torchvision
import cv2
import torch
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
import vlfm.policy.reality_policies
from  vlfm.policy.habitat_policies import HabitatITMPolicyV2
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Union, cast

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat_sim.utils import viz_utils as vut
from collections import defaultdict

def create_grid_display(frame, obstacle_map, value_map, goal_state):
    # 如果地图是 Tensor，转换为 NumPy 数组
    if isinstance(obstacle_map, torch.Tensor):
        obstacle_map = obstacle_map.cpu().numpy()
    if isinstance(value_map, torch.Tensor):
        value_map = value_map.cpu().numpy()

    # 调整 obstacle_map 和 value_map 的大小与 frame 一致
    height, width, _ = frame.shape
    obstacle_map_resized = cv2.resize(obstacle_map, (width, height))
    value_map_resized = cv2.resize(value_map, (width, height))

    # 创建一个空白图像作为占位符
    empty_placeholder = np.zeros_like(frame)

    # 在空网格中添加 goal_state
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_color = (0, 255, 0)  # 显眼的绿色
    text_size = cv2.getTextSize(str(goal_state), font, font_scale, font_thickness)[0]
    text_x = (empty_placeholder.shape[1] - text_size[0]) // 2
    text_y = (empty_placeholder.shape[0] + text_size[1]) // 2

    # 在空网格中心绘制 goal_state
    cv2.putText(empty_placeholder, str(goal_state), (text_x, text_y), font, font_scale, text_color, font_thickness)

    # 拼接成 2x2 网格
    top_row = np.hstack((frame, obstacle_map_resized))
    bottom_row = np.hstack((value_map_resized, empty_placeholder))
    grid_display = np.vstack((top_row, bottom_row))

    return grid_display


def success(distance_to_closet_target):
    if distance_to_closet_target<2:
        return True
    else:
        return False



hardcoded_params = {
    "camera_height": 0.88,  # 摄像头高度
    "min_depth": 0.5,  # 深度传感器的最小深度
    "max_depth": 5.0,  # 深度传感器的最大深度
    "camera_fov": 79,  # 摄像头视场 (FOV) 角度
    "image_width": 640,  # 图像宽度（像素）
    "dataset_type": "hm3d",  # 数据集类型
}

# 初始化策略
NAME_TO_ID = {
    "chair": 0,
    "bed": 1,
    "plant": 2,
    "toilet": 3,
    "tv_monitor": 4,
    "sofa": 5,
}
policy = HabitatITMPolicyV2(
    pointnav_policy_path = "data/pointnav_weights.pth",
    depth_image_shape = (224, 224),
    pointnav_stop_radius = 0.9,
    use_max_confidence = False,
    object_map_erosion_size  = 5,
    exploration_thresh =0.0,
    obstacle_map_area_threshold  = 1.5 , # in square meters
    min_obstacle_height = 0.61,
    max_obstacle_height =0.88,
    hole_area_thresh  = 100000,
    use_vqa = False,
    vqa_prompt  = "Is this ",
    coco_threshold = 0.8,
    non_coco_threshold = 0.4,
    agent_radius = 0.18,
    text_prompt="Seems like there is a target_object ahead.",
    camera_height=hardcoded_params["camera_height"],
    min_depth=hardcoded_params["min_depth"],
    max_depth=hardcoded_params["max_depth"],
    camera_fov=hardcoded_params["camera_fov"],
    image_width=hardcoded_params["image_width"],
    dataset_type=hardcoded_params["dataset_type"],)

print(torch.cuda.is_available()) 
# List of possible actions
possible_actions = [HabitatSimActions.move_forward,HabitatSimActions.turn_left,HabitatSimActions.turn_right]

# Load configuration
config = get_config("/home/yfx/vlfm/vlfm/multi_test/multi.yaml")

with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_goal_positions=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
            }
        )

# Initialize the environment
env = habitat.Env(config=config)


# 定义图片保存路径
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

frame_counter = 0  # 帧计数器


observations = env.reset()
for episode in range(len(env.episodes)):  # Iterate through episodes

    observations = env.reset()  # Reset environment
    observations = env.reset()  # Reset environment
    masks  = torch.zeros(
        1,
        1,
        device="cpu",
        dtype=torch.bool,
    )

    print(f"Episode {env.current_episode.episode_id} goals: {[goal.object_category for goal in env.current_episode.goals]}")
    # 提取当前的 goals
    goals = env.current_episode.goals

    # 创建一个字典用于按类别存储 goals
    category_to_goals = defaultdict(list)

    # 遍历所有的 goals 并按 category 分组
    for goal in goals:
        category_to_goals[goal.object_category].append(goal)

    # 将分组后的 goals 按类别顺序整理成列表
    separated_goals = list(category_to_goals.values()) 


    #1,2,3...
    goal_state = 1
    #after find 2 objects,goal state reach the final state
    final_state = 3  
    done = False
    prev_action = None
    rnn_hidden_states = None

    print(f"Starting episode {episode + 1}/{len(env.episodes)}")
    print(f"Episode ID: {env.current_episode.episode_id}")

   
    # Add fame to vis_frames
    vis_frames = []
    episode_over = False
    
    while not episode_over :  # Continue until the episode ends
        info = env.get_metrics()
        filtered_observations = {
            "rgb": observations["rgb"],
            "depth": observations["depth"]
        }
        frame = observations_to_image(filtered_observations, info)
        info.pop("top_down_map")
        frame = overlay_frame(frame, info)
        
        
        for key, value in observations.items():
            # 如果值是 NumPy 数组或类似结构，转换为 Tensor
            if isinstance(value, np.ndarray):
                # 转换为 Tensor 并增加一个维度
                observations[key] = torch.tensor(value)
            # 如果值已经是 Tensor，直接增加一个维度
            elif isinstance(value, torch.Tensor):
                observations[key] = value
            else:
                # 对于无法直接转换的值，可以添加特定逻辑或跳过
                raise TypeError(f"Unsupported type for key {key}: {type(value)}")
            if key=="depth":
                observations[key] = torch.tensor(value).unsqueeze(0)

        
        goal_category = separated_goals[0][0].object_category
        category_id = NAME_TO_ID.get(goal_category, None)
        observations["objectgoal"][0] = category_id
        policy_output: PolicyActionData = policy.act(
            observations=TensorDict(observations),  # 包装成 TensorDict
            rnn_hidden_states = rnn_hidden_states,
            prev_actions=prev_action,
            masks=masks  # Mask for valid step
        )

        masks = torch.tensor(
            [[1],[1]],
            dtype=torch.bool,
            device="cpu",
        )
      
        


       
        # print(policy._policy_info)
        action = policy_output.actions.item()  # 提取动作
        if action==0:
            #find an object or cannot find object
            current_position = env._sim.get_agent_state().position
            # 初始化变量以存储最小距离和对应的类别
            min_distance = float("inf")
            closest_category = None
            closest_goals = None

            # 遍历所有分组，计算到每组目标的最小距离
            for goal_group in separated_goals:
                # 提取该类别目标的位置
                positions = [goal.position for goal in goal_group]

                # 计算到该类别目标的最小距离
                distance_to_target = env._sim.geodesic_distance(
                    current_position, positions
                )

                # 如果发现更短的距离，更新记录
                if distance_to_target < min_distance:
                    min_distance = distance_to_target
                    closest_category = goal_group[0].object_category
                    closest_goals = goal_group
            print("min_distances")
            print(min_distance)
            if success(min_distance):
                separated_goals.remove(closest_goals)
                goal_state+=1
                if goal_state==final_state:
                    episode_over = True
                else:
                    observations["objectgoal"][0] = category_id
                    # policy.goal_change_reset(TensorDict(observations))
                    masks  = torch.zeros(
                                1,
                                1,
                                device="cpu",
                                dtype=torch.bool,
                            )
                    action = 2

            else:
                goal_state = -1
                episode_over = True

                


        rnn_hidden_states = policy_output.rnn_hidden_states  
        prev_action = action  # 更新上一个动作

        # 采取动作，更新环境
        observations = env.step(action)
       
        
        # Add obstacle_map and value_map to the frame
        obstacle_map = policy._policy_info.get("obstacle_map", np.zeros((100, 100)))  # Default to blank
        value_map = policy._policy_info.get("value_map", np.zeros((100, 100)))  # Default to blank

        # Create grid display
        grid_frame = create_grid_display(frame, obstacle_map, value_map, goal_state)

        vis_frames.append(grid_frame)
    current_episode = env.current_episode
    video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
    output_path = "/home/yfx/vlfm/output_frames"
    # Create video from images and save to disk
    images_to_video(
        vis_frames, output_path, video_name, fps=6, quality=9
    )
    vis_frames.clear()
    # Display video
    # vut.display_video(f"{output_path}/{video_name}.mp4")

    