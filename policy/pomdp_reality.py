import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor
import math

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections


from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from depth_camera_filtering import filter_depth


from vlfm.policy.OVAMOS.oo_pomdp.problems import MosOOPOMDP


PROMPT_SEPARATOR = "|"

class POMDP_REALITY_Policy():
    _target_objects: str = ""  # 用于存储多个目标对象
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True




    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False


    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.mean(i, axis=-1)

    def __init__(
        self,
        depth_image_shape: Tuple[int, int],
        object_map_erosion_size: float,

        text_prompt: str,
        asking_prompt: str,

        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
     
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        
        use_max_confidence: bool = False,
        sync_explored_areas: bool = False,

        pointnav_policy_path = "data/pointnav_weights.pth"

    ) -> None:
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        # self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

        
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._asking_prompt = asking_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        

        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

        self._target_objects = text_prompt
        self._non_coco_caption = text_prompt.replace(PROMPT_SEPARATOR, " . ")


        self._done_initialize_pomdp = False
        self._pomdp_planner = None
        self._prior_action = -1
        self.goal_class = None
    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
       
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def reset(self):
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        self.angle_cumulate = 0
        self.prior_angle = 0
        if self._compute_frontiers:
            self._obstacle_map.reset()

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        # self.angle_cumulate += abs(self._observations_cache["robot_heading"]-self.prior_angle )
        # self.prior_angle = self._observations_cache["robot_heading"]
        # if self.angle_cumulate>=math.pi*2:
        #     self._done_initializing = True
        # else:
        #     self._done_initializing = False
        self._done_initializing = not self._num_steps < 11  # type: ignore
       
        return (self._observations_cache["robot_xy"][0]+0,self._observations_cache["robot_xy"][1]+0,self._observations_cache["robot_heading"]+math.pi/6)
    
  
    def act(self,
            observations: Dict):
        
        
        if self._num_steps == 0:
            self.reset()
        self._cache_observations(observations)
        self._update_value_map()
        
        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        

        robot_xy = self._observations_cache["robot_xy"]
        goal,goal_class = self._get_target_object_location(robot_xy)
        print("goal:",goal,"goal_class",goal_class)
        find = False
        action = 1 #searching  0:find  2:init
        goal_pose = None
        # print(np.array([robot_xy[0], robot_xy[1], self._observations_cache["robot_heading"]]))
        if not self._done_initializing:  # Initialize
            mode = "initialize"
            goal_pose = self._initialize()
            action = 2
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            if not self._done_initialize_pomdp:
                self._initialize_pomdp()
            else:
                self._update_pomdp(goal)
            goal_pose = self._pomdp_act()
            action = 1
            
            # action_xy = action[:2]  # ✅ 取前两个元素 (x, y)

            # action = self._pointnav(np.array(action_xy),stop=False).item()
        else:
            mode = "navigate"
            if not self._done_initialize_pomdp:
                self._initialize_pomdp()
            find = self._update_pomdp(goal)
            self.goal_class = goal_class
            if find:
                print("Navigate done")
                action = 0
                goal_pose = (0,0,0)
            else:
                action = 1
                goal_pose = self._pomdp_act()
                
                # action_xy = action[:2]  # ✅ 取前两个元素 (x, y)
                # action = self._pointnav(np.array(action_xy),stop=False).item()
       
        
        

        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action}| Goal Pose: {goal_pose}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1


        frontiers = self._observations_cache["frontier_sensor"]
        self._observations_cache = {}
        self._did_reset = False
        self._prior_action = action
        return action,goal_pose,self.goal_class,frontiers
        

    def reset_targets(self,new_prompt):
        old_object_list = self._target_objects.split("|")
        new_object_list = new_prompt.split("|")
        remove_index = None
        for i, obj in enumerate(old_object_list):
            if obj not in new_object_list:
                remove_index=i
        if remove_index is None:
            assert("index not exist in old objects")
        self._value_map.remove_target_channel(remove_index)
        self._text_prompt = new_prompt
        self._target_objects = new_prompt
        self._non_coco_caption = new_prompt.replace(PROMPT_SEPARATOR, " . ")
        self._object_map.remove_target_class(old_object_list[remove_index])
    def _initialize_pomdp(self):
        if not self._observations_cache:
            raise ValueError("观测数据缓存为空，请先调用 _cache_observations()。")

     
        robot_xy = self._observations_cache["robot_xy"]
        initial_robo_pose = np.array([robot_xy[0], robot_xy[1], self._observations_cache["robot_heading"]])
 
        dim = (1000,1000)

    
        robot_id = 2

        
        object_ids = [1]

        # 提取初始地图信息
        initial_value_map = self._value_map._value_map.copy()
        initial_obstacle_map = self._obstacle_map if self._compute_frontiers else None
        frontiers = self._observations_cache["frontier_sensor"]
        initial_frontiers = [frontier[:2] for frontier in frontiers]
        self._pomdp_planner = MosOOPOMDP(
            dim,
            robot_id,
            object_ids,
            initial_robo_pose,
            sigma=1,
            epsilon=0.7,
            belief_rep="histogram",
            num_particles=1000,
            #maps
            initial_value_map = initial_value_map,
            initial_obstacle_map = initial_obstacle_map.explored_area,
            initial_frontiers = initial_frontiers,
            #sensor param
            fov=self._camera_fov,
            min_range=self._min_depth,
            max_range=self._max_depth,
            angle_increment=0.05,
            #planner param
            max_depth=50,  # planning horizon
            discount_factor=0.99,
            planning_time=1.0,  # amount of time (s) to plan each step
            exploration_const=math.sqrt(2),  # exploration constant
            max_time=120,  # maximum amount of time allowed to solve the problem
            max_steps=500)
        self._done_initialize_pomdp = True
    def _pomdp_act(self):
        action_num = self._pomdp_planner.mos_act()
        return action_num
    def _update_pomdp(self,goal):
        real_observation_pose = None
        if goal is not None :
            real_observation_pose = goal
        robot_xy = self._observations_cache["robot_xy"]
        real_robo_pose = np.array([robot_xy[0], robot_xy[1], self._observations_cache["robot_heading"]])
        new_value_map = self._value_map._value_map.copy()  
        new_obstacle_map = self._obstacle_map.explored_area if self._compute_frontiers else None
        
        frontiers = self._observations_cache["frontier_sensor"]
        new_frontiers = [frontier[:2] for frontier in frontiers]
        return self._pomdp_planner.update(self._prior_action,real_observation_pose,real_robo_pose,new_value_map,new_obstacle_map,new_frontiers)

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            # def bbox_to_mask(rgb, bbox):
            #     """
            #     Create a binary mask from bounding box coordinates.

            #     Args:
            #         rgb  : Image array of shape (H, W, 3).
            #         bbox : List or tuple containing [x_min, y_min, x_max, y_max] 
            #             bounding box coordinates.

            #     Returns:
            #         Binary mask (H, W) with 1s inside the bounding box and 0s outside.
            #     """
            #     # Convert bbox coordinates to int
            #     x_min, y_min, x_max, y_max = map(int, bbox)

            #     # Image dimensions
            #     height, width, _ = rgb.shape

            #     # Initialize a zeros mask
            #     mask = np.zeros((height, width), dtype=np.uint8)
                
            #     # Clip the coordinates so they do not exceed image bounds (optional but safer)
            #     x_min = max(0, x_min)
            #     y_min = max(0, y_min)
            #     x_max = min(width,  x_max)
            #     y_max = min(height, y_max)

            #     # Fill the bounding box region with ones
            #     mask[y_min:y_max, x_min:x_max] = 1
            #     return mask
            # object_mask = bbox_to_mask(rgb, bbox_denorm)
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.
            detected_phrase = detections.phrases[idx]
            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                detected_phrase,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_objects.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        # if has_coco and has_non_coco and detections.num_detections == 0:
        #     # Retry with non-coco object detector
        #     detections = self._object_detector.predict(img, caption=self._non_coco_caption)
        #     detections.filter_by_class(target_classes)
        #     detections.filter_by_conf(self._non_coco_threshold)

        return detections
        


    def _cache_observations(self, observations:Dict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        new_map =  observations["map"]
        print("new_map",new_map)
        height =  observations["map_height"]
        width =  observations["map_width"]
        origin_x = observations["origin_x"]
        origin_y = observations["origin_y"]
        rgb = observations["rgb"]
        depth = observations["depth"]
        x, y = observations["xy"]
        camera_yaw = observations["heading"]
        
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None,set_black_value=1.0)
        # print("Depth",depth)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)


        # Assume 'rgb' and 'depth' are obtained from your observations as in your snippet.
        # Optionally convert RGB from (R,G,B) to (B,G,R) since OpenCV expects BGR format.
        cv2.imwrite("rgb_image.png", rgb)

        # For the depth image, normalize the values to the range [0, 255] if necessary.
        # This normalization is useful if depth is a float or has a different range.
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        cv2.imwrite("depth_image.png", depth_norm)
        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            # self._obstacle_map.update_map(
            #     depth,
            #     tf_camera_to_episodic,
            #     self._min_depth,
            #     self._max_depth,
            #     self._fx,
            #     self._fy,
            #     self._camera_fov,
            # )
            self._obstacle_map.update_map_direct(new_map,height,width,tf_camera_to_episodic,origin_x,origin_y)
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
          
            frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"],
        }
    


    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_objects.split("|")[0]):
            target_point_cloud = self._object_map.get_target_cloud(self._target_objects.split("|")[0])
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_objects.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_objects.split("|")[0]),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info



    def _update_value_map(self) -> None:
        cosines = []
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        for rgb in all_rgb:
            cos_for_this_rgb = []
            for target_obj in self._target_objects.split("|"):
                # 假设 p 是个固定字符串，里面的 "target_object" 占位符要替换
                replaced_prompt = self._asking_prompt.replace("target_object", target_obj)
                cos_val = self._itm.cosine(rgb, replaced_prompt)
                cos_for_this_rgb.append(cos_val)

            # cos_for_this_rgb 形如 [cos_obj1, cos_obj2, ...]
            cosines.append(cos_for_this_rgb)
            print("cos",cos_for_this_rgb)

        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )


    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        target_objects_list = self._target_objects.split(PROMPT_SEPARATOR)
        return self._object_map.get_best_object_among_targets_reality(target_objects_list,position)


   