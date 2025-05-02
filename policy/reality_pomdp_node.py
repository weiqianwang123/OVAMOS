#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
import tf
import time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from vlfm.policy.pomdp_reality import POMDP_REALITY_Policy
from actionlib_msgs.msg import GoalID, GoalStatusArray
from nav_msgs.msg import OccupancyGrid
import tf
from std_msgs.msg import String
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker

class GoalPublisherNode:
    def __init__(self):
        rospy.init_node('goal_publisher_node', anonymous=True)

        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        # self.depth_sub = rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image, self.depth_callback)
        # self.rgb_sub = rospy.Subscriber('/zed/zed_node/rgb_raw/image_raw_color', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/rs_camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.rgb_sub = rospy.Subscriber('/rs_camera/color/image_raw', Image, self.rgb_callback)
        self.state_sub = rospy.Subscriber('/restart_signal',String, self.state_callback, queue_size=10)
        # Subscribe to /map topic
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        # Publisher
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.init_pub = rospy.Publisher('/init_signal', String, queue_size=10)
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

        self.tf_listener = tf.TransformListener()
        hardcoded_params = {
            "camera_height": 0.25,  # 摄像头高度
            "min_depth": 0.5,  # 深度传感器的最小深度
            "max_depth":  5.0,  # 深度传感器的最大深度
            "camera_fov": 58,  # 摄像头视场 (FOV) 角度
            "image_width": 640,  # 图像宽度（像素）
            "dataset_type": "reality",  # 数据集类型
        }
        self.step = 0
        self.goal_classes_string ="sink|umbrella"



        # Utility
        self.old_marker_ids = []
        self.bridge = CvBridge()
        self.current_odom = None
        self.latest_depth = None
        self.latest_rgb = None
        self.state = "init"
        self.policy = POMDP_REALITY_Policy(
            depth_image_shape = (720, 1280),
            object_map_erosion_size  = 5,
            text_prompt=self.goal_classes_string,
            asking_prompt="Seems like there is a target_object ahead.",
            camera_height=hardcoded_params["camera_height"],
            min_depth=hardcoded_params["min_depth"],
            max_depth=hardcoded_params["max_depth"],
            camera_fov=hardcoded_params["camera_fov"],
            image_width=hardcoded_params["image_width"],
            dataset_type=hardcoded_params["dataset_type"],
            visualize = True,
            compute_frontiers= True,
            min_obstacle_height = 1,
            max_obstacle_height =2,
            agent_radius = 0.18,
            obstacle_map_area_threshold  = 1.5 , # in square meters
            hole_area_thresh  = 100000,
            use_vqa = False,
            vqa_prompt  = "Is this ",
            coco_threshold = 0.8,
            non_coco_threshold = 0.4,
            use_max_confidence = False,
            sync_explored_areas = False,
        
        )
        # self.map = None
        # self.map_height = None
        # self.map_width = None
        print("init done ,search for",self.goal_classes_string)
    def delete_old_markers(self):
        for marker_id in self.old_marker_ids:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "goal_marker"
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
        # Clear the list after deleting
        self.old_marker_ids = []
    def publish_goal_marker(self, goal,color=(1.0, 0.0, 0.0),size=0.3,marker_id=0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_marker"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal[0]
        marker.pose.position.y = goal[1]
        marker.pose.position.z = 0.2  # Slightly above ground for visibility
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = size # Size of the sphere
        marker.scale.y = size
        marker.scale.z = size
        marker.color.r = color[0] 
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque
        self.marker_pub.publish(marker)

    def map_callback(self, msg: OccupancyGrid):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.origin_x =  msg.info.origin.position.x
        self.origin_y =  msg.info.origin.position.y
        resolution = msg.info.resolution

        # Convert to 2D array
        self.map = np.array(msg.data).reshape((self.map_height,self.map_width))
        # print("get new map",resolution)
        # print(self.map)

        # # Explored: non-negative
        # explored = data >= 0
      
        # # Obstacles: value >= 50
        # obstacles = data >= 50
     
        # # Save maps locally
        # timestamp = int(time.time())  # e.g., 1700000000
        # save_dir = "/home/yfx/vlfm/vlfm/reality_experiment"
        # os.makedirs(save_dir, exist_ok=True)


        # # Optional: Save as grayscale images
        # explored_img = (explored * 255).astype(np.uint8)
        # obstacle_img = (obstacles* 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(save_dir, f"explored_area.png"), explored_img)
        # cv2.imwrite(os.path.join(save_dir, f"obstacle_map.png"), obstacle_img)

    def state_callback(self,msg):
        # if len(msg.status_list)==0 or msg.status_list[-1].status =="3":
        #     time.sleep(0.5)
        #     print("ready to update")
        #     self.process_and_publish_goal()
        if msg.data ==  "Init":
            time.sleep(3)
            print("ready to move first time")
            self.process_and_publish_goal()
        elif msg.data == "Restart":
            print("ready to move")
            self.process_and_publish_goal()
        
    def odom_callback(self, msg):
        """Transforms odometry to the map frame and stores the transformed position."""
        try:
            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation

            # Convert pose to Transform
            odom_pose = PoseStamped()
            odom_pose.header.frame_id = "odom"
            odom_pose.header.stamp = rospy.Time(0)  # Use latest transform available
            odom_pose.pose.position = position
            odom_pose.pose.orientation = orientation

            # Transform the pose to the map frame
            transformed_pose = self.tf_listener.transformPose("map", odom_pose)

            # Extract the transformed position and yaw
            self.current_odom = transformed_pose
            self.x_robot = transformed_pose.pose.position.x
            self.y_robot = transformed_pose.pose.position.y
            self.theta = self.get_yaw_from_odometry(transformed_pose)

            # print(f"Transformed robot_state: x={self.x_robot}, y={self.y_robot}, theta={self.theta}")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF transform error: {e}")

    def depth_callback(self, msg):
        """Converts the depth image to OpenCV format."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            # print("shape", self.latest_depth.shape)
            nan_mask = np.isnan(self.latest_depth)
            self.latest_depth[nan_mask] = 0

            # Option B: Alternatively, np.nan_to_num can also be used:
            self.latest_depth = np.nan_to_num(self.latest_depth, nan=0.0)

            # Normalize the depth image to the 0–1 range
            # Using OpenCV's normalize:
            self.latest_depth = cv2.normalize(
                self.latest_depth,  # source
                None,               # destination (if None, returns new array)
                alpha=0,            # lower range value
                beta=1,             # upper range value
                norm_type=cv2.NORM_MINMAX
            )
            # print(self.latest_depth.shape)
            # print("depth callback")
        except Exception as e:
            rospy.logerr(f"Depth image conversion error: {e}")

    def rgb_callback(self, msg):
        """Converts the RGB image to OpenCV format."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # self.latest_rgb = cv2.resize(self.latest_rgb, (848, 480), interpolation=cv2.INTER_AREA)
            # print("rgb",self.latest_rgb.shape)
            # print("rgb callback")
        except Exception as e:
            rospy.logerr(f"RGB image conversion error: {e}")

       

    def process_and_publish_goal(self):
        """Uses odometry, depth, and RGB data to determine a goal and publish it."""
        if self.current_odom is None  or self.latest_rgb is None :
            rospy.logwarn("Waiting for all sensor data...")
            return

        self.step+=1
        # Example: Pick a point in front of the robot based on depth image
        d_height, d_width = self.latest_depth.shape
        center_x, center_y = d_width // 2, d_height // 2  # Pick center pixel

        depth_value = self.latest_depth[center_y, center_x]
        while np.isnan(depth_value) or depth_value <= 0:
            rospy.logwarn("Invalid depth value detected")
            time.sleep(1)
            center_y = center_y -5
            center_x = center_x -5
            depth_value = self.latest_depth[center_y, center_x]

        # Get the current position of the robot
        x_robot = self.current_odom.pose.position.x
        y_robot = self.current_odom.pose.position.y
        theta = self.get_yaw_from_odometry(self.current_odom)

        print("robot_state", x_robot,y_robot,theta)

        observations = {
            "rgb": self.latest_rgb,  # RGB image
            "depth": self.latest_depth,  # Depth image
            "xy": (x_robot,y_robot),  # (x, y) position of the robot
            "heading": theta,  # Heading (yaw angle in radians)
            "map":self.map,
            "map_height":self.map_height,
            "map_width":self.map_width,
            "origin_x":self.origin_x,
            "origin_y":self.origin_y
        }


        action,goal,goal_class,target_list  = self.policy.act(observations)
       
        if action == 2:
            print("INITING")
            init_state_msg = String()
            init_state_msg.data = "init"
            self.init_pub.publish(init_state_msg)
            return


        if action == 0:
            print(goal_class,"find!!!!!!!")
            class_list = self.goal_classes_string.split("|")
    
            # Remove the specified class if it exists
            if goal_class in class_list:
                class_list.remove(goal_class)
            
            # Join the remaining classes back into a string
            updated_goal_classes = "|".join(class_list)
            self.policy.reset_targets(updated_goal_classes)
            time.sleep(3)

        # Publish the goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal[0]
        goal_msg.pose.position.y = goal[1]
        goal_msg.pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, goal[2])
        goal_msg.pose.orientation.x = quaternion[0]
        goal_msg.pose.orientation.y = quaternion[1]
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]
        # goal_msg.pose.orientation.w = 1.0  # Default orientation

        self.goal_pub.publish(goal_msg)
        # Before publishing new markers
        self.delete_old_markers()

        # Now publish new ones
        for idx, target in enumerate(target_list):
            marker_id = idx + 1
            self.publish_goal_marker(target, color=(0.0, 0.0, 1.0), marker_id=marker_id)
            self.old_marker_ids.append(marker_id)

        # Publish the final goal
        self.publish_goal_marker(goal, color=(1.0, 0.0, 0.0), size = 0.4, marker_id=0)
        self.old_marker_ids.append(0)

        rospy.loginfo(f"Published goal at: ({goal[0]}, {goal[1]})")
        obstacle_map = self.policy._policy_info.get("obstacle_map", np.zeros((100, 100)))  # Default to blank
        value_map = self.policy._policy_info.get("value_map", np.zeros((100, 100)))  # Default to blank
        plt.figure(figsize=(5, 5))
        plt.imshow(obstacle_map, cmap="gray", origin="upper")
        plt.colorbar(label="Obstacle Intensity")
        plt.title("Obstacle Map")
        plt.savefig(f"/home/yfx/vlfm/vlfm/reality_experiment/obstacle_map_{self.step}.png")
        

        plt.figure(figsize=(5, 5))
        plt.imshow(value_map, cmap="viridis", origin="upper")
        plt.colorbar(label="Value Intensity")
        plt.title("Value Map")
        plt.savefig(f"/home/yfx/vlfm/vlfm/reality_experiment/value_map_{self.step}.png")

    def get_yaw_from_odometry(self, odom_msg):
        """Extracts yaw (rotation around Z) from odometry quaternion."""
        orientation_q = odom_msg.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]  # Yaw angle

if __name__ == '__main__':
    try:
        node = GoalPublisherNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
