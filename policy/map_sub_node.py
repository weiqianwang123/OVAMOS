#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
import rospy
from vlfm.mapping.obstacle_map import ObstacleMap  # adjust if in a different path
class ObstacleMapNode:
    def __init__(self):
        rospy.init_node('obstacle_map_listener')

        # Parameters (adjust if needed)
        self.min_height = rospy.get_param("~min_height", 0.0)
        self.max_height = rospy.get_param("~max_height", 2.0)
        self.agent_radius = rospy.get_param("~agent_radius", 0.25)

        self.obstacle_map = ObstacleMap(
                min_height=self.min_height,
                max_height=self.max_height,
                agent_radius=self.agent_radius
            )

        # Subscribe to /map topic
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # Convert to 2D array
        data = np.array(msg.data).reshape((height, width))
        print(data)
        # Explored: non-negative
        explored = data >= 0
      
        # Obstacles: value >= 50
        obstacles = data >= 50
     
        # Save maps locally
        timestamp = int(time.time())  # e.g., 1700000000
        save_dir = "/home/yfx/vlfm/vlfm/reality_experiment"
        os.makedirs(save_dir, exist_ok=True)


        # Optional: Save as grayscale images
        explored_img = (explored * 255).astype(np.uint8)
        obstacle_img = (obstacles* 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"explored_area.png"), explored_img)
        cv2.imwrite(os.path.join(save_dir, f"obstacle_map.png"), obstacle_img)

        rospy.loginfo(f"Saved maps to {save_dir} at {timestamp}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObstacleMapNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
