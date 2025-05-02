import pomdp_py
from vlfm.policy.OVAMOS.oo_pomdp.domain.action import *
from vlfm.policy.OVAMOS.oo_pomdp.utils.map import Map
import numpy as np
class MosRewardModel(pomdp_py.RewardModel):
    def __init__(self,map,big=1000, small=1, robot_id=None):
        """
        robot_id (int): This model is the reward for one agent (i.e. robot),
                        If None, then this model could be for the environment.
        target_objects (set): a set of objids for target objects.
        """
        self._map =  map
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self.mid = 1

    def probability(
        self, reward, state, action, next_state, normalized=False, **kwargs
    ):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state, robot_id=robot_id)

    def argmax(self, state, action, next_state, normalized=False, robot_id=None):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state, robot_id=robot_id)
    
    def update_map(self,new_map):
        self._map = new_map




class GoalRewardModel(MosRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """


    def _frontier_reward(self, frontiers, robot_pose,robot_pose_new):
        """
        基于 frontier 的奖励：
        - frontiers: frontier 列表，每个 frontier 用 (x, y) 表示
        - robot_pose: 机器人的位姿 (r_x, r_y, r_theta)
        
        返回一个 float 类型的奖励。设计思想：
          如果存在 frontier，则机器人距离最近 frontier 越近，奖励越高；
          如果不存在 frontier，则返回一个较大的负奖励。
        """
        
        reward = 0
        # 计算机器人到每个 frontier 的欧氏距离
        # distances_old = [self._map.get_distance_heading(robot_pose,(fx,fy)) for fx, fy in frontiers]
        # min_distance_old = min(distances_old)
        
        # 获取 (distance, heading) 的列表
        distances_headings = [self._map.get_distance_heading(robot_pose_new, (fx, fy)) for fx, fy in frontiers]

        # 找到最小距离及其对应的 heading
        min_distance_new, min_heading_new = min(distances_headings, key=lambda x: x[0])


        

        # closest_frontier = frontiers[np.argmin(distances_new)]  # 取最小距离对应的 frontier
        # obj_pose = closest_frontier  # 直接将最近的 frontier 作为 obj_pose

        
        # 设计奖励函数:
        # 这里采用一个简单的线性形式：当机器人离 frontier 越近时奖励越高，
        # 例如：reward = (max_reward - min_distance) 但保证不超过 max_reward 的上限。
        # 你可以根据具体需求对函数进行修改。
        
        # change = min_distance_old -min_distance_new
        # reward = change*40
        # if min_distance_new<0.2 and self._map.within_fov(robot_pose_new,obj_pose, fov=None):
        if min_distance_new<0.01 :
            reward = self.big
        
        # reward += -10 * (min_heading_new ** 2)  # 偏差越大，惩罚越大

        reward += -10* min_distance_new
       
        return reward
    def _object_goal_reward(self,obj_pose,robot_pose):
        reward = 0
        distance,heading = self._map.get_distance_heading(robot_pose,obj_pose)
        if distance<0.01 :
            reward = self.mid
        if distance is None:
            distance = 0
        reward += -10* distance
        return reward

    # def _navigate_reward(self,goal,robot_pose,robot_pose_new):
    #     reward = 0      
    #     # min_distance_old = self._map.get_rrt_distance(robot_pose,goal)
        
       
    #     # min_distance_new = self._map.get_rrt_distance(robot_pose_new,goal)

       
        
    #     # # 设计奖励函数:
    #     # # 这里采用一个简单的线性形式：当机器人离 frontier 越近时奖励越高，
    #     # # 例如：reward = (max_reward - min_distance) 但保证不超过 max_reward 的上限。
    #     # # 你可以根据具体需求对函数进行修改。
        
    #     # change = min_distance_old -min_distance_new
    #     # reward = change*40
    #     # if min_distance_new<0.1:
    #     #     reward = self.big
    #     # return reward
        

    #     distance_old,heading_old = self._map.get_distance_heading(robot_pose,goal)

    #     # 获取 (distance, heading) 的列表
    #     distance,heading = self._map.get_distance_heading(robot_pose_new,goal) 

    #     distance_change = distance_old - distance
    #     heading_change = heading_old** 2-heading** 2
        
        
    #     reward += 100* heading_change 
    #     # print("heading",headings)
    #     reward += 40*distance_change
        
    #     if distance<0.15 :
    #         reward = self.big/distance
    #     return reward


    def _reward_func(self, state, action, next_state, robot_id=None):
        if robot_id is None:
            assert (
                self._robot_id is not None
            ), "Reward must be computed with respect to one robot."
            robot_id = self._robot_id

        reward = 0
        robot_pose = next_state.object_states[robot_id].pose  # 假设格式为 (x, y, θ)，单位：米
        if  self._map.frontiers:
            reward +=self._frontier_reward(self._map.frontiers,state.object_states[robot_id].pose,next_state.object_states[robot_id].pose)
        for obj_id, obj_state in next_state.object_states.items():
            if obj_id == robot_id:
                continue  # 排除机器人自身
            obj_pose = obj_state.pose
            reward+= self._object_goal_reward(obj_pose,next_state.object_states[robot_id].pose)
        # if self._map.navigate_mode:
        #     if isinstance(action, MoveForwardAction):
        #         reward+=self._navigate_reward(self._map.navigate_goal,state.object_states[robot_id].pose,next_state.object_states[robot_id].pose)
        #     # print("Action",action,"POSE",next_state.object_states[robot_id].pose,"REWARD",reward)
        #     return reward
            
        # found = False
       
        # if found:
        #     reward += self.big

        # if isinstance(action, TurnLeftAction) or isinstance(action, TurnRightAction) or isinstance(action, MoveForwardAction):
        #     reward = reward - self.small
        # elif isinstance(action, FindAction):
        #         # transition function should've taken care of the detection.
        #         new_objects_count = next_state.object_states[robot_id].objects_found- state.object_states[robot_id].objects_found
        #         if new_objects_count == 0:
        #             # No new detection. "detect" is a bad action.
        #             reward -= self.big
        #         else:
        #             # Has new detection
        #             robot_pose = next_state.object_states[robot_id].pose  # 假设格式为 (x, y, θ)，单位：米
        #             found_close = False
        #             
        #             if found_close:
        #                 #legal find
        #                 reward += self.big
        #             else:
        #                 #illegal find
        #                 reward -= self.big


        # if not self._map.valid_pose(next_state.object_states[robot_id].pose):
        #     reward -= self.big
       
       

        # print("Action",action,"POSE",next_state.object_states[robot_id].pose,"REWARD",reward)

        
        return reward
