import pomdp_py
from vlfm.policy.OVAMOS.oo_pomdp.agent.agent import *
from vlfm.policy.OVAMOS.oo_pomdp.utils.sensor import *
import argparse
import time
import random
import csv
import os
# Helper dictionaries for conversions
ACTION_TO_INT = {
    FindAction: 0,
    MoveForwardAction:1,
    TurnLeftAction: 2,
    TurnRightAction: 3
    
}

# For actions that take parameters, you might store them in a dict.
# In this simple example, each action is parameter-free.
INT_TO_ACTION = {
    0: FindAction(),
    1: MoveForwardAction(),
    2: TurnLeftAction(),
    3: TurnRightAction()
   
}


class MosOOPOMDP():
    """
    A MosOOPOMDP is instantiated given a string description
    of the search world, sensor descriptions for robots,
    and the necessary parameters for the agent's models.

    Note: This is of course a simulation, where you can
    generate a world and know where the target objects are
    and then construct the Environment object. But in the
    real robot scenario, you don't know where the objects
    are. In that case, as I have done it in the past, you
    could construct an Environment object and give None to
    the object poses.
    """

    def __init__(
        self,
        dim,
        robot_id,
        object_ids,
        initial_robo_pose,
        sigma=0.01,
        epsilon=1,
        belief_rep="histogram",
        num_particles=100,
        

        #maps
        initial_value_map = None,
        initial_obstacle_map = None,
        initial_frontiers = None,


        #sensor param
        fov=90,
        min_range=1,
        max_range=5,
        angle_increment=5,

        #planner param
        max_depth=10,  # planning horizon
        discount_factor=0.99,
        planning_time=1.0,  # amount of time (s) to plan each step
        exploration_const=1000,  # exploration constant
        max_time=120,  # maximum amount of time allowed to solve the problem
        max_steps=500,

        
       
    ):
        """
        Args:
            robot_id (int or str): the id of the agent that will solve this MosOOPOMDP.
                If it is a `str`, it will be interpreted as an integer using `interpret_robot_id`
                in env/env.py.
            env (MosEnvironment): the environment.
            grid_map (str): Search space description. See env/env.py:interpret. An example:
                rx...
                .x.xT
                .....
                Ignored if env is not None
            sensors (dict): map from robot character to sensor string.
                For example: {'r': 'laser fov=90 min_range=1 max_range=5
                                    angle_increment=5'}
                Ignored if env is not None
            agent_has_map (bool): If True, we assume the agent is given the occupancy
                                  grid map of the world. Then, the agent can use this
                                  map to avoid planning invalid actions (bumping into things).
                                  But this map does not help the agent's prior belief directly.

            sigma, epsilon: observation model paramters
            belief_rep (str): belief representation. Either histogram or particles.
            prior (dict or str): either a dictionary as defined in agent/belief.py
                or a string, either "uniform" or "informed". For "uniform", a uniform
                prior will be given. For "informed", a perfect prior will be given.
            num_particles (int): setting for the particle belief representation
        """

        #uniform prior
        prior = {}
        self._object_ids = object_ids
        self._objid =  self._object_ids[0]
        
        self.sensor = Laser2DSensor(robot_id,fov=fov,min_range=min_range,max_range=max_range,angle_increment=angle_increment)
        self.map = Map(self.sensor,1000,1000)
        # self.map.vis_multi_value_map(initial_value_map)
        self.map.update(
            new_obstacle_map = initial_obstacle_map,
            new_value_map = self._value_cn(initial_value_map),
            new_frontiers = initial_frontiers
        )
        prior_obj = self.map.get_prior_with_value_map()
        prior[self._objid] = prior_obj
        self.robot_id = robot_id
        initial_robo_state = RobotState(robot_id,initial_robo_pose,objects_found=0)
        self.agent = MosAgent(
            robot_id,
            initial_robo_state,
            object_ids,
            dim,
            self.sensor,
            sigma=sigma,
            epsilon=epsilon,
            belief_rep=belief_rep,
            prior=prior,
            num_particles=num_particles,
            map=self.map,
        )

        self.navigate_mode = False
        self.navigate_goal = None
        self.planner = None
        

        # self.planner = pomdp_py.POMCP(
        #         max_depth=15,
        #         discount_factor=1,
        #         num_sims =10,
        #         planning_time=planning_time,
        #         exploration_const=exploration_const,
        #         rollout_policy= self.agent.policy_model,
        #     )  
        self.planner = pomdp_py.POUCT(
            max_depth=5,
            discount_factor=1,
            num_sims =20,
            planning_time=planning_time,
            exploration_const=exploration_const,
            rollout_policy=self.agent.policy_model,
        )  # Random by default
        print("pomdp inside init done")
        ### Belief Update ###
    def _belief_update(self,real_robot_pose):
    #     """Updates the agent's belief; The belief update may happen
    #     through planner update (e.g. when planner is POMCP)."""
    #     # Updates the planner; In case of POMCP, agent's belief is also updated.
    #     # self.planner.update(self.agent, real_action, real_observation)
        initial_robo_state = RobotState(self.robot_id,real_robot_pose,objects_found=0)
        prior = {}
        prior_obj = self.map.get_prior_with_value_map()
        prior[self._objid] = prior_obj
        self.agent = MosAgent(
            self.robot_id,
            initial_robo_state,
            self._object_ids,
            (1000,1000),
            self.sensor,
            sigma=0,
            epsilon=0,
            belief_rep="histogram",
            prior=prior,
            num_particles=5000,
            map=self.map,
        )


    def _value_cn(self, original_value_map: np.ndarray) -> np.ndarray:
        """
        计算 value map 在最后一个维度上的均值，将形状从 (1000, 1000, N) 转换为 (1000, 1000)。

        Args:
            original_value_map (np.ndarray): 形状为 (1000, 1000, N) 的原始 value map。

        Returns:
            np.ndarray: 形状为 (1000, 1000) 的 value map，其中每个点的值是原始 N 维值的均值。
        """
        value_map = np.mean(original_value_map, axis=-1)
        # 归一化到 [0, 1]
        min_val = np.min(value_map)
        max_val = np.max(value_map)

        if max_val > min_val:  # 避免除零错误
            value_map = (value_map - min_val) / (max_val - min_val)
        else:
            value_map = np.zeros_like(value_map)  # 如果所有值都一样，直接赋值 0
        
        return value_map




    
    def mos_act(self):
        if self.navigate_mode == True:
            return (self.map.navigate_goal[0],self.map.navigate_goal[1],0)
        # action = self.planner.plan(self.agent)
        # action_class = type(action)
        # if action_class not in ACTION_TO_INT:
        #     raise ValueError(f"Unsupported action type: {action_class}")
        # action_num = ACTION_TO_INT[action_class]
        best_frontier = self.get_best_frontier()
        
        # return action.motion()
        return (best_frontier[0],best_frontier[1],0)
    def get_best_frontier(self):
        if self.map.frontiers:
            return self.map.frontiers[0]
        else:
            return (0,0,0)
    def get_random_point(self):
        random_point = self.map.get_random_points()
        if random_point is not None:
            return random_point
        else:
            return (0,0)
    def update(self,real_action_num,real_observation_pose,real_robo_pose,new_value_map,new_obstacle_map,new_frontiers):
        if real_action_num not in INT_TO_ACTION:
            raise ValueError(f"Unsupported action code: {real_action_num}")
        # real_action = INT_TO_ACTION[real_action_num]
        # self.map.vis_multi_value_map(new_value_map)
        # 2) Update your map with new info
        self.map.update(new_obstacle_map, self._value_cn(new_value_map), new_frontiers)
        print("pomdp")
        # 3) Update the agent's internal map reference 
        self.agent.update_map(self.map)

        # 4) Clear old history if needed, then update with the latest action/observation
        self.agent.clear_history()

        find = False
        # real_observation = ObjectObservation(self._objid,ObjectObservation.NULL)
        if self.navigate_mode == False:
            if real_observation_pose is not None:
                print("object detected!!!!!!!!!!!!!!!!!!!!!!!")
                print(real_observation_pose)
                self.navigate_mode = True
                self.navigate_goal = real_observation_pose
                self.map.set_navigate_goal(real_observation_pose)
        else:
            find=self.map.check_navigate_success(real_robo_pose)
            if find:
                self.navigate_mode = False
                self.map.reset_navigate()
                
        
       

        print("robo pose",real_robo_pose)
        self._belief_update(real_robo_pose)
        print("belief update done")
        robot_px = self.map._xyz_to_px(np.array([[real_robo_pose[0], real_robo_pose[1],real_robo_pose[2]]]))[0]
        r_px, c_px,theta= robot_px
       

    
        # csv_filename = "/home/yfx/vlfm/output_frames_POMDP/path.csv"
        # # Check if file exists to write header only once
        # write_header = not os.path.exists(csv_filename)

        # # Save to CSV file in append mode ('a')
        # with open(csv_filename, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     if write_header:
        #         writer.writerow(["r_px", "c_px", "theta"])  # Write header only if file is new
        #     writer.writerow([r_px, c_px, theta])  # Append new row

        # print(f"Appended robot pose to {csv_filename}")

        self.map.save_maps(belief=self.agent.cur_belief,objid=self._objid,robot_id=self.robot_id,robot_pose=real_robo_pose,detected_pose=real_observation_pose,save_path="/home/yfx/vlfm/output_frames_POMDP/map.png")
        print("goal for map",self.map.navigate_goal)
       

        return find
        
        



        

