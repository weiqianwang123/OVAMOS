import pomdp_py
from .belief import *
from vlfm.policy.OVAMOS.oo_pomdp.models.transition_model import *
from vlfm.policy.OVAMOS.oo_pomdp.models.observation_model import *
from vlfm.policy.OVAMOS.oo_pomdp.models.reward_model import *
from vlfm.policy.OVAMOS.oo_pomdp.models.policy_model import *


class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""

    def __init__(
        self,
        robot_id,
        init_robot_state,  # initial robot state (assuming robot state is observable perfectly)
        object_ids,  # target object ids
        dim,  # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
        sensor,  # Sensor equipped on the robot
        sigma=0.01,  # parameter for observation model
        epsilon=1,  # parameter for observation model
        belief_rep="histogram",  # belief representation, either "histogram" or "particles".
        prior={},  # prior belief, as defined in belief.py:initialize_belief
        num_particles=100,  # used if the belief representation is particles
        map = None,
    ): 
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor
        self._map = map

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]
        robot_orientations = {self.robot_id: rth}
    

        # initialize belief
        init_belief = initialize_belief(
            dim,
            self.robot_id,
            self._object_ids,
            prior=prior,
            representation=belief_rep,
            robot_orientations=robot_orientations ,
            num_particles=num_particles,
        )
        transition_model = OOTransitionModel(
            self._map,dim, self.sensor, self._object_ids[0]
        )
        observation_model = ObjectObservationModel(
            self._map,self._object_ids[0], self.sensor, dim, sigma=sigma,
        )
        reward_model = GoalRewardModel(self._map, robot_id=self.robot_id)
        policy_model = PolicyModel(self._map,self.robot_id)
        super().__init__(
            init_belief=init_belief,  # 确保所有参数匹配
            policy_model=policy_model,
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=reward_model,
        )


    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
    
    
    def update_map(self,new_map):
        self.policy_model.update_map(new_map)
        self.transition_model.update_map(new_map)
        self.reward_model.update_map(new_map)
        self.observation_model.update_map(new_map)
