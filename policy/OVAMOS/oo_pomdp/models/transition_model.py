import pomdp_py
import copy
from vlfm.policy.OVAMOS.oo_pomdp.domain.state import *
from vlfm.policy.OVAMOS.oo_pomdp.domain.action import *
from vlfm.policy.OVAMOS.oo_pomdp.domain.observation import *
from vlfm.policy.OVAMOS.oo_pomdp.utils.map import Map


class OOTransitionModel(pomdp_py.TransitionModel):

    """
    :math:`T(s' | s, a) = \prod_i T(s_i' | s, a)`

    __init__(self, transition_models):
    Args:
        transition_models (dict) objid -> transition_model
    """

    def __init__(self,map,dim,sensor,object_id, epsilon=1e-9):
        """
        sensors (dict): robot_id -> Sensor
        for_env (bool): True if this is a robot transition model used by the
                Environment.  see RobotTransitionModel for details.
        """
        self._sensor = sensor
        self._objid = object_id
        self._transition_models = {}
        self._transition_models[object_id]=StaticObjectTransitionModel(object_id, epsilon=epsilon)
        self._transition_models[sensor.robot_id] = RobotTransitionModel(
                object_id,map,sensor, dim, epsilon=epsilon
            )
        super().__init__()

    def probability(self, next_state, state, action, **kwargs):
        """probability(self, next_state, state, action, **kwargs)
        Returns :math:`T(s' | s, a)
        """
        trans_prob = 1.0
        object_state = state.object_states[self._objid]
        robo_state = state.object_states[self._sensor.robot_id]
        next_object_state = next_state.object_states[self._objid]
        next_robot_state = next_state.object_states[self._sensor.robot_id]
        trans_prob = trans_prob * self._transition_models[self._objid].probability(next_object_state,object_state, action, **kwargs)
        trans_prob = trans_prob * self._transition_models[self._sensor.robot_id].probability(next_robot_state, robo_state, action, **kwargs)

        return trans_prob

    def sample(self, state, action, argmax=False, **kwargs):
        """
        sample(self, state, action, argmax=False, **kwargs)
        Returns random next_state"""
        object_states = {}
        for objid in state.object_states:
            if objid not in self._transition_models:
                # no transition model provided for this object. Then no transition happens.
                object_states[objid] = copy.deepcopy(state.object_states[objid])
                continue
            if argmax:
                next_object_state = self._transition_models[objid].argmax(state, action, **kwargs)
            else:
                next_object_state = self._transition_models[objid].sample(state, action, **kwargs)
            object_states[objid] = next_object_state
        return MosOOState(object_states)

    def argmax(self, state, action, **kwargs):
        """
        argmax(self, state, action, **kwargs)
        Returns the most likely next state"""
        return self.sample(state, action, argmax=True, **kwargs)
    
    def update_map(self,new_map):
        self.transition_models[self._sensor.robot_id].update_map(new_map)


    def __getitem__(self, objid):
        """__getitem__(self, objid)
        Returns transition model for given object"""
        return self._transition_models[objid]

    @property
    def transition_models(self):
        """transition_models(self)"""
        return self._transition_models








class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""

    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state["id"]]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])


class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self,object_id,map,sensor, dim, epsilon=0):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self._objid = object_id
        self._map= map
        self._sensor = sensor
        self._robot_id = sensor.robot_id
        self._dim = dim
        self._epsilon = epsilon

   
    def if_move_by(self, robot_id, state, action,check_collision=True):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""

        # robot_pose = state.pose(robot_id)
        # rx, ry, rth = robot_pose
      
        # # odometry motion model
        # forward, angle = action.motion()
        # rth += angle  # angle (radian)
        # rx = rx + forward * math.cos(rth)
        # ry = ry + forward * math.sin(rth)
        # rth = rth % (2 * math.pi)

        # if self._map.valid_pose(
        #     (rx, ry, rth),
        #     check_collision=check_collision,
        # ):

        return action.motion()
        # else:
        #     return robot_pose  # no change because change results in invalid pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if isinstance(state, RobotState):
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]

        next_robot_state = copy.deepcopy(robot_state)
       
        next_robot_state["pose"] = self.if_move_by(
                self._robot_id, state, action
            )
        # if isinstance(action,FindAction):
        #     robot_pose = state.pose(self._robot_id)
        #     z = self._sensor.observe(robot_pose,self._objid,state)
        #     # Update "objects_found" set for target objects
        #     if z.pose != ObjectObservation.NULL:
        #         next_robot_state["objects_found"] += 1
                
            
      
        return next_robot_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
    
    def update_map(self,new_map):
        self._map = new_map


