import pomdp_py
import random
from vlfm.policy.OVAMOS.oo_pomdp.domain.action import *
import numpy as np

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self,map,robot_id):
        self.robot_id = robot_id
        self._map = map
        self.used_action = set()

    def update_map(self,new_map):
        self._map = new_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):


        # 获取所有候选点
        candidate_points = self._map.get_candidate_points(state.pose(self.robot_id))

        # 将每个目标点封装为 MoveToAction
        actions = [MoveTo(target) for target in candidate_points]
        
        return actions


    def rollout(self, state, history=None):
    
        # 获取所有可能的 MoveTo actions（已经是排序好的）
        actions = self.get_all_actions(state=state, history=history)
        
        if not actions:
            return None  # 如果没有可选的 action，返回 None

        # 计算权重（按照索引进行指数衰减）
        decay_factor = 0.8  # 衰减因子，越小前面的点权重越高（0.5 ~ 0.9 推荐）
        weights = [decay_factor ** i for i in range(len(actions))]
        
        # 归一化，使权重和为 1
        weights = np.array(weights) / sum(weights)

        # 按权重随机抽取 action
        selected_action = random.choices(actions, weights=weights, k=1)[0]
        
        return selected_action