import pomdp_py
import math

###### Observation ######
class ObjectObservation(pomdp_py.Observation):
    """The xy pose of the object is observed; or NULL if not observed"""

    NULL = None
    def __init__(self,objid,pose):
        self.objid = objid
        self.pose = pose
       
    

    def _discrete_pose(self):
        """
        将连续位姿离散化为网格坐标（例如用四舍五入到整数）。
        这里认为每个网格单元的大小为1米。
        """
        # 如果pose是None，也直接返回None
        if self.pose is None:
            return self.NULL
        return (int(round(self.pose[0])), int(round(self.pose[1])))

    def __hash__(self):
        return hash((self.objid, self._discrete_pose()))
    
    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False
        if self.objid != other.objid:
            return False
        # 两个观测相等当且仅当它们的离散化位姿相同
        return self._discrete_pose() == other._discrete_pose()