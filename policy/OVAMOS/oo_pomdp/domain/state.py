import pomdp_py
import math


###### States ######
class ObjectState(pomdp_py.ObjectState):
    def __init__(self,objid,pose):
        objclass = "target" #no meaning ,all objects are treated as one target
        super().__init__(objclass, {"pose": pose, "id": objid})

    def __str__(self):
        return "ObjectState(%s,%s)" % (str(self.objclass), str(self.pose))

    @property
    def pose(self):
        return self.attributes["pose"]
    @property
    def objid(self):
        return self.attributes["id"]



class RobotState(pomdp_py.ObjectState):
    def __init__(self,robot_id,pose,objects_found):
        super().__init__(
            "robot",
            {
                "id": robot_id,
                "pose": tuple(pose),  # x,y,th
                "objects_found": objects_found,
            },
        )

    def __str__(self):
        return "RobotState(%s)" % (
            str(self.pose),
        )

    def __repr__(self):
        return str(self)

    @property
    def pose(self):
        return self.attributes["pose"]
    
    
    @property
    def objects_found(self):
        return self.attributes["objects_found"]



class MosOOState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)

    def object_pose(self, objid):
        return self.object_states[objid]["pose"]

    def pose(self, objid):
        return self.object_pose(objid)

    @property
    def object_poses(self):
        return {
            objid: self.object_states[objid]["pose"] for objid in self.object_states
        }

    def __str__(self):
        return "MosOOState%s" % (str(self.object_states))

    def __repr__(self):
        return str(self)

