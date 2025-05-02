import pomdp_py
import math


###### Actions ######
class Action(pomdp_py.Action):
    """Mos action; Simple named action."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name

class FindAction(Action):
    def __init__(self):
        super().__init__("find")
    def motion(self):
        return (0,0)


Find = FindAction()

class TurnLeftAction(Action):
    def __init__(self):
        super().__init__("TurnLeft")
    def motion(self):
        return (0,math.pi/6)

TurnLeft = TurnLeftAction()

class TurnRightAction(Action):
    def __init__(self):
        super().__init__("TurnRight")
    def motion(self):
        return (0,-math.pi/6)
  
TurnRight = TurnRightAction()

class MoveForwardAction(Action):
    def __init__(self):
        super().__init__("MoveForward")
    def motion(self):
        return (0.25,0)


MoveForward = MoveForwardAction()


class MoveTo(Action):
    """High-level action: 指定目标坐标"""

    def __init__(self, target_node):
        self.target_node = target_node  # (x, y)
        super().__init__("MoveTo")

    def motion(self):
        return (self.target_node[0],self.target_node[1],0)
    def __hash__(self):
        return hash(self.target_node)

    def __eq__(self, other):
        if isinstance(other, MoveTo):
            return self.target_node == other.target_node

    def __str__(self):
        return f"MoveTo({self.target_node})"

    def __repr__(self):
        return f"Action(MoveTo {self.target_node})"

ALL_MOTION_ACTIONS = [TurnLeft,TurnRight,MoveForward]



