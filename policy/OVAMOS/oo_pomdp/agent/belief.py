# Defines the belief distribution and update for the 2D Multi-Object Search domain;
#
# The belief distribution is represented as a Histogram (or Tabular representation).
# Since the observation only contains mapping from object id to their location,
# the belief update has no leverage on the shape of the sensing region; this is
# makes the belief update algorithm more general to any sensing region but then
# requires updating the belief by iterating over the state space in a nested
# loop. The alternative is to use particle representation but also object-oriented.
# We try both here.
#
# We can directly make use of the Histogram and Particle classes in pomdp_py.
import pomdp_py
import random
import copy
from vlfm.policy.OVAMOS.oo_pomdp.domain.state import *
from sklearn.cluster import DBSCAN

# robot_particle = RobotState(2,(0,0,0),0)
class MosOOBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""

    def __init__(self, robot_id, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.robot_id = robot_id
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return MosOOState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return MosOOState(pomdp_py.OOBelief.random(self, **kwargs).object_states)


def initialize_belief(
    dim,
    robot_id,
    object_ids,
    prior={},
    representation="histogram",
    robot_orientations={},
    num_particles=100,
):
    """
    Returns a GenerativeDistribution that is the belief representation for
    the multi-object search problem.

    Args:
        dim (tuple): a tuple (width, length) of the search space gridworld.
        robot_id (int): robot id that this belief is initialized for.
        object_ids (dict): a set of object ids that we want to model the belief distribution
                          over; They are `assumed` to be the target objects, not obstacles,
                          because the robot doesn't really care about obstacle locations and
                          modeling them just adds computation cost.
        prior (dict): A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}. If used, then
                      all locations not included in the prior will be treated to have 0 probability.
                      If unspecified for an object, then the belief over that object is assumed
                      to be a uniform distribution.
        robot_orientations (dict): Mapping from robot id to their initial orientation (radian).
                                   Assumed to be 0 if robot id not in this dictionary.
        num_particles (int): Maximum number of particles used to represent the belief

    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if representation == "histogram":
        return _initialize_histogram_belief(
            dim, robot_id, object_ids, prior, robot_orientations
        )
    elif representation == "particles":
        return _initialize_particles_belief(
            dim, robot_id, object_ids,prior,robot_orientations, num_particles=num_particles
        )
    else:
        raise ValueError("Unsupported belief representation %s" % representation)


def _initialize_histogram_belief(dim, robot_id, object_ids, prior, robot_orientations):
    """
    Returns the belief distribution represented as a histogram
    """
    oo_hists = {}  # objid -> Histogram
    width, length = dim
    for objid in object_ids:
        hist = {}  # pose -> prob
        total_prob = 0
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            for pose in prior[objid]:
                state = ObjectState(objid, pose)
                hist[state] = prior[objid][pose]
                total_prob += hist[state]
        else:
            # no prior knowledge. So uniform.
            for x in range(width):
                for y in range(length):
                    state = ObjectState(objid, (x, y))
                    hist[state] = 1.0
                    total_prob += hist[state]

        # Normalize
        for state in hist:
            hist[state] /= total_prob

        hist_belief = pomdp_py.Histogram(hist)
        oo_hists[objid] = hist_belief

    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    oo_hists[robot_id] = pomdp_py.Histogram(
        {RobotState(robot_id, init_robot_pose,0): 1.0}
    )

    return MosOOBelief(robot_id, oo_hists)


def _initialize_particles_belief(
    dim, robot_id, object_ids, prior, robot_orientations, num_particles=100
):
    """This returns a single set of particles that represent the distribution over a
    joint state space of all objects.

    Since it is very difficult to provide a prior knowledge over the joint state
    space when the number of objects scales, the prior (which is
    object-oriented), is used to create particles separately for each object to
    satisfy the prior; That is, particles beliefs are generated for each object
    as if object_oriented=True. Then, `num_particles` number of particles with
    joint state is sampled randomly from these particle beliefs.

    """
    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    robot_particle = RobotState(robot_id, init_robot_pose,0)
    oo_particles = {}  # objid -> Particageles
    width, length = (10,10)
    for objid in object_ids:
        particles = []  # list of states; Starting the observable robot state.
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledgeW
            prior_sum = int(sum(prior[objid][pose] for pose in prior[objid]))
            for pose in prior[objid]:
                state = ObjectState(objid, pose)
                amount_to_add = int((prior[objid][pose] / prior_sum) * num_particles)
                for _ in range(amount_to_add):
                    particles.append(state)
        else:
            # no prior knowledge. So uniformly sample `num_particles` number of states.
            for _ in range(num_particles):
                x = random.randrange(-width, width)
                y = random.randrange(-length, length)
                state = ObjectState(objid, (x, y))
                particles.append(state)
       
        particles_belief = pomdp_py.Particles(particles)
        oo_particles[objid] = particles_belief
    oo_particles[robot_id] = pomdp_py.Particles([robot_particle])

    # Return Particles distribution which contains particles
    # that represent joint object states
    particles = []
    for _ in range(num_particles):
        object_states = {}
        for objid in oo_particles:
            random_particle = random.sample(list(oo_particles[objid]), 1)[0]
            object_states[objid] = copy.deepcopy(random_particle)
        particles.append(MosOOState(object_states))
    return pomdp_py.Particles(particles)
# def update_robo_pose(new_robo_pose):
#     robot_particle.attributes["pose"]= new_robo_pose