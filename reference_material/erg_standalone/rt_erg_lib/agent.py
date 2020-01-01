from single_integrator import SingleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np


class Agent(SingleIntegrator):

    def __init__(self, agent_name):


        SingleIntegrator.__init__(self)

        self.agent_name  = agent_name

        self.model = SingleIntegrator()

        self.planner = RTErgodicControl(self.agent_name, self.model,
                                horizon=15, num_basis=10, batch_size=100)

        self.reset() # this is overwritten with the gps call back

    def planner_step(self):
        # the idea here is to use the agent as a planner to forward plan
        # where we want the rover to go and the rover will use a position
        # controller to wrap around the planner
        ctrl = self.planner(self.state)
        return self.step(ctrl)
