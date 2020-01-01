import numpy as np
from ergodic.sac import SAC
from ergodic.settings import Settings
from ergodic.system_dynamics import DoubleIntegrator
from ergodic.phik import Phik


class Target(object):

    def __init__(self, time_step):
        self.settings = Settings(time_step) # set up the settings for the target dynamics
        self.time_step = time_step
        self.sys_dyn = DoubleIntegrator(time_step)
        self.controller = SAC(self.settings)
        self.state = np.array([0.1,0.1, 0.,0.])
        position = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
        self.state[0] = position[0]
        self.state[1] = position[1]

        self.phik = Phik(self.settings)

        ##### Containers ######
        self.state_dump = []
        self.control_dump = []
        self.phik_dump = []


    def updateBelief(self, agent_state):
        self.phik.update_eid(agent_state)


    def step(self, ck):
        u = self.controller(self.state, self.phik.phik, ck=ck) # get the control
        self.state = self.sys_dyn.step(self.state, u) # move the system forwards
        self.control_dump.append(u)
        self.phik_dump.append(self.phik.phik.copy())
        self.state_dump.append(self.state.copy())

    def plotTrajectory(self,ax):
        trajectory = np.array(self.state_dump)
        ax.plot(trajectory[:,0], trajectory[:,1],'r--')

    def save_data(self, filePath=''):
        np.savetxt(filePath + 'target_state_data.csv', self.state_dump)
