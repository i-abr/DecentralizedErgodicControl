import numpy as np
from ergodic.sac import SAC
from ergodic.settings import Settings
from ergodic.system_dynamics import Quadcopter
from ergodic.phik import Phik
from recursive_least_squares import RecursiveLeastSquares
from gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
from collections import deque

class Agent(object):

    def __init__(self, time_step):
        self.settings = Settings(time_step) # initialize the settings with a time step
        self.time_step = time_step # keep that here
        self.sys_dyn = Quadcopter(time_step) # also include the dynamics of the robot
        self.controller = SAC(self.settings) # add the sac controller

        # self.estimator = RecursiveLeastSquares()
        self.estimator = GaussianProcess()
        self.state = np.array([.2,.3,0.] + [0.0]*9) # add the state where the robot starts in
        position = [np.random.uniform(0.45,0.55), np.random.uniform(0.45,0.55)] # randomize it
        self.state = np.array(  position + [0.0]*10)
        self.phik = Phik(self.settings)
        ######## Containers #############
        self.control_dump = []
        self.state_dump = []
        self.mean_dump = []
        self.covariance_dump = []
        self.phik_dump = []
        self.ck_dump = []

        self.X = []
        self.y = []

    def updateBelief(self, yk):
        # self.estimator.updateEstimate(yk, self.state[0:2].copy())
        self.y.append(yk)
        self.X.append(self.state[0:2].copy())
        self.phik.update_eid()

    def step(self, ck):
        u = self.controller(self.state, self.phik.phik, ck=ck)
        self.state = self.sys_dyn.step(self.state, u)
        self.controller.ck.update_ck(self.state.copy())

        self.control_dump.append(u)
        self.phik_dump.append(self.phik.phik.copy())
        self.state_dump.append(self.state.copy())
        self.ck_dump.append(self.controller.ck.cki.copy())

    def save_data(self, filePath=''):
        np.savetxt( filePath + 'robot_state_data.csv' , self.state_dump )
        np.savetxt( filePath + 'ck_data.csv', self.ck_dump)
        np.savetxt( filePath + 'phik_data.csv', self.phik_dump)
        np.savetxt( filePath + 'control_data.csv', self.control_dump)
        np.savetxt( filePath + 'X_data.csv', self.X)
        np.savetxt( filePath + 'y_data.csv', self.y)

    def plotTrajectory(self, ax=False):
        trajectory = np.array(self.state_dump)
        if ax is False:
            plt.plot(trajectory[:,0], trajectory[:,1])
        else:
            ax.plot(trajectory[:,0], trajectory[:,1])
