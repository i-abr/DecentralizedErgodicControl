import numpy as np
from ergodic.sac import SAC
from ergodic.settings import Settings
from ergodic.system_dynamics import Quadcopter
from ergodic.phik import Phik
from ergodic.extended_kalman_filter import BearingOnlySensor, DiffusionProcess, EKF
import matplotlib.pyplot as plt

class Agent(object):

    def __init__(self, time_step):
        self.settings = Settings(time_step) # initialize the settings with a time step
        self.time_step = time_step # keep that here
        self.sys_dyn = Quadcopter(time_step) # also include the dynamics of the robot
        self.controller = SAC(self.settings) # add the sac controller
        self.sensor = BearingOnlySensor() # add the sensor
        self.kalman_filter = EKF(DiffusionProcess(time_step), self.sensor) # also add the kalman filter
        self.state = np.array([.2,.3,0.] + [0.0]*9) # add the state where the robot starts in
        position = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)] # randomize it
        self.state = np.array(  position + [0.0]*10)
        self.phik = Phik(self.settings)
        ######## Containers #############
        self.control_dump = []
        self.state_dump = []
        self.mean_dump = []
        self.covariance_dump = []
        self.phik_dump = []
        self.ck_dump = []

    def updateBelief(self, yk):
        if yk is not None:
            yk += np.random.normal([0.]*2, np.diag(self.sensor.R))
        self.kalman_filter.update(yk, self.state[0:3])
        # self.phik.update_eid(self.kalman_filter.mu, self.kalman_filter.sigma)
        self.phik.update_eid(self.sensor.fisher_information_matrix, self.kalman_filter.mu,\
                        self.kalman_filter.sigma, self.state[0:3])

    def step(self):

        u = self.controller(self.state, self.phik.phik)
        self.state = self.sys_dyn.step(self.state, u)
        self.control_dump.append(u)
        self.phik_dump.append(self.phik.phik.copy())
        self.state_dump.append(self.state.copy())
        self.mean_dump.append(self.kalman_filter.mu.copy())
        self.covariance_dump.append(self.kalman_filter.sigma.ravel().copy())
        self.ck_dump.append(self.controller.ck.cki.copy())

    def save_data(self, filePath=''):
        np.savetxt( filePath + 'robot_state_data.csv' , self.state_dump )
        np.savetxt( filePath + 'ck_data.csv', self.ck_dump)
        np.savetxt( filePath + 'target_mean_data.csv', self.mean_dump)
        np.savetxt( filePath + 'target_covar_data.csv', self.covariance_dump)
        np.savetxt( filePath + 'phik_data.csv', self.phik_dump)
        np.savetxt( filePath + 'control_data.csv', self.control_dump)
    def plotTrajectory(self,ax):
        trajectory = np.array(self.state_dump)
        ax.plot(trajectory[:,0], trajectory[:,1])
