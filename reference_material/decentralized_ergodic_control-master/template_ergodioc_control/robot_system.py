import numpy as np
from sac import SAC
from settings import Settings
from system_dynamics import Quadcopter
from extended_kalman_filter import BearingOnlySensor, DiffusionProcess, EKF

class RobotSystem(object):

    def __init__(self, time_step, robot_type, no):
        # initialize the time stepper
        self.time_step = time_step
        # needs some dynamics
        self.sys_dyn = Quadcopter(time_step)
        # set up the settings for the robot
        self.config = Settings(time_step=time_step, robot_type = robot_type)
        self.eid = self.config.eid
        # needs a controller
        self.controller = SAC(self.config)
        # needs a sensor
        self.sensor = BearingOnlySensor()
        # add a filter
        self.kalman_filter = EKF(DiffusionProcess(time_step), self.sensor)
        # start the robot off in a good place
        self.state = np.array([.2,.3,0.] + [0.0]*9)
        position = [np.random.uniform(0,1), np.random.uniform(0,1)]
        self.state = np.array(  position + [0.0]*10)
        self.id = no
        self.control_dump = []
        self.state_dump = []
        self.mean_dump = []
        self.covariance_dump = []

    def step(self, yk):
        if yk is not None:
            yk += np.random.normal([0.]*2, np.diag(self.sensor.R))
        self.kalman_filter.update(yk, self.state[0:3])
        # self.eid.update_eid(self.kalman_filter.mu, self.kalman_filter.sigma)
        self.eid.update_eid(self.sensor.fisher_information_matrix, self.kalman_filter.mu, self.kalman_filter.sigma, self.state[0:3])
        u = self.controller(self.state, self.eid.phik)
        self.state = self.sys_dyn.step(self.state, u)

        self.control_dump.append(u)
        self.state_dump.append(self.state.copy())
        self.mean_dump.append(self.kalman_filter.mu.copy())
        self.covariance_dump.append(self.kalman_filter.sigma.ravel().copy())
    def save_data(self, filePath=''):
        np.savetxt( filePath + 'robot_state_data{}.csv'.format(self.id)  , self.state_dump )
        np.savetxt( filePath + 'target_mean_data{}.csv'.format(self.id), self.mean_dump)
        np.savetxt( filePath + 'target_covar_data{}.csv'.format(self.id), self.covariance_dump)
