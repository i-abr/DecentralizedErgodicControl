import numpy as np
from ergodic.sac import SAC
from ergodic.settings import Settings
from ergodic.system_dynamics import Quadcopter
from ergodic.phik import Phik
from ergodic.extended_kalman_filter import BearingOnlySensor, DiffusionProcess, EKF
import matplotlib.pyplot as plt
from terrain.city_block import g_city_terrain

class Agent(object):

    def __init__(self, time_step, no_targets):
        self.settings = Settings(time_step) # initialize the settings with a time step
        self.time_step = time_step # keep that here
        self.sys_dyn = Quadcopter(time_step) # also include the dynamics of the robot
        self.controller = SAC(self.settings) # add the sac controller

        self.no_targets = no_targets

        self.state = np.array([.2,.3,0.] + [0.0]*9) # add the state where the robot starts in
        position = [np.random.uniform(0.15,0.25), np.random.uniform(0.15,0.25)]
        while g_city_terrain.isInBlock(position):
            position = [np.random.uniform(0.15,0.25), np.random.uniform(0.15,0.25)]
        self.state = np.array(  position + [0.0]*10)
        self.phik = Phik(self.settings, no_targets)
        ######## Containers #############
        self.control_dump = []
        self.state_dump = []
        self.mean_dump = {}
        self.covariance_dump = {}
        for target_no in range(self.no_targets):
            self.mean_dump.update({target_no : []})
            self.covariance_dump.update({target_no : []})
        self.phik_dump = []
        self.ck_dump = []


    def step(self, ck=None):

        u = self.controller(self.state, self.phik.phik, ck=ck)
        self.state = self.sys_dyn.step(self.state, u)
        self.control_dump.append(u)
        self.phik_dump.append(self.phik.phik.copy())
        self.state_dump.append(self.state.copy())
        for target_no in range(self.no_targets):
            self.mean_dump[target_no].append(self.phik.ekfs[target_no].mu.copy())
            self.covariance_dump[target_no].append(self.phik.ekfs[target_no].sigma.copy())
        self.ck_dump.append(self.controller.ck.cki.copy())

    def save_data(self, filePath=''):
        np.save( filePath + 'robot_state_data.npy' , self.state_dump )
        np.save( filePath + 'ck_data.npy', self.ck_dump)
        for target_no in range(self.no_targets):
            np.save( filePath + 'target{}_mean_data.npy'.format(target_no), self.mean_dump[target_no])
            np.save( filePath + 'target{}_covar_data.npy'.format(target_no), self.covariance_dump[target_no])
        np.save( filePath + 'phik_data.npy', self.phik_dump)
        np.save( filePath + 'control_data.npy', self.control_dump)
    def plotTrajectory(self,ax):
        trajectory = np.array(self.state_dump)
        ax.plot(trajectory[:,0], trajectory[:,1])
