import numpy as np
import matplotlib.pyplot as plt



class Target(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.current_time = 0.0
        self.state = np.zeros(2)
        self.state[0] = 0.23
        self.state[1] = 0.7
        self.movement() # initialize the state
        self.state_dump = []

    def movement(self):
        self.state[0] = 0.25 * np.cos( 0.3 * self.current_time) + 0.5
        self.state[1] = 0.25 * np.sin( 0.2 * self.current_time) + 0.5
    def step(self):
        self.current_time += self.time_step
        self.movement()
        self.state_dump.append(self.state.copy())
    def plotTrajectory(self,ax):
        trajectory = np.array(self.state_dump)
        ax.plot(trajectory[:,0], trajectory[:,1],'r--')

    def save_data(self, filePath=''):
        np.savetxt(filePath + 'target_state_data.csv', self.state_dump)
