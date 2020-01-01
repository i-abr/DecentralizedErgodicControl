import numpy as np


class Target(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.current_time = 0.0
        self.state = np.zeros(2)
        self.movement() # initialize the state
    def movement(self):
        self.state[0] = 0.25 * np.cos( 0.3 * self.current_time) + 0.5
        self.state[1] = 0.25 * np.sin( 0.2 * self.current_time) + 0.5
    def step(self):
        self.current_time += self.time_step
        self.movement()
