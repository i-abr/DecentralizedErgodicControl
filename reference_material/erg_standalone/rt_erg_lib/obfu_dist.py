import rospy

import numpy as np
import numpy.random as npr
from utils import convert_phi2phik

from geometry_msgs.msg import Pose

class ObfuDist(object):

    def __init__(self, basis, agent_num, num_nodes=2, num_pts=50):

        self.basis = basis
        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        self.grid_vals = self.init_uniform_grid(self.grid)
        
        self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)
        self.agent_num = agent_num
        rospy.Subscriber('/agent0_loc', Pose, self.callback)

    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik.copy()
            
    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)

    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.ones(x.shape[0])
        val /= np.sum(val)
        return val

    def callback(self, data):
        if self.agent_num == 0:
            pass
        else: 
            self.means = [np.array([data.position.x, data.position.y])]
            self.vars  = [np.array([0.1,0.1])**2]
            val = np.zeros(self.grid.shape[0])
            for m, v in zip(self.means, self.vars):
                innerds = np.sum((x-m)**2 / v, 1)
                val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
            # normalizes the distribution
            val /= np.sum(val)
            self.grid_vals = val
            self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)
            