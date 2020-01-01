import rospy

import numpy as np
import numpy.random as npr
from utils import convert_phi2phik

from tanvas_comms.msg import input_array

class TanvasDist(object):

    def __init__(self, basis, num_nodes=2, num_pts=50):

        self.basis = basis
        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        self.has_update = False
        self.grid_vals = self.init_uniform_grid(self.grid)
        self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)
        # Tanvas Update Subscriber
        rospy.Subscriber('/input', input_array, self.tanvas_callback)
        self._update = False
    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik.copy()


    def tanvas_callback(self, data):
        if data.datalen != 0:
            print('recieved tanvas input')
            tan_x = data.xinput
            tan_y = data.yinput

            grid_lenx = 30
            grid_leny = 30
            tan_arr = np.ones((grid_lenx, grid_leny))*.0001
            for i in range(data.datalen):
                tan_arr[tan_x[i], tan_y[i]] = 1.0
            tan_arr = np.transpose(tan_arr)
            tan_arr = tan_arr.ravel()
            if np.max(tan_arr) > 0:
                    temp = tan_arr
            target_dist = temp
            target_dist /= np.sum(target_dist)
            grid = np.meshgrid(*[np.linspace(0,1,grid_lenx), np.linspace(0,1,grid_leny)])
            self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
            self._phik = convert_phi2phik(self.basis,target_dist, self.grid)
            self._update = True
            
            
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
        
