import rospy

import numpy as np
import numpy.random as npr
from utils import convert_phi2phik

from geometry_msgs.msg import Pose
from tanvas_comms.msg import input_array

class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    def __init__(self, basis, num_nodes=2, num_pts=50):

        self.basis = basis
        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        # Initialize with Uniform Exploration
        self.grid_vals = self.init_uniform_grid(self.grid)
        self._phik = convert_phi2phik(self.basis, self.grid_vals, self.grid)

        self.has_update = False
        # Update for Environmental Stimuli (EEs & DDs)
        self.dd_means = []
        self.ee_means = []
        self.dd_vars = []
        self.ee_vars = []
        self._ee = False
        self._dd = False
        
        rospy.Subscriber('/ee_loc', Pose, self.ee_callback)
        rospy.Subscriber('/dd_loc', Pose, self.dd_callback)

        # Update for Tanvas Inputs
        self._tanvas = False
        rospy.Subscriber('/input', input_array, self.tanvas_callback)

        
    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik not right dim'
        self._phik = self.phik.copy()

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

    def dd_callback(self,data):
        print("updating dd location dist")
        self.dd_means.append(np.array([data.position.x, data.position.y]))
        self.dd_vars.append(np.array([0.1,0.1]))
        self._dd = True
        
        ee_val = np.zeros(self.grid.shape[0])
        ee_scale = np.ones(self.grid.shape[0])
        if self._ee: 
            for m, v in zip(self.ee_means, self.ee_vars):
                innerds = np.sum((self.grid-m)**2 / v, 1)
                ee_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
            ee_val /= np.sum(ee_val)
            ee_scale = ee_val

        dd_val = np.zeros(self.grid.shape[0])
        dd_scale = np.ones(self.grid.shape[0])
        for m, v in zip(self.dd_means, self.dd_vars):
            innerds = np.sum((self.grid-m)**2 / v, 1)
            dd_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
        dd_scale = dd_val
        # Invert DD distribution
        dd_val -= np.max(dd_val)
        dd_val = np.abs(dd_val)#+1e-5
        dd_val /= np.sum(dd_val)

        if self._tanvas: 
            val = (self.tanvas_dist + ee_val + dd_val) *  dd_scale 
        else:
            val = (ee_val + dd_val)  *  dd_scale

        # normalizes the distribution
        val /= np.sum(val)
        self.grid_vals = val
        self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)

        #self.has_update = True


    def ee_callback(self,data):
        print("updating ee location dist")
        self.ee_means.append(np.array([data.position.x, data.position.y]))
        self.ee_vars.append(np.array([0.1,0.1])**2)

        self._ee = True
        ee_val = np.zeros(self.grid.shape[0])
        ee_scale = np.ones(self.grid.shape[0])
        for m, v in zip(self.ee_means, self.ee_vars):
            innerds = np.sum((self.grid-m)**2 / v, 1)
            ee_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
        ee_val /= np.sum(ee_val)
        ee_scale = ee_val

        dd_scale = np.ones(self.grid.shape[0])
        dd_val = np.zeros(self.grid.shape[0])
        if self._dd: 
            for m, v in zip(self.dd_means, self.dd_vars):
                innerds = np.sum((self.grid-m)**2 / v, 1)
                dd_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
            # Invert DD distribution
            dd_val -= np.max(dd_val)
            dd_val = np.abs(dd_val)#+1e-5
            dd_val /= np.sum(dd_val)

            dd_scale = dd_val

        if self._tanvas: 
            val = (self.tanvas_dist + ee_val + dd_val) * dd_scale
        else:
            val = (ee_val + dd_val) * dd_scale
        # normalizes the distribution
        val /= np.sum(val)
        self.grid_vals = val
        self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)

        #self.has_update = True


    def tanvas_callback(self, data):
        if data.datalen != 0:
            self._tanvas = True
            print('received tanvas input')
            tan_x = data.xinput
            tan_y = data.yinput

            grid_lenx = 50
            grid_leny = 50
            tan_arr = np.ones((grid_lenx, grid_leny))*.05#*.0001
            for i in range(data.datalen):
                tan_arr[tan_x[i], tan_y[i]] = 1.0
            tan_arr = np.transpose(tan_arr)
            tan_arr = tan_arr.ravel()
            if np.max(tan_arr) > 0:
                    temp = tan_arr
            target_dist = temp            
            target_dist /= np.sum(target_dist)
            self.tanvas_dist = target_dist

            
            grid = np.meshgrid(*[np.linspace(0,1,grid_lenx), np.linspace(0,1,grid_leny)])
            self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
            # self._phik = convert_phi2phik(self.basis,target_dist, self.grid)

            ee_val = np.zeros(self.grid.shape[0])
            ee_scale = np.ones(self.grid.shape[0])
            if self._ee: 
                for m, v in zip(self.ee_means, self.ee_vars):
                    innerds = np.sum((self.grid-m)**2 / v, 1)
                    ee_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
                ee_val /= np.sum(ee_val)
                ee_scale = ee_val

            dd_val = np.zeros(self.grid.shape[0])
            dd_scale = np.ones(self.grid.shape[0])
            if self._dd: 
                for m, v in zip(self.dd_means, self.dd_vars):
                    innerds = np.sum((self.grid-m)**2 / v, 1)
                    dd_val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
                # Invert DD distribution
                dd_val -= np.max(dd_val)
                dd_val = np.abs(dd_val)#+1e-5
                dd_val /= np.sum(dd_val)

                dd_scale = dd_val
            val =  (target_dist + ee_val +  dd_val) * dd_scale
            # normalizes the distribution
            val /= np.sum(val)
            self.grid_vals = val
            self._phik = convert_phi2phik(self.basis,self.grid_vals,self.grid)

            self.has_update = True

