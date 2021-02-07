import rospy

import numpy as np
import numpy.random as npr
from .utils import convert_phi2phik

class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    def __init__(self, basis, num_nodes=2, num_pts=50):

        # TODO: create a message class for this
        # rospy.Subscriber('/target_distribution',  CLASSNAME, self.callback)

        self.basis = basis
        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        # self.means = [npr.uniform(0.2, 0.8, size=(2,))
        #                     for _ in range(num_nodes)]
        self.means = [np.array([0.7, 0.7]), np.array([0.3,0.3]), np.array([0.2, 0.8]), np.array([0.8,0.2])]
        self.vars  = [np.array([0.1,0.1])**2, np.array([0.1,0.1])**2, np.array([0.1,0.1])**2, np.array([0.1,0.1])**2]


        self.has_update = False
        self.grid_vals = self.__call__(self.grid)
        self._phik = convert_phi2phik(self.basis, self.grid_vals, self.grid)

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


    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])
        for m, v in zip(self.means, self.vars):
            innerds = np.sum((x-m)**2 / v, 1)
            val += np.exp(-innerds/2.0)# / np.sqrt((2*np.pi)**2 * np.prod(v))
        # normalizes the distribution
        val /= np.sum(val)
        # val -= np.max(val)
        # val = np.abs(val)
        return val
