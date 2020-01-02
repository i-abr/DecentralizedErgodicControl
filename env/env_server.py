import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray

import tf

def t_mat(p, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), p[0]],
        [np.sin(theta), np.cos(theta), p[1]],
        [0., 0., 1.]
    ])

class Objects(object):

    def __init__(self, p, theta, l, w):

        # length, width
        self.l = l
        self.w = w
        self.dim = np.array([l, w])
        self._t_mat = t_mat(p, theta)
        self._quat = tf.transformations.quaternion_from_euler(0,0,theta)

    def is_close(self, x):
        xh = np.concatenate([x, 1], axis=0) # homogenous
        xt = np.abs(np.dot(self._trans, xh))[:2]
        if (self.dim < xt+0.1).any():
            return 1.0
        return False

    def dx(self, x):


class Map(Visual):

    def __init__(self, num_objs=10):

        self.objects = [
            Object(npr.uniform(0, 1, size=(2,)), npr.uniform(-np.pi, np.pi))
        ]
