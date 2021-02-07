import numpy as np
# import autograd.numpy as anp
# from autograd import grad
import numpy.random as npr
import rospy
from visualization_msgs.msg import Marker, MarkerArray

import tf

from .rendering import Visual

# def cost(_s, gBA, dim):
#     return anp.prod(0.5*(anp.tanh(-20 * (anp.abs(anp.dot(gBA,_s))-dim))+1))
#
# dcost = grad(cost)

def t_mat(p, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), p[0]],
        [np.sin(theta), np.cos(theta), p[1]],
        [0., 0., 1.]
    ])

class Object(object):

    def __init__(self, p, theta, l, w):

        # length, width
        self.l = l
        self.w = w
        self.dim = np.array([l, w])
        self._p = p
        self._t_mat = t_mat(p, theta)
        self._quat = tf.transformations.quaternion_from_euler(0,0,theta)

    def is_close(self, x):
        xh = np.concatenate([x, 1], axis=0) # homogenous
        xt = np.abs(np.dot(self._trans, xh))[:2]
        if (self.dim < xt+0.1).any():
            return 1.0
        return False


class Map(Visual):

    def __init__(self, num_objs=10):

        rospy.init_node('env')
        self.objects = []

        for o in range(num_objs):
            p = npr.uniform(0, 1, size=(2,))
            theta = npr.uniform(-np.pi, np.pi)
            l,w = npr.uniform(0.1,0.3, size=(2,))
            self.objects.append(Object(p, theta, l, w))

        Visual.__init__(self)

        self.update_rendering()
        self._rate = rospy.Rate(1)

    def run(self):
        while not rospy.is_shutdown():
            self.update_rendering() # TODO: is this even necessary?
            self._rate.sleep()
