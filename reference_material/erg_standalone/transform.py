import numpy as np

# Global means the lat-long frame centered at the equator
# Local is on the ergodic box defined by coor1, coor2, coor3

# When initializing an ergodic box, call this function with the bounds
#                 ex : transform = CoordTransform(origin, x, y)
#                       where orgin, x, and y are defined in lat-long
# To get the anchor's position in the lat-long frame to the frame of the ergodic box, use
#                 ex : anchor_inergbox = transform.gtol(anchor)
#                       where anchor is a np.array with x and y coordinates

class CoordTransform(object):
    def __init__(self, coord1, coord2, coord3):
        self.origin = coord1
        self.x = coord2
        self.y = coord3
        #self.test = testpoint
        self.theta = self.calc_theta()
        self.dx = np.linalg.norm(self.origin - self.x)
        self.dy = np.linalg.norm(self.origin - self.y)

    def calc_theta(self):
        trans_points=self.x-self.origin
        theta = np.arctan2(trans_points[1], trans_points[0])
        return theta

    # rotates clockwise
    def rotate(self):
        return np.array([
            [np.cos(self.theta), np.sin(self.theta)],
            [-np.sin(self.theta), np.cos(self.theta)],])

    def scale_down(self,testpoint):
        return np.array([testpoint[0]/self.dx,
            testpoint[1]/self.dy])

    def scale_up(self,testpoint):
        return np.array([testpoint[0]*self.dx,
            testpoint[1]*self.dy])

    def l_to_g(self, testpoint):
        x_drop = int(testpoint[0])
        y_drop = int(testpoint[1])
        scaling_factor = 10000
        p_scaled = np.array([testpoint[0]-x_drop,testpoint[1]-y_drop,1])*scaling_factor
        RT = self.rotate().T
        translate = np.array([
            [1, 0,  self.origin[0]],
            [0, 1,  self.origin[1]],
            [     0,      0,   1]])
        rotateP = np.array([
            [RT[0,0], RT[0,1],    0],
            [RT[1,0], RT[1,1],    0],
            [     0,    0,   1]])
        glob = np.dot(translate,np.dot(rotateP,p_scaled))
        glob_scaled = self.scale_up(glob/scaling_factor)

        return glob_scaled

    def g_to_l(self, testpoint):
        x_drop = int(testpoint[0])
        y_drop = int(testpoint[1])
        scaling_factor = 10000
        p_scaled = np.array([testpoint[0]-x_drop,testpoint[1]-y_drop,1])*scaling_factor
        R = self.rotate()
        translate = np.array([
            [1, 0,  -self.origin[0]],
            [0, 1,  -self.origin[1]],
            [     0,      0,   1]])
        rotateP = np.array([
            [R[0,0], R[0,1],    0],
            [R[1,0], R[1,1],    0],
            [     0,    0,   1]])
        local = np.dot(rotateP, np.dot(translate,testpoint))
        local_scaled = self.scale_down(local/scaling_factor)

        return local_scaled
