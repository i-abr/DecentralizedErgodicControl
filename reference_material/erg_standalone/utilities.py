#!/usr/bin/python

import utm
import time
import math
from global_pose import GlobalPose

import tf
from geometry_msgs.msg import PoseStamped

EAST = 0
NORTH = 1
MAX_DISTANCE = 10.0

class Utilities:

    def __init__(self):

        pass

    def calculate_distance(self, init, target):

        theta = init.heading

        init_utm = utm.from_latlon(init.latitude, init.longitude)

        target_utm = utm.from_latlon(target.latitude, target.longitude)

        east = (target_utm[EAST] - init_utm[EAST])

        north = (target_utm[NORTH] - init_utm[NORTH])

        # calculating the distance to the point
        calc_distance = math.sqrt(math.pow(east, 2.0) + math.pow(north, 2.0))

        # calculating the goal position in body frame
        calc_x = north * math.cos(theta) + east * math.sin(theta)
        
        calc_y = north * math.sin(theta) - east * math.cos(theta)

        return calc_x, calc_y, calc_distance

    
    def calculate_pose_distance(self, init, target, threshold):

        # if the current pose exists, return the distance between poses
        dx = target.pose.position.x - init.pose.position.x

        dy = target.pose.position.y - init.pose.position.y

        d = math.sqrt(math.pow(dx, 2.0) + math.pow(dy, 2.0))

        return d
