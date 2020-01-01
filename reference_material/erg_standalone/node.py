#!/usr/bin/python

import time
import math
import numpy as np
from datetime import datetime
from math import pi, cos, sin
from utilities import Utilities
from global_pose import GlobalPose
from position_controller import PositionController

import tf
import rospy
import rosnode
import actionlib
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from gps_common.msg import GPSFix
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import NavSatFix
from move_base_msgs.msg import MoveBaseAction
from geometry_msgs.msg import Quaternion, Twist
from global_position_controller.srv import GoalPosition, GoalPositionResponse

class Node:

    def __init__(self):

        self.controller = PositionController()

        self.utilities = Utilities()

        self.pose = GlobalPose()

        self.target_pose = GlobalPose()

        self.goal = PoseStamped()

        self.new_goal = PoseStamped()

        self.response = GoalPositionResponse()

        self.check_msg = Float64()

        self.loop_threshold = 2.0 # used to determine  if position is reached

        self.send_threshold = 2.0 # used to determine whether to resend goal or not

        self.control_status = 'run'

        rospy.init_node('GLOBAL_POS')

        self.rate = 0.1

        self.sub_gps = rospy.Subscriber('gps_fix', GPSFix, self.gps_callback)

        self.sub_manage = rospy.Subscriber('manage_controller', String, self.manage_callback)

        self.srv_cmd_position = rospy.Service('goto_position', GoalPosition, self.goto_position_callback)

        self.pub_goal = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size = 1)

        self.client_goal = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.pub_check = rospy.Publisher('controller_check', Float64, queue_size = 1)

        rospy.loginfo('Starting global position controller...')
        

    def manage_callback(self, msg):
        
        self.control_status = msg.data


    def gps_callback(self, msg):

        self.pose.latitude = msg.latitude

        self.pose.longitude = msg.longitude

        self.pose.heading = msg.dip * math.pi / 180.0

        self.check_msg.data = 1.0

        self.pub_check.publish(self.check_msg)


    def goto_position_callback(self, msg):

        self.control_status = 'run'

        self.target_pose.latitude = msg.target_latitude

        self.target_pose.longitude = msg.target_longitude

        self.target_pose.heading = msg.target_heading

        print(self.target_pose.latitude)

        print(self.target_pose.longitude)

        distance = 2.0 * self.loop_threshold

        # now we implement a SUPER dumb time-base position control loop
        while distance > self.loop_threshold:

            if self.control_status == 'stop':
                
                self.client_goal.cancel_goal()

                self.client_goal.cancel_all_goals()

                break

            elif self.control_status == 'pause':

                # we wait 2 seconds while paused
                self.client_goal.cancel_goal()

                self.client_goal.cancel_all_goals()
                print('this does pause')
                # time.sleep(2.0)

            elif self.control_status == 'run':

                # first we calculate the target global position in local body frame
                self.new_goal, distance = self.controller.calculate_new_goal(self.pose, self.target_pose)
                
                # publish the goal and...
                self.pub_goal.publish(self.new_goal)

            # DONT wait 5 seconds
            # time.sleep(5.0)

        # once the platform has reached its goal we cancel all move_base goals
        self.client_goal.cancel_goal()

        self.client_goal.cancel_all_goals()

        # and return the status
        self.response.status = "Done"

        return self.response


if __name__ == '__main__':

    try:

        node = Node()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass

    rospy.loginfo('Exiting')
