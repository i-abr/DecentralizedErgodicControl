#!/usr/bin/env python

import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray, Pose
import tf
import math
import actionlib
from move_base_msgs.msg import MoveBaseAction
from global_position_controller.srv import GoalPosition
import utm

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
# from global_position_controller.srv import GoalPosition, GoalPositionResponse
from rt_erg_lib import Agent
from ccast_services.srv import GetHardwarePlatformUid, CheckInObstacle

class ControlNode(Agent):
    def __init__(self):

        rospy.init_node("control_node", anonymous=True)

        # --- new code
        self._controller = PositionController()
        self._utilities = Utilities()
        self._pose = GlobalPose()
        self._target_pose = GlobalPose()
        self._check_msg = Float64()
        self._control_status = 'run'
        self._rate = 1 

        self.rate = rospy.Rate(self._rate)

        self._coord1 = Pose()
        self._coord2 = Pose()

        # Grassy Test Coordinates
        self._coord1.position.x = 31.1379680
        self._coord1.position.y = -89.0647412

        self._coord2.position.x = 31.137823
        self._coord2.position.y = -89.064585

        # Range Full Coords
        #self._coord1.position.x = 31.1364917
        #self._coord1.position.y = -89.0650045

        #self._coord2.position.x = 31.1377765
        #self._coord2.position.y = -89.0620306

        ## Range Main Village
        #self._coord1.position.x = 31.1366588
        #self._coord1.position.y = -89.0638039

        #self._coord2.position.x = 31.1377765
        #self._coord2.position.y = -89.0620306
        
        # Range Main Village
        self._coord1.position.x = 31.13770839
        self._coord1.position.y = -89.0646414

        self._coord2.position.x = 31.1378790
        self._coord2.position.y = -89.0644859

        self._delta_x = self._coord2.position.x-self._coord1.position.x
        self._delta_y = self._coord2.position.y-self._coord1.position.y

        self._sub_gps = rospy.Subscriber('gps_fix', GPSFix, self.gps_callback)
        self._sub_manage = rospy.Subscriber('manage_controller', String, self.manage_callback)

        self._agent_id_pub = rospy.Publisher('agent_id', String)
        
        self._client_goal = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self._pub_goal = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size = 1)
        self._pub_check = rospy.Publisher('controller_check', Float64, queue_size = 1)

        self.curr_lat = None
        self.curr_lon = None

        rospy.Subscriber('grid_pts', PoseArray, self.get_grid)

        self.pose_publisher = rospy.Publisher('target_pose', PoseStamped, queue_size=1)
        self.goal_pose = GoalPosition()
        self.goal_pose.target_heading = -1
        
        self.agent_id = rospy.ServiceProxy('get_hardware_platform_uid', GetHardwarePlatformUid)().uid
        print(self.agent_id)
        self.convert_agent_id_to_name()
        print('agent name is', self.agent_name)
        agent_id_msg = String()
        agent_id_msg.data = self.agent_id
        self._agent_id_pub.publish(agent_id_msg)

        Agent.__init__(self, self.agent_name)

        #self._check_obs = rospy.ServiceProxy('check_in_obstacle_{}'.format(self.agent_id), CheckInObstacle)

    def convert_agent_id_to_name(self):
        agent_id =''
        for i in range(len(self.agent_id[4:])):
            agent_id+=str(ord(self.agent_id[i]))
        self.agent_name = int(agent_id)
        
    def manage_callback(self, msg):
        self.control_status = msg.data

    def gps_callback(self, msg):
        self._pose.latitude = msg.latitude
        self._pose.longitude = msg.longitude
        self._pose.heading = msg.dip * math.pi / 180.0
        local_pose = self._coord_to_dist(self._pose)
        self.state = np.array(local_pose)
        self._check_msg.data = 1.0
        self._pub_check.publish(self._check_msg)


    def _coord_to_dist(self, coord):
        x = (coord.latitude-self._coord1.position.x)/self._delta_x
        y = (coord.longitude-self._coord1.position.y)/self._delta_y
        return [x, y]

    def _dist_to_coord(self, dist):
        lat = dist[0] * self._delta_x + self._coord1.position.x
        lon = dist[1] * self._delta_y + self._coord1.position.y
        return [lat, lon]

    def get_grid(self, msg):
        pass
        #print(msg)
        #self._coord1 = msg.poses[0]
        #self._coord2 = msg.poses[1]
        #self._delta_x = self._coord2.position.x-self._coord1.position.x
        #self._delta_y = self._coord2.position.y-self._coord1.position.y


    def step_rover(self):
        if self._control_status == 'run':
            if (self._coord1 is None) or self._coord2 is None:
                pass
            else:
                next_pose = self.planner_step()
                
                coord = self._dist_to_coord(next_pose)
                self._target_pose.latitude = coord[0]
                self._target_pose.longitude = coord[1]
                self._target_pose.heading = -1
                new_goal, distance = self._controller.calculate_new_goal(self._pose, self._target_pose)
                try:
                    #is_in_obstacle = self._check_obs(self._target_pose.latitude, self._target_pose.longitude, 0)
                    # DO service
                    #if is_in_obstacle.inObstacle:
                    #    self.planner.replay_buffer.push(next_pose[self.planner.model.explr_idx].copy())
                    #    print(self.state)
                    #else:
                    print("Sending pose")
                    print("(%f,%f)" % (coord[0], coord[1]))
                    print('next pose', next_pose, 'curr state', self.state)
                    print('MOOOOVVVVVEEE')
                    self._pub_goal.publish(new_goal)        
                    self.rate.sleep()

                except rospy.ServiceException, e:
                    print "Service call failed: %s"%e


if __name__ == '__main__':
    rospy.wait_for_service('goto_position')
    print("Got the goto_position service.")
    gv = ControlNode()
    try:
        while not rospy.is_shutdown():
            gv.step_rover()
    except rospy.ROSInterruptException as e:
        print('clean break')
