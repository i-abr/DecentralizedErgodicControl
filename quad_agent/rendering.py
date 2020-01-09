import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
import tf
from tf import transformations as trans
from tanvas_comms.msg import input_array

import seaborn as sns
sns_palette = sns.color_palette("bright")
palette = [sns_palette[2],sns_palette[3],sns_palette[6],sns_palette[7],sns_palette[8],sns_palette[9]]

class QuadVisual(object):

    def __init__(self, agent_names, scale=0.1):

        rospy.init_node("agent_rendering")

        self._agent_names = agent_names
        self._scale      = scale

        self._agent_markers = MarkerArray()
        self._markers = [
            Marker() for i in range(len(agent_names))
        ]
        self._path_markers = [
            Marker() for i in range(len(agent_names))
        ]
        self._agent_markers.markers = self._markers + self._path_markers

        # self._agent_marker   = Marker()
        # self._path_track     = Marker()
        # self._sensor_range   = Marker() # maybe not used for all

        # instantiatiate the publishers
        self._marker_pub = rospy.Publisher('agent/visual', MarkerArray, queue_size=1)
        rospy.Subscriber('/ee_loc',Pose,self.tdist_callback)
        rospy.Subscriber('/dd_loc',Pose,self.tdist_callback)
        rospy.Subscriber('/input',input_array,self.tdist_callback)
        
        self.__build_rendering()
        self.listener = tf.TransformListener()

        self._rate = rospy.Rate(10)

    def run(self):
        while not rospy.is_shutdown():
            self.update_rendering()
            self._rate.sleep()

    def tdist_callback(self,data):
        for agent_name, line_m in zip(self._agent_names, self._path_markers):
            del line_m.points[:]
            
    def update_rendering(self):
        for agent_name, marker, line_m in zip(self._agent_names, self._markers, self._path_markers):
            try:
                (trans, rot) = self.listener.lookupTransform(
                    "world", agent_name, rospy.Time(0)
                )
                marker.pose.position.x = trans[0]
                marker.pose.position.y = trans[1]

                marker.pose.orientation.x = rot[0]
                marker.pose.orientation.y = rot[1]
                marker.pose.orientation.z = rot[2]
                marker.pose.orientation.w = rot[3]

                line_m.points.append(
                    Point(marker.pose.position.x, marker.pose.position.y,
                          0.1))
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        self._marker_pub.publish(self._agent_markers)

    def __build_rendering(self):

        i=0
        for name, agent_marker, path_marker in zip(self._agent_names, self._markers,self._path_markers):
            rgb = np.random.uniform(0,1, size=(3,))
            quad_scale = 0.5
            agent_marker.header.frame_id = "world"
            agent_marker.header.stamp = rospy.Time(0)
            agent_marker.ns = name
            agent_marker.id = 0
            agent_marker.action = Marker.ADD
            agent_marker.scale.x = quad_scale
            agent_marker.scale.y = quad_scale
            agent_marker.scale.z = quad_scale
            agent_marker.color.a = 1.0
            agent_marker.color.r = palette[i][0] #rgb[0]
            agent_marker.color.g = palette[i][1] #rgb[1]
            agent_marker.color.b = palette[i][2] #rgb[2]
            agent_marker.pose.position.z = np.random.uniform(1.6,4)
            agent_marker.type = Marker.MESH_RESOURCE
            agent_marker.mesh_resource = "package://decentralized_ergodic/mesh/quad_base.stl"

            # Make Trajectory Lines
            line_scale = 0.1
            path_marker.header.frame_id = "world"
            path_marker.header.stamp = rospy.Time(0)
            path_marker.ns = name+'/path'
            path_marker.id = i
            path_marker.action = Marker.ADD
            path_marker.scale.x = line_scale
            path_marker.color.a = 0.7
            path_marker.color.r = palette[i][0] #rgb[0]
            path_marker.color.g = palette[i][1] #rgb[1]
            path_marker.color.b = palette[i][2] #rgb[2]
            path_marker.pose.position.z = 0.1
            path_marker.type = Marker.LINE_STRIP

            i+=1

        # self._path_track.header.frame_id = "world"
        # self._path_track.header.stamp = rospy.Time(0)
        # self._path_track.ns = self._agent_name
        # self._path_track.id = 1
        # self._path_track.action = Marker.ADD
        # self._path_track.points = []
        # self._path_track.scale.x = 0.01
        # self._path_track.color.a = 1.0
        # self._path_track.color.r = rgb[0]
        # self._path_track.color.g = rgb[1]
        # self._path_track.color.b = rgb[2]
        # self._path_track.type = Marker.LINE_STRIP
        #
        # self._sensor_range.header.frame_id = "world"
        # self._sensor_range.header.stamp = rospy.Time(0)
        # self._sensor_range.ns = self._agent_name
        # self._sensor_range.id = 1
        # self._sensor_range.action = Marker.ADD
        # self._sensor_range.points = []
        # self._sensor_range.scale.x = 0.4
        # self._sensor_range.scale.y = 0.4
        # self._sensor_range.scale.z = 0.01
        # self._sensor_range.color.a = 0.2
        # self._sensor_range.color.r = rgb[0]
        # self._sensor_range.color.g = rgb[1]
        # self._sensor_range.color.b = rgb[2]
        #
        # self._sensor_range.type = Marker.CYLINDER
