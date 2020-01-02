import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
import tf
from tf import transformations as trans


class QuadVisual(object):

    def __init__(self, agent_names, scale=0.1):

        rospy.init_node("agent_rendering")

        self._agent_names = agent_names
        self._scale      = scale

        self._agent_markers = MarkerArray()
        self._markers = [
            Marker() for i in range(len(agent_names))
        ]
        self._agent_markers.markers = self._markers

        # self._agent_marker   = Marker()
        # self._path_track     = Marker()
        # self._sensor_range   = Marker() # maybe not used for all

        # instantiatiate the publishers
        self._marker_pub = rospy.Publisher('agent/visual', MarkerArray, queue_size=1)
        self.__build_rendering()
        self.listener = tf.TransformListener()

        self._rate = rospy.Rate(10)

    def run(self):
        while not rospy.is_shutdown():
            self.update_rendering()
            self._rate.sleep()


    def update_rendering(self):
        for agent_name, marker in zip(self._agent_names, self._agent_markers.markers):
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
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue




        # self._path_track.points = [
        #     Point(x[0]*10.,x[1]*10., self._agent_marker.pose.position.z) for x in path
        # ]

        self._marker_pub.publish(self._agent_markers)
        # self._path_pub.publish(self._path_track)

    def __build_rendering(self):

        for name, agent_marker in zip(self._agent_names, self._markers):
            rgb = np.random.uniform(0,1, size=(3,))
            scale = 0.25
            agent_marker.header.frame_id = "world"
            agent_marker.header.stamp = rospy.Time(0)
            agent_marker.ns = name
            agent_marker.id = 0
            agent_marker.action = Marker.ADD
            agent_marker.scale.x = scale
            agent_marker.scale.y = scale
            agent_marker.scale.z = scale
            agent_marker.color.a = 1.0
            agent_marker.color.r = rgb[0]
            agent_marker.color.g = rgb[1]
            agent_marker.color.b = rgb[2]
            agent_marker.pose.position.z = np.random.uniform(1.6,4)
            agent_marker.type = Marker.MESH_RESOURCE
            agent_marker.mesh_resource = "package://decentralized_ergodic/mesh/quad_base.stl"

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
