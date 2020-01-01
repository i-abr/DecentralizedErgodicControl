import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans



class Visual(object):

    def __init__(self, agent_name, scale-0.1):

        # initialize the markers
        # TODO: figure out how the rover will render (if with full CAD URDF model)
        self._agent_name = agent_name
        self._scale      = scale

        self._agent_marker   = Marker()
        self._path_track     = Marker()
        self._sensor_range   = Marker() # maybe not used for all

        # instantiatiate the publishers
        self._marker_pub         = rospy.Publisher(agent_name + '/visual', Marker, queue_size=10)
        self._path_pub           = rospy.Publisher(agent_name + '/path', Marker, queue_size=10)
        self._sensor_range_pub   = rospy.Publisher(agent_name + '/sensor', Marker, queue_size=10)


    def update_rendering(self):
        # TODO: shpuld also publish the rendering
        pass

    def __build_rendering(self):

        rgb = np.random.uniform(0,1, size=(3,))

        scale = 0.1
        self._agent_marker.header.frame_id = "world"
        self._agent_marker.header.stamp = rospy.Time(0)
        self._agent_marker.ns = self._agent_name
        self._agent_marker.id = 0
        self._agent_marker.action = Marker.ADD
        self._agent_marker.scale.x = scale
        self._agent_marker.scale.y = scale
        self._agent_marker.scale.z = scale
        self._agent_marker.color.a = 1.0
        self._agent_marker.color.r = rgb[0]
        self._agent_marker.color.g = rgb[1]
        self._agent_marker.color.b = rgb[2]

        self._agent_marker.type = Marker.MESH_RESOURCE
        self._agent_marker.mesh_resource = "package://decentralized_ergodic/mesh/quad_base.stl"

        self._path_track.header.frame_id = "world"
        self._path_track.header.stamp = rospy.Time(0)
        self._path_track.ns = self._agent_name
        self._path_track.id = 1
        self._path_track.action = Marker.ADD
        self._path_track.points = []
        self._path_track.scale.x = 0.01
        self._path_track.color.a = 1.0
        self._path_track.color.r = rgb[0]
        self._path_track.color.g = rgb[1]
        self._path_track.color.b = rgb[2]
        self._path_track.type = Marker.LINE_STRIP

        self._sensor_range.header.frame_id = "world"
        self._sensor_range.header.stamp = rospy.Time(0)
        self._sensor_range.ns = self._agent_name
        self._sensor_range.id = 1
        self._sensor_range.action = Marker.ADD
        self._sensor_range.points = []
        self._sensor_range.scale.x = 0.4
        self._sensor_range.scale.y = 0.4
        self._sensor_range.scale.z = 0.01
        self._sensor_range.color.a = 0.2
        self._sensor_range.color.r = rgb[0]
        self._sensor_range.color.g = rgb[1]
        self._sensor_range.color.b = rgb[2]

        self.sensor_region.type = Marker.CYLINDER
