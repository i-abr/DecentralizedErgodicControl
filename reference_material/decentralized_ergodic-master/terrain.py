import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Point
import numpy as np


class CityTerrain(object):


    def __init__(self, filePath):

        self.blocks = MarkerArray()
        self.layout = np.loadtxt(filePath)
        for i,layout in enumerate(self.layout):
            self.blocks.markers.append(self._build_marker(layout,i))
        self.marker_pub = rospy.Publisher('city_block', MarkerArray, queue_size=10)

    def publishMarker(self):
        self.marker_pub.publish(self.blocks)


    def _build_marker(self, layout, i):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time(0)
        marker.ns = "buildings"
        marker.id = i
        marker.action = Marker.ADD
        marker.pose.position.x = layout[0]
        marker.pose.position.y = layout[1]
        marker.pose.position.z = 0.25
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.0
        marker.scale.x = 2*layout[2]
        marker.scale.y = 2*layout[2]
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.type = Marker.CUBE
        return marker
