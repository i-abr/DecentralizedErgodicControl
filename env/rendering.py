import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans

class Visual(object):

    def __init__(self):

        self._env_markers = MarkerArray()

        self._markers = [
            Marker() for i in range(len(self.objects))
        ]

        self._marker_pub = rospy.Publisher('env/visual', MarkerArray, queue_size=1)
        self.__build_rendering()

    def update_rendering(self):
        self._env_markers.markers = self._markers
        self._marker_pub.publish(self._env_markers)

    def __build_rendering(self):

        i = 0
        for obj, marker in zip(self.objects, self._markers):

            rgb = np.random.uniform(0,1, size=(3,))
            scale = 0.1
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time(0)
            marker.ns = 'building_' + str(i)
            marker.id = 0
            marker.action = Marker.ADD
            marker.scale.x = obj.l * 10.
            marker.scale.y = obj.w * 10.
            marker.scale.z = np.random.uniform(0.01, 0.15) * 10.
            marker.color.a = 1.0
            marker.color.r = rgb[0]
            marker.color.g = rgb[1]
            marker.color.b = rgb[2]
            marker.pose.position.x = obj._p[0] * 10.
            marker.pose.position.y = obj._p[1] * 10.
            marker.pose.position.z = marker.scale.z/2.0

            marker.pose.orientation.x = obj._quat[0]
            marker.pose.orientation.y = obj._quat[1]
            marker.pose.orientation.z = obj._quat[2]
            marker.pose.orientation.w = obj._quat[3]
            marker.type = Marker.CUBE

            i += 1
