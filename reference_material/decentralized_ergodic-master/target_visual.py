import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Point
import numpy as np
from tf.transformations import quaternion_from_euler


class TargetVisual(object):

    def __init__(self, filePath=None, data=None, no=None, estimate=False):

        self.no = no
        self.estimate = estimate
        self.target_visual = Marker()
        self.line_trail = Marker()
        self.target_name = 'target'
        if no is not None:
            self.target_name = self.target_name + '{}'.format(self.no)
        if estimate is True:
            self.target_name += '_belief'
        self.target_visual_pub = rospy.Publisher(self.target_name + '_visual', Marker, queue_size=10)
        self.line_pub = rospy.Publisher(self.target_name + 'visual_line', Marker, queue_size=10)

        if filePath is None:
            self.data = data
        else:
            self.data = np.load(filePath)

        self.data_length = len(self.data)
        self.idx = 0
        self._build_quadcopter_visual()

    def _build_quadcopter_visual(self):

        scale = 0.05
        self.target_visual.header.frame_id = "world"
        self.target_visual.header.stamp = rospy.Time(0)
        self.target_visual.ns = "target"
        self.target_visual.id = 0
        self.target_visual.action = Marker.ADD
        self.target_visual.pose.position.x = self.data[0,0]
        self.target_visual.pose.position.y = self.data[0,1]
        self.target_visual.pose.position.z = 0.0
        quaternion = quaternion_from_euler(0.,0.,0.)
        self.target_visual.pose.orientation.x = quaternion[0]
        self.target_visual.pose.orientation.y = quaternion[1]
        self.target_visual.pose.orientation.z = quaternion[2]
        self.target_visual.pose.orientation.w = quaternion[3]
        self.target_visual.scale.x = scale
        self.target_visual.scale.y = scale
        self.target_visual.scale.z = scale
        if not self.estimate:
            self.target_visual.color.a = 1.0
            self.target_visual.color.r = 1.0
            self.target_visual.color.g = 0.0
            self.target_visual.color.b = 0.0
        else:
            self.target_visual.color.a = 0.4
            self.target_visual.color.r = 1.0
            self.target_visual.color.g = 0.0
            self.target_visual.color.b = 0.0

        if self.no == 0:
            self.target_visual.type = Marker.CUBE
        else:
            self.target_visual.type = Marker.CUBE
        # self.target_visual.type = Marker.MESH_RESOURCE
        # self.target_visual.mesh_resource = "package://decentralized_ergodic/mesh/FutureOffRoadVehicle.stl"


        self.line_trail.header.frame_id = "world"
        self.line_trail.header.stamp = rospy.Time(0)
        self.line_trail.ns = "target"
        self.line_trail.id = 1
        self.line_trail.action = Marker.ADD
        self.line_trail.points = []
        self.line_trail.scale.x = 0.01
        self.line_trail.color.a = 1.0
        self.line_trail.color.r = np.random.uniform(0,1)
        self.line_trail.color.g = np.random.uniform(0,1)
        self.line_trail.color.b = np.random.uniform(0,1)
        self.line_trail.type = Marker.LINE_STRIP


    def stepAndUpdateMarker(self):
        self.idx += 1
        if self.idx >= self.data_length:
            self.idx = 0
            del self.line_trail.points[:]
        self.target_visual.pose.position.x = self.data[self.idx,0]
        self.target_visual.pose.position.y = self.data[self.idx,1]

        # if self.no == 0:
        #     quaternion = quaternion_from_euler(0.0, 0.0, self.data[self.idx,2])
        #     self.target_visual.pose.orientation.x = quaternion[0]
        #     self.target_visual.pose.orientation.y = quaternion[1]
        #     self.target_visual.pose.orientation.z = quaternion[2]
        #     self.target_visual.pose.orientation.w = quaternion[3]

        self.line_trail.points.append(
                Point(self.data[self.idx,0], self.data[self.idx,1], 0.0)
        )
        if len(self.line_trail.points) > 100:
            del self.line_trail.points[0]
        self.line_pub.publish(self.line_trail)
        self.target_visual_pub.publish(self.target_visual)
