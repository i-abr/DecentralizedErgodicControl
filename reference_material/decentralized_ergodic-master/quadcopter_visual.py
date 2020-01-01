import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Point
import numpy as np
from tf.transformations import quaternion_from_euler

class QuadcopterVisual(object):

    def __init__(self, filePath, no=None):

        self.quadcopter_visual = Marker()
        self.line_trail = Marker()
        self.sensor_region = Marker()
        if no is None:
            self.quadcopter_visual_pub = rospy.Publisher('quadcopter_visual', Marker, queue_size=10)
            self.line_pub = rospy.Publisher('quadcopter_visual_line', Marker, queue_size=10)
            self.sensor_region_pub = rospy.Publisher('quadcopter_sensor_region', Marker, queue_size=10)
        else:
            self.quadcopter_visual_pub = rospy.Publisher('quadcopter_visual{}'.format(no), Marker, queue_size=10)
            self.line_pub = rospy.Publisher('quadcopter_visual{}_line'.format(no), Marker, queue_size=10)
            self.sensor_region_pub = rospy.Publisher('quadcopter_sensor{}_region'.format(no), Marker, queue_size=10)

        self.data = np.load(filePath)
        self.data_length = len(self.data)
        self.idx = 0
        self._build_quadcopter_visual()

    def _build_quadcopter_visual(self):

        scale = 0.1
        self.quadcopter_visual.header.frame_id = "world"
        self.quadcopter_visual.header.stamp = rospy.Time(0)
        self.quadcopter_visual.ns = "quadcopter"
        self.quadcopter_visual.id = 0
        self.quadcopter_visual.action = Marker.ADD
        self.quadcopter_visual.pose.position.x = self.data[0,0]
        self.quadcopter_visual.pose.position.y = self.data[0,1]
        self.quadcopter_visual.pose.position.z = self.data[0,2]
        self.quadcopter_visual.pose.orientation.x = self.data[0,4]
        self.quadcopter_visual.pose.orientation.y = self.data[0,5]
        self.quadcopter_visual.pose.orientation.z = self.data[0,3]
        self.quadcopter_visual.pose.orientation.w = 0.0
        self.quadcopter_visual.scale.x = scale
        self.quadcopter_visual.scale.y = scale
        self.quadcopter_visual.scale.z = scale
        self.quadcopter_visual.color.a = 1.0
        self.quadcopter_visual.color.r = np.random.uniform(0,1)
        self.quadcopter_visual.color.g = np.random.uniform(0,1)
        self.quadcopter_visual.color.b = np.random.uniform(0,1)

        self.quadcopter_visual.type = Marker.MESH_RESOURCE
        self.quadcopter_visual.mesh_resource = "package://decentralized_ergodic/mesh/quad_base.stl"


        self.line_trail.header.frame_id = "world"
        self.line_trail.header.stamp = rospy.Time(0)
        self.line_trail.ns = "quadcopter"
        self.line_trail.id = 1
        self.line_trail.action = Marker.ADD
        self.line_trail.points = []
        self.line_trail.scale.x = 0.01
        self.line_trail.color.a = 1.0
        self.line_trail.color.r = np.random.uniform(0,1)
        self.line_trail.color.g = np.random.uniform(0,1)
        self.line_trail.color.b = np.random.uniform(0,1)
        self.line_trail.type = Marker.LINE_STRIP

        self.sensor_region.header.frame_id = "world"
        self.sensor_region.header.stamp = rospy.Time(0)
        self.sensor_region.ns = "quadcopter"
        self.sensor_region.id = 1
        self.sensor_region.action = Marker.ADD
        self.sensor_region.points = []
        self.sensor_region.scale.x = 0.4
        self.sensor_region.scale.y = 0.4
        self.sensor_region.scale.z = 0.01
        self.sensor_region.color.a = 0.2
        self.sensor_region.color.r = self.quadcopter_visual.color.r
        self.sensor_region.color.g = self.quadcopter_visual.color.g
        self.sensor_region.color.b = self.quadcopter_visual.color.b

        self.sensor_region.type = Marker.CYLINDER



    def stepAndUpdateMarker(self):
        self.idx += 1
        if self.idx >= self.data_length:
            self.idx = 0
            del self.line_trail.points[:]
        yaw, pitch, roll = self.data[self.idx,3], self.data[self.idx,4], self.data[self.idx, 5]
        x,y,z = self.data[self.idx, 0], self.data[self.idx, 1], self.data[self.idx,2]*0.1
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        self.quadcopter_visual.pose.position.x = x
        self.quadcopter_visual.pose.position.y = y
        self.quadcopter_visual.pose.position.z = z
        self.quadcopter_visual.pose.orientation.x = quaternion[0]
        self.quadcopter_visual.pose.orientation.y = quaternion[1]
        self.quadcopter_visual.pose.orientation.z = quaternion[2]
        self.quadcopter_visual.pose.orientation.w = quaternion[3]

        self.sensor_region.pose.position.x = x
        self.sensor_region.pose.position.y = y
        self.sensor_region.pose.position.z = 0.0

        self.line_trail.points.append(
                Point(x, y, z)
        )
        if len(self.line_trail.points) > 100:
            del self.line_trail.points[0]
        self.line_pub.publish(self.line_trail)
        self.sensor_region_pub.publish(self.sensor_region)
        self.quadcopter_visual_pub.publish(self.quadcopter_visual)
