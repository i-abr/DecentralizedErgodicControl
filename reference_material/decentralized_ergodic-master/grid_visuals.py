import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Point
import numpy as np
from tf.transformations import quaternion_from_euler
from grid_map_msgs.msg import GridMap, GridMapInfo

from basis import Basis


class TargetDistributionVisual(object):

    def __init__(self, filePath):

        self.grid_map = GridMap()
        self.grid_map_pub = rospy.Publisher('target_distribution', GridMap, queue_size=10)

        self.data = np.loadtxt(filePath)

        xlim = [[0.,1.0],[0.,1.0]]
        coef = np.array([2]*2)
        self.k_list = []
        for i in range(coef[0]+1):
            for j in range(coef[1]+1):
                self.k_list.append([i,j])
        self.basis = Basis(xlim, coef, self.k_list)

        self.grid_map.info.header.frame_id = "world"
        self.grid_map.info.header.stamp = rospy.Time(0)
        self.grid_map.info.resolution = 0.1
        self.grid_map.layers.append("test_layer")

        self.grid_map.data.layout.dim[0].size = 10
        self.grid_map.data.layout.dim[1].size = 10
