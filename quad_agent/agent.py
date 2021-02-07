import sys
sys.path.append('../')

import rospy
import tf

from d_erg_lib import DErgControl
from d_erg_lib.utils import convert_ck2dist
from .model import Model

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import tf

class Agent(Model):

    def __init__(self, agent_name):

        self.agent_name = agent_name
        rospy.init_node(agent_name)
        self._rate = rospy.Rate(10)
        Model.__init__(self)
        # Visual.__init__(self, agent_name)
        self.ctrllr = DErgControl(agent_name, Model())
        self.__br = tf.TransformBroadcaster()

        # TODO: consider moving this into the target dist class
        self._target_dist_pub = rospy.Publisher(agent_name+'/target_dist', GridMap, queue_size=1)

        gridmap = GridMap()
        arr = Float32MultiArray()

        arr.data = self.ctrllr._targ_dist.grid_vals[::-1]
        arr.layout.dim.append(MultiArrayDimension())
        arr.layout.dim.append(MultiArrayDimension())

        arr.layout.dim[0].label="column_index"
        arr.layout.dim[0].size=50
        arr.layout.dim[0].stride=50*50

        arr.layout.dim[1].label="row_index"
        arr.layout.dim[1].size=50
        arr.layout.dim[1].stride=50


        gridmap.layers.append("elevation")
        gridmap.data.append(arr)
        gridmap.info.length_x=10
        gridmap.info.length_y=10
        gridmap.info.pose.position.x=5
        gridmap.info.pose.position.y=5

        gridmap.info.header.frame_id = "world"
        gridmap.info.resolution = 0.2

        self._grid_msg = gridmap

    def step(self):
        ctrl = self.ctrllr(self.state)
        pred_path = self.ctrllr.pred_path
        super(Agent, self).step(ctrl)
        # self.update_rendering(pred_path)
        self.__br.sendTransform(
            (self.state[0]*10, self.state[1]*10, 0.),
            (0.,0.,0.,1.),
            rospy.Time.now(),
            self.agent_name,
            "world"
        )

        grid_vals = convert_ck2dist(self.ctrllr._basis, self.ctrllr._ck_mean)
        self._grid_msg.data[0].data = grid_vals[::-1]
        self._target_dist_pub.publish(self._grid_msg)

    def run(self):
        while not rospy.is_shutdown():
            self.step()
            self._rate.sleep()
