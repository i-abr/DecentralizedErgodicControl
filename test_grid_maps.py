#!/usr/bin/env python

import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import tf

if __name__ == '__main__':

    rospy.init_node('dist_test')

    gridmap = GridMap()
    x,y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
    samples = np.c_[np.ravel(x, order="F"), np.ravel(y,order="F")]
    grid_val = np.exp(-10. * np.sum((samples-np.array([0.2,0.2]))**2, axis=1))*10 \
                    + np.exp(-10. * np.sum((samples-np.array([0.8,0.5]))**2, axis=1))*10


    arr = Float32MultiArray()

    arr.data = grid_val[::-1]
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

    print gridmap

    map_pub = rospy.Publisher("grid", GridMap, queue_size=1)

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        map_pub.publish(gridmap)
        rate.sleep()
