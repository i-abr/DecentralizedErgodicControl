#!/usr/bin/env python

import rospy
import numpy as np
from quadcopter_visual import QuadcopterVisual
from target_visual import TargetVisual
from terrain import CityTerrain
from grid_visuals import TargetDistributionVisual
def main():
    rospy.init_node('quadcopter_visual')
    filePath = '/home/anon/ros_ws/src/decentralized_ergodic/' + 'multi_target_local2/'# 'target_evasion_data/'
    quads = []
    targets = []
    target_beliefs = []
    # target_distribution = TargetDistributionVisual(filePath + 'agent0/phik_data.csv')
    no_agents = 3
    no_targets = 4
    for i in range(no_agents):
        quads.append(
                QuadcopterVisual(filePath + 'agent{}/robot_state_data.npy'.format(i), no=i)
        )
        # for target_no in range(no_targets):
        #     data = np.load(filePath + 'agent{}/target{}_mean_data.npy'.format(i, target_no))
        #     target_beliefs.append(
        #             TargetVisual(data=data, no=i, estimate=True)
        #     )
    for i in range(no_targets):
        targets.append(
                TargetVisual(filePath=filePath + 'target{}/target_state_data.npy'.format(i), no=i)
        )



    terrain = CityTerrain(filePath + 'city_blocks.csv')
    loop_rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        for quad in quads:
            quad.stepAndUpdateMarker()
        for target_belief in target_beliefs:
            target_belief.stepAndUpdateMarker()
        for target in targets:
            target.stepAndUpdateMarker()
        terrain.publishMarker()
        loop_rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
