#!/usr/bin/env python


import rospy
import roslaunch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_agents', type=int, help='agent number', default=1)
args, unknown = parser.parse_known_args()



if __name__ == '__main__':
    package = 'decentralized_ergodic'
    executable = 'create_agent.py'

    nodes       = []
    processes   = []
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()


    num_agents = args.num_agents
    for i in range(num_agents):
        node_name = 'agent{}'.format(i)
        args = '{}'.format(node_name)
        if i == 0:
            nodes.append(
                roslaunch.core.Node(package=package, node_type=executable, name=node_name, args=args, output="screen")
            )
        else:
            nodes.append(
                roslaunch.core.Node(package=package, node_type=executable, name=node_name, args=args)
            )
        processes.append(launch.launch(nodes[-1]))

    rospy.init_node('test', anonymous=True)
    rospy.spin()
