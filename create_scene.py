#!/usr/bin/env python3


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
    agent_names = []
    for i in range(num_agents):
        node_name = 'agent{}'.format(i)
        agent_names.append(node_name)
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

    # create a process for visualizing swarm
    robot_rendering_node = roslaunch.core.Node(package=package, args=str(agent_names)[1:-1].replace(",",""),
                    node_type='create_agent_rendering.py', name='robot_rendering', output="screen")
    processes.append(launch.launch(robot_rendering_node))

    env_rendering_node = roslaunch.core.Node(package=package, node_type='create_env_rendering.py',
                        name='env_rendering',output="screen", args=str(6))
    processes.append(launch.launch(env_rendering_node))

    rospy.init_node('launch_node', anonymous=True)
    rospy.spin()
