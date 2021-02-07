#!/usr/bin/env python3

import os
import argparse
import ast
import rospy

from quad_agent import QuadVisual


parser = argparse.ArgumentParser()
parser.add_argument('agent_names', type=str, nargs='+')
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    agent_rendering = QuadVisual(args.agent_names[:-2])
    try:
        agent_rendering.run()
    except rospy.ROSInterruptException:
        pass
