#!/usr/bin/env python3

import os
import argparse
import rospy

from quad_agent import Agent


parser = argparse.ArgumentParser()
parser.add_argument('agent_name', type=str, help='agent name', default="thaddius")
args, unknown = parser.parse_known_args()

if __name__ == '__main__':

    agent = Agent(args.agent_name)
    try:
        agent.run()
    except rospy.ROSInterruptException:
        pass
