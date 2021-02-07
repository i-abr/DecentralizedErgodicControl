#!/usr/bin/env python3

import os
import argparse
import ast
import rospy

from env import Map

parser = argparse.ArgumentParser()
parser.add_argument('num_buildings', type=int, help='num buildings in map', default=4)
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    map = Map(args.num_buildings)
    try:
        map.run()
    except rospy.ROSInterruptException:
        pass
