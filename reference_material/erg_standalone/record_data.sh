#!/bin/bash

curr_date=`date '+%m%d%Y-%H-%M-%S'`
fname="fx3_data_$curr_date.bag"

rosbag record /agent_id move_base_simple/goal /gps_fix /grid_pts /input /pose_override /indoor_check /controller_check /manage_controller /agent_loc /HVT_pose /IED_pose /received_floats /send_floats -O $fname

