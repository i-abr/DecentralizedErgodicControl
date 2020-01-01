#!/bin/bash

rostopic pub -1 "demo_mode" std_msgs/String "tanvas"

rosrun tanvas_comms client_demo

