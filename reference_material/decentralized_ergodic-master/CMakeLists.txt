cmake_minimum_required(VERSION 2.8.3)
project(decentralized_ergodic)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  visualization_msgs
  genmsg
  message_generation
  cmake_modules
  tf
)

catkin_package(
    CATKIN_DEPENDS rospy roscpp std_msgs geometry_msgs genmsg visualization_msgs message_runtime
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)


