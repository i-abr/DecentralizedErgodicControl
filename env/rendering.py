import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans

class Visual(object):

    def __init__(self):

        
