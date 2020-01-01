import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
# from gps_common.msg import GPSFix
from tanvas_comms.msg import input_array

pos_data = []
goal_data = []
tdist_data = []

# fname = 'fx3_data_11292019-14-13-38.bag'
fname = 'fx3_data_11292019-14-36-37.bag'
bag = rosbag.Bag(fname)

for topic, msg, t in bag.read_messages(['move_base_simple/goal', '/gps_fix', '/input']):
    if topic == 'move_base_simple/goal':
        goal_data.append([msg.pose.position.x, msg.pose.position.y])
    elif topic == '/gps_fix':
        pos_data.append([msg.latitude, msg.longitude])
    elif topic =='/input':
        if msg.datalen !=0:
            tdist_data.append([msg.xinput,msg.yinput])

bag.close()

transformed_t_inputs = []
for t_inputs in tdist_data:
    print(t_inputs)
    xin = np.array(list(t_inputs[0]))/30.
    yin = np.array(list(t_inputs[1]))/30.
    #
    transformed_t_inputs.append([xin,yin])

print(transformed_t_inputs)

print(len(pos_data), len(goal_data),len( tdist_data))

def _coord_to_dist(coord):
    x1 = 31.1379680
    y1 = -89.0647412

    x2 = 31.137823
    y2 = -89.064585
    x = (coord[0]-x1)/(x2-x1)
    y = (coord[1]-y1)/(y2-y1)
    return [x, y]



pos_data = np.array([_coord_to_dist(pos_data[i]) for i in range(len(pos_data))])
tdist_data = np.array(tdist_data)
# goal_data = np.array(goal_data)
plt.plot(pos_data[:,0], pos_data[:,1],'-b.')
plt.plot(transformed_t_inputs[0][0], transformed_t_inputs[0][1], '.r')
plt.plot(transformed_t_inputs[1][0], transformed_t_inputs[1][1], '.m')
plt.plot(transformed_t_inputs[2][0], transformed_t_inputs[2][1], '.g')

plt.show()


