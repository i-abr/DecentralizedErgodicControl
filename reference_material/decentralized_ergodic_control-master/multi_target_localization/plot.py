import numpy as np
import matplotlib.pyplot as plt

filepath = 'target_test_data_for_ros/'
target = np.loadtxt(filepath + 'target{}/target_state_data.csv'.format(0))
for i in range(3):
    target_mean = np.loadtxt(filepath + 'agent{}/target_mean_data.csv'.format(i))
    robot_state = np.loadtxt(filepath + 'agent{}/robot_state_data.csv'.format(i))
    plt.figure(1)
    plt.plot(target_mean)
    plt.plot(target_mean[:,0], target_mean[:,1])
    plt.figure(2)
    plt.plot(robot_state[:,0], robot_state[:,1])
    #plt.plot(robot_state[:,2])
plt.figure(1)
plt.plot(target[:,0:2],'k')
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.show()
