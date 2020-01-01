import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys

def rotation_matrix(angles):
    ct = math.cos(angles[0])
    cp = math.cos(angles[1])
    cg = math.cos(angles[2])
    st = math.sin(angles[0])
    sp = math.sin(angles[1])
    sg = math.sin(angles[2])
    R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
    R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


fig = plt.figure()
ax = Axes3D.Axes3D(fig)

quads = {0:{},1:{}, 2:{}}
no_agents = 3
for agent_no in range(no_agents):
    quads[agent_no]['l1'], = ax.plot([],[],[], color='black', linewidth=3, antialiased=True)
    quads[agent_no]['l2'], = ax.plot([],[],[], color='black', linewidth=3, antialiased=True)
    quads[agent_no]['height'], = ax.plot([],[],[], color='red', ls='--',linewidth=1, antialiased=True)
X,Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))
collective_estimate = np.load('collective_gp_data_at_index_800.npy')
eta = np.max(collective_estimate)
surf_fig = ax.plot_surface(X,Y,0.5*collective_estimate/eta, rstride=1, cstride=1, cmap='Greys', linewidth=0.01)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
surf_fig.set_zorder(21)
L = 0.05
frame_points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
data = [np.loadtxt('agent{}/robot_state_data.csv'.format(i)) for i in range(no_agents)]
# for i in range(len(data[0])):
i = 99
print(i)
for agent_no in range(no_agents):
    yaw, pitch, roll = data[agent_no][i,3:6]
    angles = [roll, pitch, yaw]
    position = data[agent_no][i,0:3]
    position[-1] *= 0.5
    frame_points_trans = rotation_matrix(angles).dot(frame_points)
    frame_points_trans[0,:] += position[0]
    frame_points_trans[1,:] += position[1]
    frame_points_trans[2,:] += position[2]
    quads[agent_no]['l1'].set_data(frame_points_trans[0,0:2],frame_points_trans[1,0:2])
    quads[agent_no]['l1'].set_3d_properties(frame_points_trans[2,0:2])
    quads[agent_no]['l1'].set_zorder(20)

    quads[agent_no]['l2'].set_data(frame_points_trans[0,2:4],frame_points_trans[1,2:4])
    quads[agent_no]['l2'].set_3d_properties(frame_points_trans[2,2:4])
    quads[agent_no]['l2'].set_zorder(20)

    quads[agent_no]['height'].set_data([position[0]]*2, [position[1]]*2)
    quads[agent_no]['height'].set_3d_properties([0, position[2]])
    # quads[agent_no]['height'].set_zorder(20)



plt.show()
# plt.pause(0.01)
