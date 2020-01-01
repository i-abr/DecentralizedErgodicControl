import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import mpl_toolkits.mplot3d.axes3d as Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
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

no_agents = 3
no_targets = 4

agent_data = {}
for agent_no in range(no_agents):
    agent_data.update({
            agent_no : {
                'state' : np.load('agent{}/robot_state_data.npy'.format(agent_no))
            }
        })
    for target_no in range(no_targets):
        agent_data[agent_no].update({
            'target{}_mean'.format(target_no) : np.load('agent{}/target{}_mean_data.npy'.format( agent_no,  target_no )),
            'target{}_covar'.format(target_no) : np.load('agent{}/target{}_covar_data.npy'.format( agent_no,  target_no )),
            target_no : np.load('agent{}/target{}_pdf_time_data.npy'.format( agent_no,  target_no ))
        })
target_data = {}
for target_no in range(no_targets):
    target_data.update({
        target_no : np.load('target{}/target_state_data.npy'.format(target_no))[0]
    })

data_len = agent_data[0][0].shape[0]
time_indx = [i for i in range(0, data_len,80)]
time_indx.append(399)
print(time_indx)
fig, ax = plt.subplots(1, len(time_indx), sharex='col', sharey='row')

X,Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))

city_patches = []
block_positions = np.loadtxt('city_blocks.csv')
xpos = block_positions[:,0]
ypos = block_positions[:,1]
zpos = np.zeros(xpos.shape)
dx = np.ones(xpos.shape) * block_positions[:,2] * 1.5
dy = np.ones(xpos.shape) * block_positions[:,2] * 1.5
dz = np.ones(xpos.shape) * 0.75
for i in range(len(xpos)):
    city_patches.append(
            Rectangle((xpos[i], ypos[i]), dx[i], dy[i])
    )

pc = PatchCollection(city_patches, facecolor='g', edgecolor='g')

for i in range(len(time_indx)):
    for agent_no in range(no_agents):
        _data = None
        for target_no in range(no_targets):
            if _data is None:
                _data = agent_data[agent_no][target_no][time_indx[i]].ravel()
                # _data /= np.sum(_data)
            else:
                _temp = agent_data[agent_no][target_no][time_indx[i]].ravel()
                # _temp /= np.sum(_temp)
                _data += _temp
                # _data = np.amax( np.c_[_data.ravel(), _temp.ravel() ] ,axis=1)
        _data /= np.sum(_data)
        ax[i].contourf(X,Y, _data.reshape(X.shape), cmap="Greys")
        pc = PatchCollection(city_patches, facecolor='k', edgecolor='r')
        ax[i].add_collection(pc)
        for target_no in range(no_targets):
            ax[i].plot(target_data[target_no][0], target_data[target_no][1],'rx',lw=10)
        trajectory = agent_data[agent_no]['state']
        if i != 50:
            ax[i].plot(trajectory[0:time_indx[i],0], trajectory[0:time_indx[i],1])
        else:
            ax[i].plot(trajectory[time_indx[i-1]:time_indx[i],0], trajectory[time_indx[i-1]:time_indx[i],1])
        ax[i].add_patch(Circle( (trajectory[time_indx[i],0], trajectory[time_indx[i],1]), 0.18, alpha=0.2 ))
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
# for i in range(0,agent0_data[0].shape[0]):
#     data_for_plt = None
#     for j in range(len(agent0_data)):
#         # if data_for_plt is None:
#         #     data_for_plt = agent0_data[j][i]/(np.sum(agent0_data[j][i]))
#         # else:
#         #     _data =  agent0_data[j][i].ravel()
#         #     _data /= np.sum(_data)
#         #     print()
#         #
#         #     data_for_plt = np.amax( np.c_[ data_for_plt.ravel(), _data],axis=1)
#         _data = agent0_data[j][i]
#         ax[j, i].imshow(_data, cmap='Greys')

fig, ax = plt.subplots(1, 4, sharex='col', sharey='row')


time_vec = [i*0.05 for i in range(data_len)]
for target_no in range(no_targets):
    plt.figure(target_no)
    for agent_no in range(no_agents):
        mean_traj = agent_data[agent_no]['target{}_mean'.format(target_no)]
        var = agent_data[agent_no]['target{}_covar'.format(target_no)]
        mean_p = np.empty(mean_traj.shape)
        mean_m = np.empty(mean_traj.shape)
        for i,mean_val in enumerate(mean_traj):
            mean_p[i] = mean_traj[i] + 0.1*np.sqrt(np.diag(var[i]))
            mean_m[i] = mean_traj[i] - 0.1*np.sqrt(np.diag(var[i]))
        ax[target_no].plot(time_vec, mean_traj)
        ax[target_no].fill_between( time_vec, mean_m[:,0], mean_p[:,0], alpha=0.2, color='c')
        ax[target_no].fill_between( time_vec, mean_m[:,1], mean_p[:,1], alpha=0.2, color='b')
        ax[target_no].set_xlim(0,20)
        ax[target_no].set_ylim(0,1)
plt.plot()


plt.show()






# collective_estimate = np.load('collective_gp_data_at_index_800.npy')
# eta = np.max(collective_estimate)
# surf_fig = ax.plot_surface(X,Y,0.5*collective_estimate/eta, rstride=1, cstride=1, cmap='Greys', linewidth=0.01)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# surf_fig.set_zorder(21)
# data1 = np.load('agent0_pdf_data/pdf_data_at_0.npy')
#
# ax.contourf(X,Y, data1, zdir='z', offset=0)
# # surf_fig.set_data(X,Y,data1)
# L = 0.05
# frame_points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
# data = [np.loadtxt('agent{}/robot_state_data.csv'.format(i)) for i in range(no_agents)]
# target_data = [np.loadtxt('target{}/target_state_data.csv'.format(i)) for i in range(no_targets)]
# # c1 = Circle((0.5,0.5), 0.2)
# # ax.add_patch(c1)
# # art3d.pathpatch_2d_to_3d(c1, z=0)
# pc = PatchCollection(city_patches, facecolor='g', edgecolor='g')
# ax.add_collection(pc)
# plt.ion()
# trail = 0
# for i in range(len(data[0])-1):
#     if i > 10:
#         trail = i-5
#
#     ax.clear()
#     data1 = np.load('agent0_pdf_data/pdf_data_at_{}.npy'.format(i))
#     ax.contourf(X,Y, data1, zdir='z', offset=0, cmap="Greys")
#     for agent_no in range(no_agents):
#         positions = data[agent_no][trail:i+1,0:3]
#         positions[:,2] *= 0.25
#         ax.plot(positions[:,0], positions[:,1])
#         c1 = Circle((positions[-1,0],positions[-1,1]), 0.2, alpha=0.2)
#         ax.add_patch(c1)
#     for target_no in range(no_targets):
#         positions = target_data[target_no][trail:i+1,0:3]
#         ax.plot(positions[:,0], positions[:,1])
#         c1 = Circle((positions[-1,0],positions[-1,1]), 0.025, alpha=0.2)
#         ax.add_patch(c1)
#     ax.add_collection(pc)
#
#     plt.pause(0.01)
