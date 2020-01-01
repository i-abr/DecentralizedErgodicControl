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

no_agents = 1
no_targets = 4
fig = plt.figure()
ax = fig.add_subplot(111)

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


X,Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))
# collective_estimate = np.load('collective_gp_data_at_index_800.npy')
# eta = np.max(collective_estimate)
# surf_fig = ax.plot_surface(X,Y,0.5*collective_estimate/eta, rstride=1, cstride=1, cmap='Greys', linewidth=0.01)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# surf_fig.set_zorder(21)
data1 = np.load('agent0_pdf_data/pdf_data_at_0.npy')

ax.contourf(X,Y, data1, zdir='z', offset=0)
# surf_fig.set_data(X,Y,data1)
L = 0.05
frame_points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
data = [np.loadtxt('agent{}/robot_state_data.csv'.format(i)) for i in range(no_agents)]
target_data = [np.loadtxt('target{}/target_state_data.csv'.format(i)) for i in range(no_targets)]
# c1 = Circle((0.5,0.5), 0.2)
# ax.add_patch(c1)
# art3d.pathpatch_2d_to_3d(c1, z=0)
pc = PatchCollection(city_patches, facecolor='g', edgecolor='g')
ax.add_collection(pc)
plt.ion()
trail = 0
for i in range(len(data[0])-1):
    if i > 10:
        trail = i-5

    ax.clear()
    data1 = np.load('agent0_pdf_data/pdf_data_at_{}.npy'.format(i))
    ax.contourf(X,Y, data1, zdir='z', offset=0, cmap="Greys")
    for agent_no in range(no_agents):
        positions = data[agent_no][trail:i+1,0:3]
        positions[:,2] *= 0.25
        ax.plot(positions[:,0], positions[:,1])
        c1 = Circle((positions[-1,0],positions[-1,1]), 0.2, alpha=0.2)
        ax.add_patch(c1)
    for target_no in range(no_targets):
        positions = target_data[target_no][trail:i+1,0:3]
        ax.plot(positions[:,0], positions[:,1])
        c1 = Circle((positions[-1,0],positions[-1,1]), 0.025, alpha=0.2)
        ax.add_patch(c1)
    ax.add_collection(pc)

    plt.pause(0.01)
