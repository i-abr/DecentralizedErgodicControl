import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
from single_ergodic_quad.agent import QuadcopterAgent
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

from IPython import embed

agent = QuadcopterAgent()
target_p = agent.objective.target_distribution
target = agent.objective.target_distribution.targets[0]
floor_plan = agent.objective.floor_plan


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
ax = Axes3D(fig)
ax.set_zlim3d(0,0.2)

ax.grid(False)
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

scale = np.diag([1,1,0.5, 1])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def proj():
    return np.dot(Axes3D.get_proj(ax) , scale)

ax.get_proj = proj

quads = {0:{},1:{},2:{},3:{}}

num_agents = 4
print(ax.plot([],[],[], color='black', linewidth=3, antialiased=True))
for agent_no in range(num_agents):
    quads[agent_no]['l1'], = ax.plot([],[],[], color='black', linewidth=1, antialiased=True)
    quads[agent_no]['l2'], = ax.plot([],[],[], color='black', linewidth=1, antialiased=True)
    quads[agent_no]['height'], = ax.plot([],[],[], color='red', ls='--',linewidth=1, antialiased=True)
    quads[agent_no]['trajectory'], = ax.plot([],[],[])
    quads[agent_no]['trajectory'].set_zorder(5)
# contour_fig, = ax.contourf([],[], [], zdir='z',cmap='Greys')

# embed()
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# surf_fig.set_zorder(21)


floor_plan_x = [0., 0., 0.25, 0.25, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.]
floor_plan_y = [0., 0.5, 0.5, 0.75, 0.75, 0., 0., 0.5, 0.5, 0.25, 0.25, 0., 0.]
floor_plan_z = [0. for i in range(len(floor_plan_x))]
L = 0.03
frame_points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T

path = 'time_varying_target_with_obstacles_data/'

data = [np.load(path + 'agent{}.npy'.format(i)) for i in range(num_agents)]
particle_data = [np.load(path + 'agent{}_particles.npy'.format(i)) for i in range(num_agents)]

particle_x = {'x': particle_data[0][0]}
scat = ax.scatter(particle_x['x'][:,0],particle_x['x'][:,1])

from matplotlib import animation

height_scale = 0.2
target_path = np.array([
    target.trajectory(i*0.05)
    for i in range(len(data[0]))
])

# for i in range(len(data[0])):
# for i in range(250, 290):
def animate(i):
    ax.clear()
    floor_plan.plot_ground(ax)
    # target_p.update_phik(data[0][i], i*0.05)
    # target_p.plot()
    # contour_fig.remove()
    # del contour_fig
    # contour_fig = ax.contourf(X,Y, evalu, zdir='z', offset=0.0, cmap='Greys')
    # cf.set_zorder(1)
    # contour_fig(X,Y,evalu)
    ax.plot(target_path[:i,0], target_path[:i,1], 'k', lw=2, zorder=0)

    for agent_no in range(num_agents):

        ax.scatter(particle_data[agent_no][i][:,0], particle_data[agent_no][i][:,1], alpha=0.5,zorder=-agent_no)


        yaw, pitch, roll = data[agent_no][i,3:6]
        angles = [roll, pitch, yaw]
        position = data[agent_no][i,0:3]
        position[-1] *= 0.5
        frame_points_trans = rotation_matrix(angles).dot(frame_points)
        frame_points_trans[0,:] += position[0]
        frame_points_trans[1,:] += position[1]
        frame_points_trans[2,:] += height_scale*position[2]

        ax.plot(frame_points_trans[0,0:2],frame_points_trans[1,0:2],frame_points_trans[2,0:2], color='black', linewidth=1, antialiased=True)
        ax.plot(frame_points_trans[0,2:4],frame_points_trans[1,2:4],frame_points_trans[2,2:4], color='black', linewidth=1, antialiased=True)
        ax.plot([position[0]]*2, [position[1]]*2,[0, height_scale*position[2]], color='red', ls='--',linewidth=1, antialiased=True,)

        # quads[agent_no]['l1'].set_data(frame_points_trans[0,0:2],frame_points_trans[1,0:2])
        # quads[agent_no]['l1'].set_3d_properties(frame_points_trans[2,0:2])
        # quads[agent_no]['l1'].set_zorder(3)
        #
        # quads[agent_no]['l2'].set_data(frame_points_trans[0,2:4],frame_points_trans[1,2:4])
        # quads[agent_no]['l2'].set_3d_properties(frame_points_trans[2,2:4])
        # quads[agent_no]['l2'].set_zorder(3)
        #
        # quads[agent_no]['height'].set_data([position[0]]*2, [position[1]]*2)
        # quads[agent_no]['height'].set_3d_properties([0, height_scale*position[2]])
        #
        # quads[agent_no]['height'].set_zorder(4)
        if i > 50:
            # quads[agent_no]['trajectory'].set_data(data[agent_no][i-50:i,0], data[agent_no][i-50:i,1])
            # quads[agent_no]['trajectory'].set_3d_properties(0*data[agent_no][i-50:i,2])
            # quads[agent_no]['trajectory'].set_zorder(6)
            ax.plot(data[agent_no][i-50:i,0], data[agent_no][i-50:i,1],0*data[agent_no][i-50:i,2])

        else:
            # quads[agent_no]['trajectory'].set_data(data[agent_no][0:i,0], data[agent_no][0:i,1])
            # quads[agent_no]['trajectory'].set_3d_properties(0*data[agent_no][0:i,2])
            ax.plot(data[agent_no][0:i,0], data[agent_no][0:i,1],0*data[agent_no][0:i,2])
            # quads[agent_no]['trajectory'].set_zorder(6)

        range_sensor = Circle((data[agent_no][i,0],data[agent_no][i,1]) , 0.15, color='b', alpha=0.2)
        ax.add_patch(range_sensor)
        art3d.pathpatch_2d_to_3d(range_sensor, z=0, zdir='z')

    target_artist = Circle((target_path[i,0], target_path[i,1]) , 0.02, color='k', zorder=100+agent_no)
    ax.add_patch(target_artist)
    art3d.pathpatch_2d_to_3d(target_artist, z=0, zdir='z')

    ax.set_zlim3d(0,0.2)
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    # quads[agent_no]['trajectory'].set_zorder(5)
    plt.pause(0.01)
    # plt.show()
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=10)

anim.save(path + 'target_localization.mp4', fps=30, dpi=300)
plt.show()
