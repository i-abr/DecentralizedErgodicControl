import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
from single_ergodic_quad.agent import QuadcopterAgent

from IPython import embed

agent = QuadcopterAgent()
target_p = agent.objective.target_distribution
floor_plan = agent.objective.floor_plan
X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
samples = np.c_[X.ravel(), Y.ravel()]
def eval_time_varying_distr(time):
    target_p.update_phik(time)
    evals = np.array([target_p(sample) for sample in samples])
    evals = evals.reshape(X.shape)
    return evals


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

quads = {0:{},1:{},2:{},3:{},4:{}}

num_agents = 5
print(ax.plot([],[],[], color='black', linewidth=3, antialiased=True))
for agent_no in range(num_agents):
    quads[agent_no]['l1'], = ax.plot([],[],[], color='black', linewidth=1, antialiased=True)
    quads[agent_no]['l2'], = ax.plot([],[],[], color='black', linewidth=1, antialiased=True)
    quads[agent_no]['height'], = ax.plot([],[],[], color='red', ls='--',linewidth=1, antialiased=True)
    quads[agent_no]['trajectory'], = ax.plot([],[],[])
    quads[agent_no]['trajectory'].set_zorder(5)
evalu = eval_time_varying_distr(0.0)
contour_fig = ax.contourf(X,Y,evalu, zdir='z', offset=0.0, cmap='Greys',zorder=0)
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
# data = [np.load('trial1/agent{}.npy'.format(i)) for i in range(num_agents)]

data = [np.load('obstacle_trial1/agent{}.npy'.format(i)) for i in range(num_agents)]


from matplotlib import animation

height_scale = 0.2*0.5
# for i in range(len(data[0])):
def animate(i):
    evalu = eval_time_varying_distr(0.05*i)
    contour_fig.set_array(evalu[:-1,:-1].flatten())
    ax.collections = []
    ax.contourf(X,Y,evalu, zdir='z', offset=0.0, cmap='Greys', zorder=4)
    floor_plan.plot_ground(ax)
    # contour_fig.remove()
    # del contour_fig
    # contour_fig = ax.contourf(X,Y, evalu, zdir='z', offset=0.0, cmap='Greys')
    # cf.set_zorder(1)
    # contour_fig(X,Y,evalu)
    for agent_no in range(num_agents):
        yaw, pitch, roll = data[agent_no][i,3:6]
        angles = [roll, pitch, yaw]
        position = data[agent_no][i,0:3]
        position[-1] *= 0.5
        frame_points_trans = rotation_matrix(angles).dot(frame_points)
        frame_points_trans[0,:] += position[0]
        frame_points_trans[1,:] += position[1]
        frame_points_trans[2,:] += height_scale#*position[2]

        # ax.plot(frame_points_trans[0,0:2],frame_points_trans[1,0:2],frame_points_trans[2,0:2], color='black', linewidth=1, antialiased=True, zorder=3)
        # ax.plot(frame_points_trans[0,2:4],frame_points_trans[1,2:4],frame_points_trans[2,2:4], color='black', linewidth=1, antialiased=True,zorder=3)
        # ax.plot([position[0]]*2, [position[1]]*2,[0, height_scale*position[2]], color='red', ls='--',linewidth=1, antialiased=True,zorder=4)

        quads[agent_no]['l1'].set_data(frame_points_trans[0,0:2],frame_points_trans[1,0:2])
        quads[agent_no]['l1'].set_3d_properties(frame_points_trans[2,0:2])
        quads[agent_no]['l1'].set_zorder(3)

        quads[agent_no]['l2'].set_data(frame_points_trans[0,2:4],frame_points_trans[1,2:4])
        quads[agent_no]['l2'].set_3d_properties(frame_points_trans[2,2:4])
        quads[agent_no]['l2'].set_zorder(3)

        quads[agent_no]['height'].set_data([position[0]]*2, [position[1]]*2)
        # quads[agent_no]['height'].set_3d_properties([0, height_scale*position[2]])
        quads[agent_no]['height'].set_3d_properties([0, height_scale])

        quads[agent_no]['height'].set_zorder(4)
        if i > 50:
            quads[agent_no]['trajectory'].set_data(data[agent_no][i-50:i,0], data[agent_no][i-50:i,1])
            quads[agent_no]['trajectory'].set_3d_properties(0*data[agent_no][i-50:i,2])
            quads[agent_no]['trajectory'].set_zorder(6)
            # ax.plot(data[agent_no][i-50:i,0], data[agent_no][i-50:i,1],0*data[agent_no][i-50:i,2], zorder=6)

        else:
            quads[agent_no]['trajectory'].set_data(data[agent_no][0:i,0], data[agent_no][0:i,1])
            quads[agent_no]['trajectory'].set_3d_properties(0*data[agent_no][0:i,2])
            # ax.plot(data[agent_no][0:i,0], data[agent_no][0:i,1],0*data[agent_no][0:i,2], zorder=6)
            quads[agent_no]['trajectory'].set_zorder(6)

        # ax.set_zlim3d(0,0.2)
        #
        # ax.grid(False)
        # # Hide axes ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # quads[agent_no]['trajectory'].set_zorder(5)
    # plt.pause(0.000001)
    # plt.show()
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=10)

anim.save('obstacle_trial1/obstacle_time_varying.mp4', fps=30)
plt.show()
