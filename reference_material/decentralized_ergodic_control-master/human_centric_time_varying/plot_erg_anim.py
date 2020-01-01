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


fig, ax = plt.subplots(figsize=(10,4))



quads = {0:{},1:{},2:{},3:{},4:{}}

num_agents = 5

evalu = eval_time_varying_distr(0.0)
contour_fig = ax.contourf(X,Y,evalu, cmap='Greys',zorder=0)
# contour_fig, = ax.contourf([],[], [], zdir='z',cmap='Greys')

# embed()
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.set_zticklabels([])
# surf_fig.set_zorder(21)


ax.grid(False)
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
# ax.axis('square')

data = [np.load('obstacle_trial3/agent{}.npy'.format(i)) for i in range(num_agents)]

erg_data = np.load('obstacle_trial3/collective_ergodicity2.npy')
ind_erg_data = np.load('obstacle_trial3/individual_ergodicity2.npy')
t = [i*0.05 for i in range(len(erg_data))]
from matplotlib import animation

height_scale = 0.2
# for i in range(len(erg_data)):
def animate(i):
    ax.clear()
    ax.plot(t[0:i],erg_data[0:i],'k')
    ax.plot(t[0:i],ind_erg_data[0:i],'r')
    ax.set_xlim(0,20)
    ax.set_ylim(0,4.5)
    ax.set_ylabel('Ergodicity')
    ax.set_xlabel('time (s)')

    plt.pause(0.000001)
    # plt.show()
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=10)

anim.save('obstacle_trial3/obstacle_time_varying_2d_erg_wind2.mp4', fps=30)
plt.show()
