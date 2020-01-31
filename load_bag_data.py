import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")

pos_data = []
pos_time = []
pos_postee_data = []
pos_postee_time = []

tdist0_data = []
tdist0_time = []
tdist0_postee_data = []
tdist0_postee_time = []

tdist1_data = []
tdist1_time = []
tdist1_postee_data = []
tdist1_postee_time = []

ee_data = []
ee_time = []

# fname = 'data/het_sim_6agents.bag'
fname = '/home/anon/darpa_sim_data/hetsim_data/het_sim_6agents_3.bag'
bag = rosbag.Bag(fname)

ee = False
for topic, msg, t in bag.read_messages(['/agent0/target_dist', '/agent1/target_dist','/tf', '/ee_loc' ]): 
    if topic == '/agent0/target_dist':
        if ee:
            tdist0_postee_time.append([t.secs])
            tdist0_postee_data.append([msg.data[0].data])
        else:
            tdist0_time.append([t.secs])
            tdist0_data.append([msg.data[0].data]) 
    elif topic == '/agent1/target_dist':
        if ee:
            tdist1_postee_time.append([t.secs])
            tdist1_postee_data.append([msg.data[0].data])
        else:
            tdist1_time.append([t.secs])
            tdist1_data.append([msg.data[0].data])
    elif topic == '/tf':
        if ee:
            pos_postee_time.append([t.secs])
            pos_postee_data.append([msg.transforms])
        else:
            pos_time.append([t.secs])
            pos_data.append([msg.transforms])
    elif topic =='/ee_loc': 
        ee_time.append([t.secs])
        ee_data.append([msg.position.x, msg.position.y])
        ee = True
bag.close()

print(len(pos_data), len(tdist0_data), len(tdist1_data))

# fig, (ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(np.array(tdist0_postee_data[-1][0]).reshape((50,50)), origin='lower', extent=(0,1,0,1))
# ax2.imshow(np.array(tdist1_postee_data[-1][0]).reshape((50,50)), origin='lower',extent=(0,1,0,1))
# plt.show()

swarm_pos = [[],[],[],[],[],[]]
for i in range(len(pos_data)):
    ind = int(pos_data[i][0][0].child_frame_id[5:])
    swarm_pos[ind].append([pos_data[i][0][0].transform.translation.x/10,pos_data[i][0][0].transform.translation.y/10])

swarm_pos_postee = [[],[],[],[],[],[]]
for i in range(len(pos_postee_data)):
    ind = int(pos_postee_data[i][0][0].child_frame_id[5:])
    swarm_pos_postee[ind].append([pos_postee_data[i][0][0].transform.translation.x/10,pos_postee_data[i][0][0].transform.translation.y/10])

# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

# agent_name = 'Agent 1'
# traj = np.array(swarm_pos[0])
# ax1.plot(traj[:,0],traj[:,1],label=agent_name)
# ax1.axis('square')
# ax1.set_xlim(0,1)
# ax1.set_ylim(0,1)


# for i in range(1,len(swarm_pos)):
#     agent_name = 'Agent ' + str(i+1)
#     traj = np.array(swarm_pos[i])
#     ax2.plot(traj[:,0],traj[:,1],label=agent_name)
# ax2.axis('square')
# ax2.set_xlim(0,1)
# ax2.set_ylim(0,1)


# agent_name = 'Agent 1'
# traj = np.array(swarm_pos_postee[0])
# ax3.plot(traj[:,0],traj[:,1],label=agent_name)
# ax3.axis('square')
# ax3.set_xlim(0,1)
# ax3.set_ylim(0,1)

# for i in range(1,len(swarm_pos_postee)):
#     agent_name = 'Agent ' + str(i+1)
#     traj = np.array(swarm_pos_postee[i])
#     ax4.plot(traj[:,0],traj[:,1],label=agent_name)
# ax4.axis('square')
# ax4.set_xlim(0,1)
# ax4.set_ylim(0,1)

# plt.show()


from d_erg_lib.utils import *
from quad_agent.agent import Agent

sns.set(style="dark")

def plot_ck(xt, fig=None): 
    agent = Agent('agent1')
    ck = convert_traj2ck(agent.ctrllr._basis, xt)
    ck_grid = convert_ck2dist(agent.ctrllr._basis, ck).reshape((50,50))
    if fig is not None:
        fig.imshow(ck_grid, origin="lower", extent=(0,1,0,1))
    else: 
        plt.imshow(ck_grid, origin="lower", extent=(0,1,0,1))

def plot_combined_ck(xt_list,fig=None): 
    agent = Agent('agent1')
    cklist = []
    for i in range(len(xt_list)):                   
        ck = convert_traj2ck(agent.ctrllr._basis, np.array(xt_list[i]))
        cklist.append(ck)
    cklist = np.array(cklist)
    ck_avg = np.mean(cklist, axis=0)
    ck_grid = convert_ck2dist(agent.ctrllr._basis, ck_avg).reshape((50,50))
    if fig is not None:
        fig.imshow(ck_grid, origin="lower", extent=(0,1,0,1))
    else:
        plt.imshow(ck_grid, origin="lower", extent=(0,1,0,1))


fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, sharex=True,sharey=True)

# agent_name = 'Agent 1'
# traj = np.array(swarm_pos[0])
# ax1.plot(traj[:,0],traj[:,1],label=agent_name)
# ax1.axis('square')
# ax1.set_xlim(0,1)
# ax1.set_ylim(0,1)
# ax1.set_title('Uniform: Agent 1')
# ax1.set_ylabel('y')

for i in range(0,len(swarm_pos)):
    agent_name = 'Agent ' + str(i+1)
    traj = np.array(swarm_pos[i])
    ax1.plot(traj[:,0],traj[:,1],label=agent_name)
ax1.axis('square')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
# ax1.set_title('Uniform Exploration') 

agent_name = 'Agent 1'
traj = np.array(swarm_pos_postee[0])
ax2.plot(traj[:,0],traj[:,1],label=agent_name)
ax2.axis('square')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
# ax2.set_title('DD: Agent 1')

for i in range(1,len(swarm_pos_postee)):
    agent_name = 'Agent ' + str(i+1)
    traj = np.array(swarm_pos_postee[i])
    ax3.plot(traj[:,0],traj[:,1],label=agent_name)
ax3.axis('square')
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)

# ax3.set_title('DD: Agent 2-6')

# plot_ck(np.array(swarm_pos[0]),fig=ax5)
# ax5.set_ylabel('y')
# ax5.set_xlabel('x')
# ax5.set_title('Uniform Ck: 1')

plot_combined_ck(swarm_pos,fig=ax4)
ax4.set_xlabel('x')
# ax4.set_title('Uniform Ck: 2-6')
plot_ck(np.array(swarm_pos_postee[0]),fig=ax5)
ax5.set_xlabel('x')
# ax5.set_title('DD Ck: 1')
plot_combined_ck(swarm_pos_postee[1:],fig=ax6)
ax6.set_xlabel('x')
# ax6.set_title('DD Ck: 2-6')
plt.show()
