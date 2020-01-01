import numpy as np
import matplotlib.pyplot as plt

no_agents = 3
time_stamps = [10,20,30,40]
time_step = 0.05
indexes = [ int(time_stamp/time_step) for time_stamp in time_stamps]

X,Y = np.meshgrid( np.linspace(0, 1, 60), np.linspace(0,1,60))

fig, ax = plt.subplots(2, len(time_stamps), sharex='col', sharey='row' )
# fig, ax = plt.subplots(2, len(time_stamps) )


for i,index in enumerate(indexes):
    fig_data = np.load('agent2_gp_data_at_index_{}.npy'.format(index))
    x_data = np.loadtxt('agent2/robot_state_data.csv')
    ax[0, i].contourf(X, Y, fig_data.reshape((60,60)), cmap="Greys")
    if i == 0:
        ax[0, i].plot(x_data[0:index,0], x_data[0:index,1])
    else:
        past_idx = indexes[i-1]
        ax[0, i].plot(x_data[past_idx:index,0], x_data[past_idx:index,1])
    # ax[0,i].set(adjustable='box-forced')
    # ax[0,i].axis('equal')
    # ax[0,i].axis('square')
    fig_data = np.load('collective_gp_data_at_index_{}.npy'.format(index))
    for agent_no in range(no_agents):
        x_data = np.loadtxt('agent{}/robot_state_data.csv'.format(agent_no))
        if i == 0:
            ax[1, i].plot(x_data[0:index,0], x_data[0:index,1])
        else:
            past_idx = indexes[i-1]
            ax[1, i].plot(x_data[past_idx:index,0], x_data[past_idx:index,1])

    ax[1, i].contourf(X,Y,fig_data, cmap="Greys")
    # ax[1,i].set(adjustable='box-forced')

    # ax[1,i].axis('equal')
    # ax[1,i].axis('square')
    # plt.axis('equal')
    # plt.axis('square')

plt.show()
