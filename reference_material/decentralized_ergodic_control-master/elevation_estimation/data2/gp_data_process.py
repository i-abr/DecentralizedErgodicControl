import numpy as np
import matplotlib.pyplot as plt
from basis import Basis
from gaussian_process import GaussianProcess


X,Y = np.meshgrid( np.linspace(0, 1, 60), np.linspace(0,1,60))

agent_data = []
no_agents = 3
for i in range(no_agents): # for each agent
    # I need  to get the data from each agent and train a GP
    Xdata = np.loadtxt('agent{}/X_data.csv'.format(i))
    ydata = np.loadtxt('agent{}/y_data.csv'.format(i))
    agent_data.append([Xdata, ydata])

time_stamps = [10,20,30,40]
time_step = 0.05

indexes = [ int(time_stamp/time_step) for time_stamp in time_stamps]
agent_gp = [GaussianProcess() for i in range(no_agents)]

# for i in range(no_agents):
#     for index in indexes:
#         Xdata, ydata = agent_data[i]
#         gp = GaussianProcess(X = Xdata[0:index], y = ydata[0:index])
#
#         data = np.array(map(gp, np.c_[X.ravel(), Y.ravel()]))
#         print('Saving agent {} index {} data'.format(i, index))
#         np.save('agent{}_gp_data_at_index_{}'.format(i, index), data)
#         # plt.show()
#         # plt.imshow(data.reshape(X.shape), origin='lower', extent=(0,1,0,1))


gp = GaussianProcess()
for index in indexes:
    X_collective = []
    y_collective = []
    for i in range(no_agents):
        Xdata, ydata = agent_data[i]
        X_collective = X_collective + list(Xdata[0:index])
        y_collective = y_collective + list(ydata[0:index])
    gp.compute(X_collective, y_collective)

    data = np.array(map(gp, np.c_[X.ravel(), Y.ravel()])).reshape(X.shape)

    print('Saving index {} data'.format(index))
    np.save('collective_gp_data_at_index_{}'.format(index), data)
    # plt.imshow(data, origin='lower', extent=(0,1,0,1))
    # plt.show()
