import numpy as np
import matplotlib.pyplot as plt
from terrain import Terrain

terrain = Terrain()

X,Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))

terrain_data = np.array(map(terrain.getElevationAtX, np.c_[X.ravel(), Y.ravel()]))

time_stamps = [10,20,30,40]
time_step = 0.05
no_agents = 3

indexes = [ int(time_stamp/time_step) for time_stamp in time_stamps]

single_agent_err = []
collective_err = []
for index in indexes:
    data = np.load('agent2_gp_data_at_index_{}.npy'.format(index))
    single_agent_err.append(np.sum((data.ravel() - terrain_data)**2)/(60**2))
    data = np.load('collective_gp_data_at_index_{}.npy'.format(index))
    collective_err.append(
            np.sum( (data.ravel() - terrain_data)**2 ) / (60**2)
    )

plt.plot(time_stamps, single_agent_err)
plt.plot(time_stamps, collective_err,'r')
plt.grid(True)

plt.figure(2)
plt.contourf(X, Y, terrain_data.reshape((60,60)), cmap="Greys")
# steps = 5
# len_data = len(centralized_ergodic_err)
# centr_err_resampled = []
# decentr_err_resampled = []
# time_arr = []
# for i in range(0, len_data, steps):
#     centr_err_resampled.append(centralized_ergodic_err[i])
#     decentr_err_resampled.append(decentralized_ergodic_err[i])
#     time_arr.append(i * 0.05)
# plt.plot(time_arr, centr_err_resampled, 'r')
# plt.plot(time_arr, decentr_err_resampled, 'k')
plt.show()
