import numpy as np
import matplotlib.pyplot as plt

centralized_ergodic_err = np.load('centralized_ergodic_err.npy')
decentralized_ergodic_err = np.load('decentralized_ergodic_err.npy')

steps = 5
len_data = len(centralized_ergodic_err)
centr_err_resampled = []
decentr_err_resampled = []
time_arr = []
for i in range(0, len_data, steps):
    centr_err_resampled.append(centralized_ergodic_err[i])
    decentr_err_resampled.append(decentralized_ergodic_err[i])
    time_arr.append(i * 0.05)
plt.plot(time_arr, centr_err_resampled, 'r')
plt.plot(time_arr, decentr_err_resampled, 'k')
plt.grid(True)
plt.show()
