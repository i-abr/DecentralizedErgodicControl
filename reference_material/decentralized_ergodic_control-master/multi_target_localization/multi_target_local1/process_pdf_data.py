import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def pdf(mu, sigma, _eval):
    k = len(mean)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    Z = 1.0/np.sqrt(det_sigma * (2.0 * pi)**k)
    delta = mean - _eval
    return np.exp(-0.5 * (delta).dot(inv_sigma).dot(delta))

X, Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))

no_targets = 4
no_agents = 5
time_indx = [0, 10, 20, 100,200,300,399]
for agent_no in range(no_agents):
    for target_no in range(no_targets):

        target_data = [
            np.load('agent{}/target{}_mean_data.npy'.format(agent_no, target_no)),
            np.load('agent{}/target{}_covar_data.npy'.format(agent_no, target_no))
        ]
        data_array = []
        for i in range(0,400):
            mean = target_data[0][i]
            covar = target_data[1][i].reshape((2,2))
            data = np.array(map(lambda x: pdf(mean, covar, x), np.c_[X.ravel(),Y.ravel()]))
            data = data.reshape(X.shape)
            data_array.append(data)
        np.save('agent{}/target{}_pdf_time_data.npy'.format(agent_no, target_no), data_array)
        print('agent {}, target {}'.format(agent_no, target_no))
