import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from basis import Basis
def pdf(mu, sigma, _eval):
    k = len(mean)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    Z = 1.0/np.sqrt(det_sigma * (2.0 * pi)**k)
    delta = mean - _eval
    return np.exp(-0.5 * (delta).dot(inv_sigma).dot(delta))

xlim = [[0.0,1.0]]*2
coef = np.array([2]*2)
lamk = np.zeros(coef+1)
k_list = []
# for i in range(coef[0] + 1):
#     for j in range(coef[1] + 1):
#         # self.lamk[i,j] = np.exp(-0.8 * np.linalg.norm([i,j]))
#         lamk[i,j] = 1.0/(np.linalg.norm([i,j])+1)**(3.0/2.0)
#         k_list.append([i,j])
# lamk = lamk.ravel()
basis = Basis(xlim, coef, k_list)


X, Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))
no_agents = 3
no_targets = 1
true_target_data = np.loadtxt('target0/target_state_data.csv')

agents = {0:{}, 1:{}, 2:{}}
for agent_no in range(no_agents):
    agents[agent_no]['state'] = np.loadtxt('agent{}/robot_state_data.csv'.format(agent_no))
    agents[agent_no]['phik'] = np.loadtxt('agent{}/phik_data.csv'.format(agent_no))
    agents[agent_no]['target_covar'] = np.loadtxt('agent{}/target_covar_data.csv'.format(agent_no))
    agents[agent_no]['target_mean'] = np.loadtxt('agent{}/target_mean_data.csv'.format(agent_no))



time_stamps = [300,320, 330, 360]
# plt.ion()
for i in time_stamps:
    for agent_no in range(no_agents):
        plt.figure(agent_no+i)
        mean = agents[agent_no]['target_mean'][i]
        covar = agents[agent_no]['target_covar'][i].reshape((2,2))
        data = np.array(map(lambda x: pdf(mean, covar, x), np.c_[X.ravel(),Y.ravel()]))
        data = data.reshape(X.shape)
        data /= np.max(data)
        plt.imshow(data, cmap='Greys')

plt.show()
        # np.save('agent0_pdf_data/pdf_data_at_{}.npy'.format(i), data)
