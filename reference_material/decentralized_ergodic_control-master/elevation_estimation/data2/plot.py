import numpy as np
import matplotlib.pyplot as plt
from basis import Basis
from gaussian_process import GaussianProcess

xlim = [[0.0,1.0],[0.0,1.0]]
coef = np.array([6] * 2)
lamk = np.zeros(coef+1)
k_list = []
for i in range(coef[0]+1):
    for j in range(coef[1]+1):
        lamk[i,j] = 1.0/(np.linalg.norm([i,j]) + 1)**(3.0/2.0)
        k_list.append([i,j])
lamk = lamk.ravel()

basis = Basis(xlim, coef, k_list)

def get_ck(x, basis, coef, k_list):
    N = len(x)
    cktemp = np.zeros(coef+1).ravel()
    for i,k in enumerate(k_list):
        _fk = [basis.fk(k, x[ii]) for ii in range(N)]
        cktemp[i] = np.sum(_fk)
    return cktemp

no_agents = 3

phik = np.loadtxt('agent0/phik_data.csv')
agent_cks = []
agent_trajectory = []
ck_elements = []
for i in range(no_agents):
    agent_cks.append(np.loadtxt('agent{}/ck_data.csv'.format(i)))
    agent_trajectory.append(np.loadtxt('agent{}/robot_state_data.csv'.format(i)))
    plt.plot(agent_trajectory[-1][:,0], agent_trajectory[-1][:,1])
plt.show()

# for each agent calculate the ck values and store then at time i




# for time i, calculate the cumulative error and then for each agent calculate the individual errors



steps = 1
total_err = []
cum_err = []
for agent_path in agent_trajectory:
    ck_i = []
    for i in range(0, len(agent_path)-1, steps):
        ck_i.append(get_ck(agent_path[0:i+1], basis, coef, k_list)/(i+1))
        print('{} out of {}'.format(i, len(agent_path)-1))
    cum_err.append(ck_i) # store these
    err = []
    for i in range(0, len(ck_i)):
        err.append(
                np.sum(
                     lamk * (phik[i] - ck_i[i])**2
                )
        )
    total_err.append(err)

other_err = []
for i in range(0, len(ck_i)):
    combined_err = None
    for c in cum_err:
        if combined_err is None:
            combined_err = c[i]
        else:
            combined_err += c[i]
    other_err.append(
            np.sum(
                lamk * ( phik[i] - combined_err/no_agents ) ** 2
            )
    )

for err_array in total_err:
    plt.plot(err_array)
plt.plot(other_err,'k')
# plt.xlim(0,20)
plt.show()




n_samples = len(agent_cks[0])

err = []
ck_sum = None
agent_ck_sum = [None, None, None]
agent_err = []
for i in range(n_samples):
    avg_ck = None
    for j,agent_ck in enumerate(agent_cks):

        if agent_ck_sum[j] is None:
            agent_ck_sum[j] = agent_ck[i].copy()
        else:
            agent_ck_sum[j] += agent_ck[i].copy()

        if avg_ck is None:
            avg_ck = agent_ck[i].copy()
        else:
            avg_ck += agent_ck[i].copy()
    avg_ck /= 3.0
    if ck_sum is None:
        ck_sum = avg_ck.copy()
    else:
        ck_sum += avg_ck.copy()

    err.append( np.sum(lamk*(phik[i] - ck_sum/(i+1))**2) )
    agent_err.append([ np.sum(lamk*(phik[i] - cki/(i+1) )**2) for cki in agent_ck_sum ])


np.save('centralized_ergodic_err', other_err)
np.save('decentralized_ergodic_err', err)
# plt.semilogy(err)
plt.plot(err,'r')
plt.plot(other_err,'b')
# plt.plot(agent_err,'k')

# for err_array in total_err:
#     plt.plot(err_array,'k')

plt.xlim(0,500)

plt.show()
