import numpy as np
import matplotlib.pyplot as plt

from single_ergodic_quad.ergodic_objective import ErgodicObjective

from IPython import embed

def main():
    horizon = 1.5
    time_step = 0.05
    objective = ErgodicObjective(horizon, time_step)

    floor_plan_x = [0., 0., 0.25, 0.25, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.]
    floor_plan_y = [0., 0.5, 0.5, 0.75, 0.75, 0., 0., 0.5, 0.5, 0.25, 0.25, 0., 0.]

    # plt.plot(floor_plan_x,floor_plan_y, linewidth=2)

    data = []
    num_agents = 10
    for i in range(num_agents):
        data.append(np.load('trial2/agent{}.npy'.format(i)))


    err_in_time = []
    phik = objective.target_distribution.get_phik()
    ind_err_in_time = []
    t = []
    for i in range(2,len(data[0]), 10):
        cki = None
        ind_err = []
        for trajectory in data:
            if cki is None:
                cki = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[0:i]))
                ck = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[0:i]))
            else:
                ck = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[0:i]))
                cki += objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[0:i]))
            print(np.sum(objective.lamk * (ck - phik)**2))
            # plt.plot(trajectory[:,0], trajectory[:,1])
            ind_err.append(np.sum(objective.lamk * (ck - phik)**2))
        ind_err_in_time.append(ind_err)
        cki /= 10.0
        print(cki)
        print(np.sum(objective.lamk * (cki - phik)**2))
        err_in_time.append(np.sum(objective.lamk * (cki - phik)**2))
        t.append(i*time_step)
    plt.plot(t,err_in_time, 'k')
    plt.plot(t,ind_err_in_time, 'r')
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    main()
