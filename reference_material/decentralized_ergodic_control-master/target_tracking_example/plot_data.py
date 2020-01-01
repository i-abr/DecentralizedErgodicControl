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
    num_agents = 5
    for i in range(num_agents):
        data.append(np.load('trial1/agent{}.npy'.format(i)))


    err_in_time = []
    phik = objective.target_distribution.get_phik()
    ind_err_in_time = []
    t = []
    for i in range(2,len(data[0]),10):
        cki = None
        phik = objective.target_distribution.update_phik(i*time_step)
        ind_err = []
        if i+int(horizon/time_step) >= len(data[0]):
            break
        for trajectory in data:
            if i > objective.ergodic_memory:
                start_idx = i-objective.ergodic_memory
                end_idx = i+int(horizon/time_step)
            else:
                start_idx = 0
                end_idx = i+int(horizon/time_step)
            if cki is None:
                cki = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[start_idx:end_idx]))
                ck = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[start_idx:end_idx]))
            else:
                ck = objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[start_idx:end_idx]))
                cki += objective.trajectory_distribution.get_ck_from_trajectory(list(trajectory[start_idx:end_idx]))
            # print(np.sum(objective.lamk * (ck - phik)**2))
            # plt.plot(trajectory[:,0], trajectory[:,1])
            ind_err.append(np.sum(objective.lamk * (ck - phik)**2))
        ind_err_in_time.append(ind_err)
        cki /= num_agents
        print(cki)
        print(np.sum(objective.lamk * (cki - phik)**2))
        err_in_time.append(np.sum(objective.lamk * (cki - phik)**2))
        t.append(i*time_step)

    # np.save('obstacle_trial1/individual_ergodicity.npy', ind_err_in_time)
    # np.save('obstacle_trial1/collective_ergodicity.npy', err_in_time)
    plt.plot(t,err_in_time, 'k')
    plt.plot(t,ind_err_in_time, 'r')
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    main()
