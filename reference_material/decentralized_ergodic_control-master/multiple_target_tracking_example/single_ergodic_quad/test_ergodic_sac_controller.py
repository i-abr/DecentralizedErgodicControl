import numpy as np
from lqr_controller import InfHorizLQR
from quadcopter_dynamics import Quadcopter
from sac import SAC
from ergodic_objective import ErgodicObjective
import matplotlib.pyplot as plt

def main():


    model = Quadcopter()
    target_state = np.array([0. for _ in range(12)])
    target_state[0] = 0.
    target_state[1] = 0.
    target_state[2] = 1.0


    print('Initializing linearization and constructing the lqr controller')
    # Sort of just copied this from else where
    A, B = model.get_linearization(target_state, np.array([1.0, 0., 0., 0.]))
    Q, R = np.diag([0.00,0.00,20.] + [0.1]*9), np.diag([0.1,0.1,0.1,0.1])
    lqr_controller = InfHorizLQR(A, B, Q, R, target_state=target_state)


    print('Initializing the state and the simulation')

    horizon_in_sec = 1.5
    time_step = 0.01
    time_iter = 5
    controller_time_step = time_iter * time_step

    objective = ErgodicObjective(horizon_in_sec, controller_time_step)
    controller = SAC(model, objective, controller_time_step, horizon_in_sec, default_control=lqr_controller)
    state = np.array([0. for _ in range(12)])
    state[0] = 0.63
    state[1] = 0.22
    state[2] = 0.1

    state[3] = 0.0
    state[4] = 0.0

    simulation_time = 0.0
    final_simulation_time = 40.0
    trajectory = state.copy()
    while simulation_time < final_simulation_time:
        u = controller(state)
        for _ in range(time_iter):
            state = model.step(state, u, time_step)
            simulation_time += time_step
        print(state[0:3])
        controller.objective.trajectory_distribution.remember(state.copy())
        trajectory = np.vstack((trajectory, state.copy()))
        plt.clf()
        plt.plot(trajectory[:,0], trajectory[:,1])
        plt.axis('square')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pause(0.001)

if __name__ == '__main__':
    main()
