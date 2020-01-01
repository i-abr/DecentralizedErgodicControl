import numpy as np
from lqr_controller import InfHorizLQR
from quadcopter_dynamics import Quadcopter


def main():
    '''
        Script to test out the continuous LQR controller on the quadcopter
    '''


    model = Quadcopter()
    target_state = np.array([0. for _ in range(12)])
    target_state[2] = 2.0


    print('Initializing linearization and constructing the lqr controller')
    # Sort of just copied this from else where
    A, B = model.get_linearization(target_state, np.array([1.0]*4))
    Q, R = np.diag([20.,20.,20.] + [0.1]*9), np.diag([0.1,0.1,0.1,0.1])
    lqr_controller = InfHorizLQR(A, B, Q, R, target_state=target_state)


    print('Initializing the state and the simulation')


    state = np.array([0. for _ in range(12)])
    state[0] = 2.0
    state[1] = 1.2
    state[2] = 2.0

    initial_state = state.copy()

    time_step = 0.05
    simulation_time = 0.0
    final_simulation_time = 20.0

    print('Running Simulation')
    while simulation_time < final_simulation_time:
        u = lqr_controller(state)
        state = model.step(state, u, time_step)
        simulation_time += time_step
    print('Simulation Finished')
    print('Initial state : {}'.format(initial_state))
    print('State : {}, \n Control : {}'.format(state, u))


if __name__ == '__main__':
    main()
