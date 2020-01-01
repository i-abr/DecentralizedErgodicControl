import numpy as np
from .lqr_controller import InfHorizLQR
from .quadcopter_dynamics import Quadcopter
from .sac import SAC
from .ergodic_objective import ErgodicObjective


class QuadcopterAgent(object):
    ''' Quadcopter ergodic agent '''
    def __init__(self):
        self.model = Quadcopter()
        target_state = np.array([0. for _ in range(12)])
        target_state[0] = 0.
        target_state[1] = 0.
        target_state[2] = 1.0

        # Sort of just copied this from else where
        A, B = self.model.get_linearization(target_state, np.array([1.0, 0., 0., 0.]))
        Q, R = np.diag([0.00,0.00,20.] + [0.01]*9), np.diag([0.1,0.1,0.1,0.1])
        self.lqr_controller = InfHorizLQR(A, B, Q, R, target_state=target_state)

        horizon_in_sec = 1.5
        self.time_step = 0.01
        self.time_iter = 5
        self.controller_time_step = self.time_iter * self.time_step

        self.objective = ErgodicObjective(horizon_in_sec, self.controller_time_step)
        self.controller = SAC(self.model, self.objective, self.controller_time_step, horizon_in_sec, default_control=self.lqr_controller)
        self.state = np.array([0. for _ in range(12)])
        self.state[0] = np.random.uniform(0.3, 0.5)
        self.state[1] = np.random.uniform(0.5, 0.65)
        self.state[2] = np.random.uniform(0.0, 0.5)

        self.state[3] = 0.0
        self.state[4] = 0.0


        self.trajectory = self.state.copy()

        self.control_step() # preinit system


    def control_step(self, cki=None):
        u, self.ck = self.controller(self.state, cki=cki)
        for _ in range(self.time_iter):
            self.state = self.model.step(self.state, u, self.time_step)
        self.controller.objective.trajectory_distribution.remember(self.state.copy())

        self.trajectory = np.vstack((self.trajectory, self.state.copy()))
        # self.ck = self.controller.get_ck(self.state)

    def save(self, filePath):
        np.save(filePath,self.trajectory)
