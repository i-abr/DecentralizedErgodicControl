import numpy as np
from .basis import Basis
from .barrier import Barrier
from .replay_buffer import ReplayBuffer
from copy import deepcopy
import rospy
from std_msgs.msg import String, Float32MultiArray
from .target_dist import TargetDist

from decentralized_ergodic.msg import Ck

class DErgControl(object):

    def __init__(self, agent_name, model,
                    weights=None, t_horizon=10, num_basis=5,
                    capacity=100000, batch_size=20):

        self._agent_name     = agent_name
        self._model          = model
        self._t_horizon        = t_horizon
        self._replay_buffer = ReplayBuffer(capacity)
        self._batch_size    = batch_size

        self._basis = Basis(self._model.explr_space, num_basis=num_basis)
        self._lamk  = np.exp(-0.8*np.linalg.norm(self._basis.k, axis=1))
        self._barr  = Barrier(self._model.explr_space)

        self._targ_dist = TargetDist(basis=self._basis)

        self._u = [0.0*np.zeros(self._model.action_space.shape[0])
                        for _ in range(t_horizon)]
        # TODO: implement more weights
        if weights is None:
            weights = {'R' : np.eye(self._model.action_space.shape[0])}
        self._Rinv = np.linalg.inv(weights['R'])

        self._phik      = None
        self._ck_mean = None
        self._ck_msg    = Ck()
        self._ck_msg.name = self._agent_name
        self._ck_dict   = {}

        self.pred_path = []

        # set up the eos stuff last
        rospy.Subscriber('ck_link', Ck, self._ck_link_callback)
        self._ck_pub = rospy.Publisher('ck_link', Ck, queue_size=1)


    def _ck_link_callback(self, msg):
        if msg.name != self._agent_name:
            if msg.name in self._ck_dict:
                self._ck_dict[msg.name] = np.array(msg.ck)
            else:
                self._ck_dict.update({msg.name : np.array(msg.ck)})

    def reset(self):
        self._u = [0.0*np.zeros(self._model.action_space.shape[0])
                for _ in range(self._t_horizon)]
        self._replay_buffer.reset()


    def __call__(self, state):
        assert self._targ_dist.phik is not None, 'Forgot to set phik'

        self._u[:-1] = self._u[1:]
        self._u[-1]  = np.zeros(self._model.action_space.shape[0])

        x = self._model.reset(state.copy())

        pred_traj = []
        dfk       = []
        fdx       = []
        fdu       = []
        dbar      = []
        for t in range(self._t_horizon):

            # collect all the information that is needed
            pred_traj.append(
                    x[self._model.explr_idx]
            )
            dfk.append(
                    self._basis.dfk(x[self._model.explr_idx])
            )

            dbar.append(
                    self._barr.dx(x[self._model.explr_idx])
            )
            # step the model forwards
            x = self._model.step(self._u[t])
            fdx.append(self._model.A)
            fdu.append(self._model.B)


        self.pred_path = deepcopy(pred_traj)
        # sample any past experiences
        if len(self._replay_buffer) > self._batch_size:
            past_states     = self._replay_buffer.sample(self._batch_size)
            pred_traj       = pred_traj + past_states
        else:
            past_states = self._replay_buffer.sample(len(self._replay_buffer))
            pred_traj   = pred_traj + past_states

        # calculate the cks for the trajectory
        # *** this is also in the utils file
        N = len(pred_traj)
        ck = np.sum([self._basis.fk(xt) for xt in pred_traj], axis=0) / N
        self._ck_msg.ck = ck.copy()
        self._ck_pub.publish(self._ck_msg)

        if len(self._ck_dict.keys()) > 1:
            self._ck_dict[self._agent_name] = ck
            cks = []
            for key in self._ck_dict.keys():
                cks.append(self._ck_dict[key])
            ck = np.mean(cks, axis=0)
            # print('sharing and make sure first ck is 0 ', ck[0])
        self._ck_mean = ck

        fourier_diff = self._lamk * (ck - self._targ_dist.phik)
        fourier_diff = fourier_diff.reshape(-1,1)


        # backwards pass
        rho = np.zeros(self._model.observation_space.shape[0])
        for t in reversed(range(self._t_horizon)):
            edx = np.zeros(self._model.observation_space.shape[0])
            edx[self._model.explr_idx] = np.sum(dfk[t] * fourier_diff, 0)

            bdx = np.zeros(self._model.observation_space.shape[0])
            bdx[self._model.explr_idx] = dbar[t]
            rho = rho - self._model.dt * (-edx-bdx-np.dot(fdx[t].T, rho))

            self._u[t] = -np.dot(np.dot(self._Rinv, fdu[t].T), rho)
            if (np.abs(self._u[t]) > 1.0).any():
                self._u[t] /= np.linalg.norm(self._u[t])
        self._replay_buffer.push(state[self._model.explr_idx].copy())
        return self._u[0].copy()
