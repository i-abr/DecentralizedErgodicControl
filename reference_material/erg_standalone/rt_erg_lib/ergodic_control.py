import numpy as np
from basis import Basis
from barrier import Barrier
from replay_buffer import ReplayBuffer

import rospy
from std_msgs.msg import String, Float32MultiArray

from demo_dist import TargetDist
#from obfu_dist import ObfuDist
from tanvas_dist import TanvasDist
from hideNseek_dist import HideNSeekDist
from hvt_dist import HVTDist
class RTErgodicControl(object):

    def __init__(self, agent_name, model,
                    weights=None, horizon=100, num_basis=5,
                    capacity=100000, batch_size=20):

        self.agent_name = agent_name
        self.model       = model
        self.horizon     = horizon
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size    = batch_size
        
        self.basis = Basis(self.model.explr_space, num_basis=num_basis)
#         self.lamk  = 1.0/(np.linalg.norm(self.basis.k, axis=1) + 1)**(3.0/2.0)
        self.lamk = np.exp(-0.8*np.linalg.norm(self.basis.k, axis=1))
        self.barr = Barrier(self.model.explr_space)
        
        #self.t_dist_dict = {
        #            'demo' : TargetDist(basis=self.basis),
        #            # 'tanvas' : TanvasDist(basis=self.basis),
        #            # 'obfu' : ObfuDist(basis=self.basis, agent_num=agent_name),
        #            'hideNseek': HideNSeekDist(basis=self.basis),
        #            'hvt': HVTDist(basis=self.basis)
        #        }
        self.t_dist_dict = {
                    'demo' : TargetDist(basis=self.basis),
                    'tanvas' : TanvasDist(basis=self.basis),
                    # 'obfu' : ObfuDist(basis=self.basis, agent_num=agent_name),
                    'ied': HideNSeekDist(basis=self.basis),
                    'obfu': HVTDist(basis=self.basis)
                }
        
        self._mode = 'tanvas'
        self.target_dist = self.t_dist_dict[self._mode]
        rospy.Subscriber('demo_mode', String, self.set_mode_callback)
        rospy.Subscriber('received_floats', Float32MultiArray, self.ck_comms_callback) 
        self._ck_pub = rospy.Publisher('send_floats', Float32MultiArray)
        self._mode_pub = rospy.Publisher('explr_mode', String)
        
        mode_msg = String()
        mode_msg.data = self._mode
        self._mode_pub.publish(mode_msg)
            
        self.u_seq = [0.0*np.zeros(self.model.action_space.shape[0])
                        for _ in range(horizon)]
        if weights is None:
            weights = {'R' : np.eye(self.model.action_space.shape[0])}

        self.Rinv = np.linalg.inv(weights['R'])

        self._phik = None
        self.ck = None

        self._ck_dict = {}

    def ck_comms_callback(self, msg):
        if int(msg.data[0]) != self.agent_name:
            if self._ck_dict.has_key(int(msg.data[0])):
                self._ck_dict[int(msg.data[0])] = np.array(msg.data[1:])
            else:
                self._ck_dict.update({int(msg.data[0]) : np.array(msg.data[1:])})

    def set_mode_callback(self, msg):
        self._mode = msg.data
        self.target_dist = self.t_dist_dict[self._mode]
        self.replay_buffer.reset()
        print('switching to mode', self._mode)
        
        mode_msg = String()
        mode_msg.data = self._mode
        self._mode_pub.publish(mode_msg)

    def reset(self):
        self.u_seq = [0.0*np.zeros(self.model.action_space.shape[0])
                for _ in range(self.horizon)]
        self.replay_buffer.reset()


    def __call__(self, state, ck_list=None, agent_num=None):
        assert self.target_dist.phik is not None, 'Forgot to set phik, use set_target_phik method'
        
        # TODO: update each ck every iteration

        if self._mode == 'obfu' or self._mode == 'ied' or self._mode == 'tanvas':
            if self.t_dist_dict[self._mode]._update == True:
                self.replay_buffer.reset()
                self.t_dist_dict[self._mode]._update = False
        
        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1]  = np.zeros(self.model.action_space.shape[0])

        x = self.model.reset(state)

        pred_traj = []
        dfk = []
        fdx = []
        fdu = []
        dbar= []
        for t in range(self.horizon):
            # collect all the information that is needed
            pred_traj.append(x[self.model.explr_idx])
            dfk.append(self.basis.dfk(x[self.model.explr_idx]))
            fdx.append(self.model.fdx(x, self.u_seq[t]))
            fdu.append(self.model.fdu(x))
            dbar.append(self.barr.dx(x[self.model.explr_idx]))
            # step the model forwards
            x = self.model.step(self.u_seq[t] * 0.)

        # sample any past experiences
        if len(self.replay_buffer) > self.batch_size:
            past_states = self.replay_buffer.sample(self.batch_size)
            pred_traj = pred_traj + past_states
        else:
            past_states = self.replay_buffer.sample(len(self.replay_buffer))
            pred_traj = pred_traj + past_states

        # calculate the cks for the trajectory *** this is also in the utils file
        N = len(pred_traj)
        ck = np.sum([self.basis.fk(xt) for xt in pred_traj], axis=0) / N

        self.ck = ck.copy()
        ## publish CK
        ck_msg = Float32MultiArray()
        _send_arr = np.zeros(len(ck)+1)
        _send_arr[0] = self.agent_name
        _send_arr[1:] = ck.copy()
        ck_msg.data = _send_arr
        # TODO this might be wrong since ck is a np array
        self._ck_pub.publish(ck_msg)
        
        if len(self._ck_dict.keys()) > 1:
            self._ck_dict[self.agent_name] = ck
            ck_list = []
            for key in self._ck_dict.keys():
                ck_list.append(self._ck_dict[key])
            ck = np.mean(ck_list, axis=0)
            print('sharing and make sure first ck is 0 ', ck[0])

        fourier_diff = self.lamk * (ck - self.target_dist.phik)
        fourier_diff = fourier_diff.reshape(-1,1)

        # backwards pass
        rho = np.zeros(self.model.observation_space.shape[0])
        for t in reversed(range(self.horizon)):
            edx = np.zeros(self.model.observation_space.shape[0])
            edx[self.model.explr_idx] = np.sum(dfk[t] * fourier_diff, 0)

            bdx = np.zeros(self.model.observation_space.shape[0])
            bdx[self.model.explr_idx] = dbar[t]
            rho = rho - self.model.dt * (- edx - bdx - np.dot(fdx[t].T, rho))

            self.u_seq[t] = -np.dot(np.dot(self.Rinv, fdu[t].T), rho)
            if (np.abs(self.u_seq[t]) > 1.0).any():
                self.u_seq[t] /= np.linalg.norm(self.u_seq[t])

        self.replay_buffer.push(state[self.model.explr_idx].copy())
        return self.u_seq[0].copy()
