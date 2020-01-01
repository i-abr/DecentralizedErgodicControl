import numpy as np
import matplotlib.pyplot as plt
from sac import SAC
from settings import Settings
from system_dynamics import DoubleIntegrator, CartPendulum, Quadcopter
import time
from extended_kalman_filter import BearingOnlySensor, DiffusionProcess, EKF

np.set_printoptions(precision=5)

time_step = 0.05
robot = Quadcopter(time_step)
config = Settings(time_step=time_step)
controller = SAC(config)

sensor = BearingOnlySensor()
ekf = EKF(DiffusionProcess(0.01), sensor)

x0 = np.array([.2,.3,0.] + [0.0]*9)

u = []
x = [x0]
spotlight_x = []
elapsed_time = []
current_time = 0.0
final_time = 60.0
target_location = np.array([0.2,0.7,0.])
while current_time < final_time:
    # take measurements
    yk = sensor.h(target_location, x0[0:3])
    if yk is not None:
        yk += np.random.normal([0.]*2, np.diag(sensor.R))
    ekf.update(yk, x0[0:3])
    # print ekf.mean, ekf.covar, np.linalg.norm(ekf.covar)

    # config.eid.update_eid(current_time)
    config.eid.update_eid(ekf.mean, ekf.covar)
    # config.eid.update_eid(sensor.fisher_information_matrix, ekf.mean, ekf.covar, x0[0:3])
    # config.eid.update_eid_v2(sensor.fisher_information_matrix, ekf.mean, ekf.covar, x0[0:3])

    start = time.time()
    unow = controller(x0,config.eid.phik)
    end = time.time()
    elapsed_time.append(end-start)
    u.append(unow)
    x0 = robot.step(x0, unow)
    x.append(x0)

    print "Time : ", elapsed_time[-1], "\t State : ", x0[0:6], "\t Target at : ", target_location[0:2]
    current_time += time_step
    target_location[0] = (0.25 * np.cos(0.2 * current_time) + 0.5)
    target_location[1] = (0.25 * np.sin(0.2 * current_time) + 0.5)
    spotlight_x.append(target_location.copy())

# print controller.ck.cki
# print config.eid.phik
#
np.savetxt('robot_trajectory.csv', x)
np.savetxt('spotlight_x.csv', spotlight_x)
x = np.array(x)
plt.figure(1)
plt.scatter(x[:,0], x[:,1])
plt.figure(2)
plt.plot(u)
# plt.semilogy(elapsed_time)
plt.show()
