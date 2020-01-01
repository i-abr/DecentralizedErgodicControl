import numpy as np
import matplotlib.pyplot as plt
from robot_system import RobotSystem
from graph import Graph

def main():
    time_step = 0.05
    robots = [RobotSystem(time_step, 'pursuer', 0), RobotSystem(time_step, 'pursuer', 1)]
    target = RobotSystem(time_step, 'evader', 0)
    current_time = 0.0
    final_time = 20.0
    target_location = np.array([0.3,0.5])
    target_state_dump = []

    # make the graph
    num_robots = len(robots)
    degree = [2]*num_robots
    edges = []
    for i in range(num_robots-1):
        edges.append([i+1,i+2])
        edges.append([i+2,i+1])
        if i == 0:
            edges.append([1,num_robots])
            edges.append([num_robots,1])
    np.save('edges.npy',edges)
    # gp = Graph(num_robots, edges, degree, ncoef)
    ergodic_graph = Graph(num_robots, edges, degree, np.product(robots[0].config.coef+1))
    # ekf_graph = Graph(num_robots, edges, degree, 2)
    while current_time < final_time:
        # for each robot do stuff
        for robot in robots:
            yk = robot.sensor.h(target_location[0:2], robot.state[0:3])
            robot.step(yk)
            target_location[0] = (0.25 * np.cos(2*0.2 * current_time) + 0.5)
            target_location[1] = (0.25 * np.sin(3*0.2 * current_time) + 0.5)
            print "Time : ", current_time, "\t State : ", robot.state[0:3], "\t Target at : ", target_location[0:2]
        ck_stack = np.hstack(( robots[0].controller.ck.cki.ravel() , robots[1].controller.ck.cki.ravel() ))
        mu_stack = np.hstack(( robots[0].kalman_filter.mu.ravel(), robots[1].kalman_filter.mu.ravel() ))
        ck_update = ergodic_graph.update_consensus(ck_stack)
        # mu_update = ekf_graph.update_consensus(mu_stack)
        target_state_dump.append(target_location.copy())
        # for ii, robot in enumerate(robots):
            # robot.controller.ck.cki = ck_update[ii].reshape(robot.config.coef+1)
            # robot.kalman_filter.mu = mu_update[ii]
        current_time += time_step

    for i,robot in enumerate(robots):
        robot.save_data()

    np.savetxt('target_state_data.csv', target_state_dump)


if __name__ == '__main__':
    main()

'''
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
'''
