import numpy as np
from single_ergodic_quad.agent import QuadcopterAgent
import matplotlib.pyplot as plt
from graph import Graph

def main():

    num_agents = 10
    agents = [QuadcopterAgent() for _ in range(num_agents)]
    degree = [2]*num_agents
    edges = []
    for i in range(num_agents-1):
        edges.append([i+1, i+2])
        edges.append([i+2, i+1])
        if i == 0:
            edges.append([1, num_agents])
            edges.append([num_agents, 1])
    no_param = np.product(agents[0].objective.coefs+1)
    comm_link = Graph(num_agents, edges, degree, no_param)

    simulation_time = 0.0
    time_step = 0.05
    final_simulation_time = 15.0
    while simulation_time < final_simulation_time:

        plt.clf()
        ck_stack = None
        # Communication loop
        for agent in agents:
            if ck_stack is None:
                ck_stack = agent.ck.copy()
            else:
                ck_stack = np.hstack((ck_stack, agent.ck.copy()))

        ck_update = comm_link.update_consensus(ck_stack)
        for agent_num, agent in enumerate(agents):
            agent.control_step(cki=ck_update[agent_num])
            # agent.control_step()
            plt.plot(agent.trajectory[:,0], agent.trajectory[:,1])
        agents[0].objective.floor_plan.plot_ground()
        plt.axis('square')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pause(0.001)
        simulation_time += time_step
        print('\nTime : {}'.format(simulation_time))

    print('saving agent data...')
    for i,agent in enumerate(agents):
        agent.save('trial3/agent{}.npy'.format(i))

    print('saved data')
if __name__ == '__main__':
    main()
