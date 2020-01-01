import numpy as np
from single_ergodic_quad.agent import QuadcopterAgent
import matplotlib.pyplot as plt
from graph import Graph

np.random.seed(1887)

def main():

    num_agents = 4
    agents = [QuadcopterAgent() for _ in range(num_agents)]
    # for agent in agents:
    #     agent.objective.target_distribution = agents[0].objective.target_distribution
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
    final_simulation_time = 30.0

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
            # plt.plot(agent.trajectory[:,0], agent.trajectory[:,1])
            # agent.objective.target_distribution.plot()
        # agents[0].objective.floor_plan.plot_ground()
        # agents[0].objective.target_distribution.plot()
        #
        plt.axis('square')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pause(0.001)
        simulation_time += time_step

        phik_stack = None
        for agent in agents:
            if phik_stack is None:
                phik_stack = agent.objective.target_distribution.update_phik(agent.state,simulation_time)
            else:
                phik_temp = agent.objective.target_distribution.update_phik(agent.state,simulation_time)
                phik_stack = np.hstack((phik_stack, phik_temp.copy()))

        phik_update = comm_link.update_consensus(phik_stack)
        for agent_num, agent in enumerate(agents):
            agent.objective.target_distribution.phik = phik_update[agent_num]

        print('\nTime : {}'.format(simulation_time))

    print('saving agent data...')
    fpath = 'trial6/'
    for i,agent in enumerate(agents):
        agent.save(fpath + 'agent{}.npy'.format(i))
        agent.objective.target_distribution.save(fpath + 'agent{}_particles.npy'.format(i))
    print('saved data')
if __name__ == '__main__':
    main()
