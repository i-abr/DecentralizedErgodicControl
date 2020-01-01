import numpy as np
from agent import Agent
from target import Target
import matplotlib.pyplot as plt
from graph import Graph

class World(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.no_agents = 3
        self.no_targets = 1
        self.agents = [Agent(self.time_step) for i in range(self.no_agents)]
        self.targets = [Target(self.time_step)] * self.no_targets
        self.ck_comm_link, self.target_comm_link = self._makeCommunicationLink()

    def _makeCommunicationLink(self):
        degree = [2]*self.no_agents
        edges = []
        for i in range(self.no_agents-1):
            edges.append([i+1, i+2])
            edges.append([i+2, i+1])
            if i == 0:
                edges.append([1, self.no_agents])
                edges.append([self.no_agents, 1])
        no_param = np.product(self.agents[0].settings.coef+1)
        no_target_param = 2
        return Graph(self.no_agents, edges, degree, no_param), Graph(self.no_agents, edges, degree, no_target_param)

    def world_step(self):

        ck_stack = None
        mean_stack = None
        for i,agent in enumerate(self.agents):
            for target in self.targets:
                yk = agent.sensor.h(target.state, agent.state[0:3])
                agent.updateBelief(yk)
                target.step() # step the target
            agent.step()
            if ck_stack is None:
                ck_stack = agent.controller.ck.cki.copy()
            else:
                ck_stack = np.hstack(( ck_stack, agent.controller.ck.cki.copy() ))
            if mean_stack is None:
                mean_stack = agent.kalman_filter.mu
            else:
                mean_stack = np.hstack((mean_stack, agent.kalman_filter.mu))

        print('Agent {} state : {} \t Agent {} state : {}'.format(i, self.agents[0].state[0:3], i, self.agents[1].state[0:3]))
        print('Target state : {}, Mean state : {}, {}'.format(self.targets[0].state, self.agents[0].kalman_filter.mu, self.agents[1].kalman_filter.mu))
        # communication loop
        ck_update = self.ck_comm_link.update_consensus(ck_stack)
        mean_update = self.target_comm_link.update_consensus(mean_stack)

        for i,agent in enumerate(self.agents):
            agent.controller.ck.cki = ck_update[i].copy()
            agent.kalman_filter.mu = mean_update[i].copy()

    def save_agent_data(self):
        import os
        import errno
        for i,agent in enumerate(self.agents):
            filePath = 'agent{}/'.format(i)
            try:
                os.makedirs(filePath)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            agent.save_data('agent{}/'.format(i))
def main():
    world = World(0.05) # instantiate a world with time step 0.05
    tf = int(30/0.05)
    for i in range(tf):
        world.world_step()
        print('Step : {} out of {}'.format(i, tf))
    world.save_agent_data()
if __name__ == '__main__':
    main()
