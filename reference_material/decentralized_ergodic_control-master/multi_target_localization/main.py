import numpy as np
from terrain.city_block import g_city_terrain
from agent.agent import Agent
from target.target import Target
import matplotlib.pyplot as plt
from graph import Graph
# from ergodic.city_block import g_city_terrain
import sys, os

class World(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.no_agents = 3
        self.no_targets = 4
        self.agents = [Agent(self.time_step, self.no_targets) for i in range(self.no_agents)]
        positions = [[0.15, 0.6], [0.45, 0.4], [0.7, 0.6], [0.75,0.8]]
        self.targets = [Target(self.time_step, position=positions[i]) for i in range(self.no_targets)]

        self.ck_comm_link, self.mu_comm_link = self._makeCommunicationLink()
        self.agents[0].phik.update_eid(self.agents[0].state[0:3], self.targets)

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

    def world_step(self): # Step the world forward

        ck_stack = None
        mean_stacks = [None for target_no in range(self.no_targets)]
        sigma_stacks = [None for target_no in range(self.no_targets)]
        # for target in self.targets: # step the targets forward in time
        #     for agent in self.agents: # TODO: add capabilities to differentiate other agents for kalman filter
        #         target.updateBelief(agent.state) # update agent
        #     print('Target state : {}'.format(target.state))
        #
        #     if ck_stack is None: # collect the ck values
        #         ck_stack = target.controller.getCKFromDefaultControl(target.state).copy()
        #     else:
        #         ck_stack = np.hstack(( ck_stack, target.controller.getCKFromDefaultControl(target.state).copy() ))
        for i,agent in enumerate(self.agents): # For each agent...
            agent.phik.update_eid(agent.state[0:3], self.targets)

            if ck_stack is None:
                # ck_stack = agent.controller.getCKFromDefaultControl(agent.state).copy()
                ck_stack = agent.controller.ck.cki.copy()
            else:
                ck_stack = np.hstack((
                            ck_stack,
                            agent.controller.ck.cki.copy()
                            #agent.controller.getCKFromDefaultControl(agent.state).copy()
                ))

            for mean_i, mean_stack in enumerate(mean_stacks):
                if mean_stack is None:
                    mean_stack = agent.phik.ekfs[mean_i].mu.copy()
                    sigma_stacks[mean_i] = np.diag(agent.phik.ekfs[mean_i].sigma.copy())
                else:
                    mean_stack = np.hstack(( mean_stack, agent.phik.ekfs[mean_i].mu.copy() ))
                    sigma_stacks[mean_i] = np.hstack((
                            sigma_stacks[mean_i],
                            np.diag(agent.phik.ekfs[mean_i].sigma.copy())
                    ))
                mean_stacks[mean_i] = mean_stack
                # sigma_stacks[mean_i] = sigma_stacks[] # don't need this line
            print('Agent {} state : {}'.format(i, agent.state[0:3])) # print information

        ###### communication loop #####
        ck_update = self.ck_comm_link.update_consensus(ck_stack)
        mean_updates = []
        sigma_updates = []
        for i,mean_stack in enumerate(mean_stacks):
            mean_updates.append( self.mu_comm_link.update_consensus(mean_stack) )
            sigma_updates.append( self.mu_comm_link.update_consensus(sigma_stacks[i]))
        for target_no in range(self.no_targets):
            for i,agent in enumerate(self.agents):
                agent.phik.ekfs[target_no].mu = mean_updates[target_no][i]
                agent.phik.ekfs[target_no].updateSigma(np.diag(sigma_updates[target_no][i]))

        for i,agent in enumerate(self.agents):
            agent.controller.ck.cki = ck_update[i].copy()
            # agent.step(ck_update[i].copy()) # step the target
            agent.step()


    def draw(self, ax):
        g_city_terrain.plotCity(ax)
        for agent in self.agents:
            agent.plotTrajectory(ax)
        for target in self.targets:
            target.plotTrajectory(ax)

    def save_agent_data(self, dirPath):
        import os
        import errno
        for i,agent in enumerate(self.agents):
            filePath = dirPath + 'agent{}/'.format(i)
            try:
                os.makedirs(filePath)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            agent.save_data(dirPath + 'agent{}/'.format(i))

        for i, target in enumerate(self.targets):
            filePath = dirPath + 'target{}/'.format(i)
            try:
                os.makedirs(filePath)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            target.save_data(dirPath + 'target{}/'.format(i))
def main():
    world = World(0.05) # instantiate a world with time step 0.05
    tf = int(20/0.05)
    tic = 0
    toc = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.ion()
    for i in range(tf):
        world.world_step()
        if tic > toc:
            # plt.clf()
            ax.clear()
            world.draw(ax)
            plt.draw()
            plt.pause(0.01)
            tic = 0

        tic += 1
        # print('Step : {} out of {}'.format(i, tf))
        # os.system('cls' if os.name == 'nt' else 'clear')
    filePath = 'multi_target_local2/'
    world.save_agent_data(filePath)
    g_city_terrain.save_data(filePath)
if __name__ == '__main__':
    main()
