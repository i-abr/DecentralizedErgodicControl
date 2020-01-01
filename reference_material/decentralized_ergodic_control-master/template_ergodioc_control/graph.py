import numpy as np

class Graph(object):

    def __init__(self, nodes, edges, degree, n_coef):
        self.nodes = nodes
        self.edges = edges
        self.n_coef = n_coef
        self.D = np.diag(degree)
        self.A = np.zeros((nodes, nodes))
        for edge in edges:
            self.A[edge[0]-1,edge[1]-1] = 1 # assign the adjacency matrix
            self.A[edge[1]-1,edge[0]-1] = 1
        self.L = self.D - self.A # generate the laplacian
        self.P = np.eye(nodes) - 0.5 * np.linalg.inv(self.D).dot(self.L) # create the consensus
        self.P_ext = np.kron(self.P, np.eye(n_coef)) # extend the matrix to the number of parameters

    def update_consensus(self, x, iterations=None):
        """ Pass the graph values to update the estimate"""
        if iterations is None:
            x = self.P_ext.dot(x)
        else:
            for i in range(iterations):
                x = self.P_ext.dot(x)
        return x.reshape((self.nodes, self.n_coef))


if __name__ == '__main__':
    nodes = 3
    degree = [2,2,2]
    edges = [[1,2],[1,3],[2,1],[2,3],[3,2],[3,1]]
    gp = Graph(nodes, edges, degree, 1)
