import networkx as nx
import numpy as np
import os
import random

GRAPH_SIZE = 10
DATASET_SIZE = 1000

#generate set of random graphs of preset size
def generate_graph_dataset(vertices, n):
    data = np.array([nx.convert_matrix.to_numpy_array(nx.generators.random_graphs.gnp_random_graph(vertices,
                np.random.uniform(0,1))) for x in range(n)])
    print('Dataset of size {} generated'.format(n))
    return data


if __name__ == '__main__':
    #set seed for reproduceability
    random.seed(57)
    np.random.seed(1234)
    
    data = generate_graph_dataset(GRAPH_SIZE, DATASET_SIZE)
    np.save('graph_data', data)
