from abc import ABC, abstractmethod
from collections import defaultdict
import math
import networkx as nx
import numpy as np
import os
import random
from matplotlib import pyplot as plt

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

#generate random graph of preset size
def generate_graph(vertices):
    return nx.generators.random_graphs.gnp_random_graph(vertices,
                np.random.uniform(0,0.5))

class VertexCoverInstance(Node):
    def __init__(self, graph, cover = [], reward = 0):
        self.graph = graph
        self.cover = cover
        self.reward = reward
        self.possible_moves = [node for node in list(graph.nodes) if graph.degree[node]]

    def find_children(self):
        possiblemoves = []
        possiblenodes = []
        if self.is_terminal():  # If the game is finished then no moves can be made
            return possiblemoves
        for i in list(self.graph.nodes):
            H = self.graph.copy()
            if H.degree[i]:
                step_reward = -1
                H.remove_node(i)
                H.add_node(i)
                possiblenodes.append(i)
                possiblemoves.append(VertexCoverInstance(H, self.cover+[i], self.reward+step_reward))
        return possiblemoves

    def take_action(self, action):
        if self.is_terminal():
            raise IndexError("Terminal state cannot be acted upon")
        if action not in self.possible_moves:
            raise IndexError("This action is illegal")
        H = self.graph.copy()
        step_reward = -1
        H.remove_node(action)
        H.add_node(action)
        return VertexCoverInstance(H, self.cover+[action], self.reward+step_reward)

    def find_random_child(self):
        if self.is_terminal():
            return None  # If the game is finished then no moves can be made
        temp = self.find_children()
        return random.sample(set(temp),1)[0]

#     def reward(board):
#         if not board.terminal:
#             raise RuntimeError(f"reward called on nonterminal board {board}")
#         return self.reward #reward comes upon reaching terminal state

    def is_terminal(self):
        return nx.classes.function.is_empty(self.graph)

    def to_pretty_string(self):
        return str(list(self.graph.nodes()))

    def get_cover(self):
        return self.cover

    def get_reward(self):
        return self.reward

class VertexCoverAddRemove(Node):
    def __init__(self, graph, cover = [], reward = 0):
        self.init_graph = graph
        self.cover = cover
        self.reward = reward
        self.size = self.init_graph.number_of_nodes()

        self.graph = self.init_graph.remove_nodes_from(cover).add_nodes_from(cover)
        self.possible_deletions = [node for node in list(self.graph.nodes) if self.graph.degree[node]]
        self.possible_additions = [node+self.init_graph.number_of_nodes() for node in self.cover]
        self.possible_moves = list(self.possible_deletions+self.possible_additions)


    def find_children(self):
        if self.is_terminal():  # If the game is finished then no moves can be made
            return []
        return [self.take_action(i) for i in self.possible_moves]

    def take_action(self, action):
        if self.is_terminal():
            raise IndexError("Terminal state cannot be acted upon")
        if action not in self.possible_moves:
            raise IndexError("This action is illegal")
        if action >= self.size:
            return VertexCoverAddRemove(self.init_graph, self.cover-[add_action-self.size], self.reward+1)
        return VertexCoverAddRemove(self.init_graph, self.cover+[action], self.reward-1)

    def find_random_child(self):
        if self.is_terminal():
            return None  # If the game is finished then no moves can be made
        temp = self.find_children()
        return random.sample(set(temp),1)[0]


    def is_terminal(self): #30 percent chance of terminal if condition is satisfied (change for exploitation weight)
        prob = 0
        if nx.classes.function.is_empty(self.graph):
            prob = np.random.random_sample()
        if prob > 0.7:
            return True
        return False

    def get_cover(self):
        return self.cover

    def get_reward(self):
        return self.reward
