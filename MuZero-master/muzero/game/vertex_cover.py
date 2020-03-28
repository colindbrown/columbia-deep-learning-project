from typing import List
import networkx as nx
import gym

from game.game import Action, AbstractGame

import torch
from torch_geometric import utils
from torch_geometric.data import Data


def to_pytorch(G):
    g = utils.from_networkx(G)
    g.x = torch.tensor([[1] for x in range(g.num_nodes)], dtype=torch.float)
    return g

def to_nx(G):
    return utils.to_networkx(G, to_undirected=True)


class VertexCover(AbstractGame):

    def __init__(self, vertices = 10, discount: float):
        super().__init__(discount)
        init_graph = nx.generators.random_graphs.gnp_random_graph(vertices,
np.random.uniform(0,0.5))
        self.actions = list(init_graph.nodes())
        self.env = to_pytorch(init_graph)
        self.observations = [self.env]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        new_obs = to_nx(self.env).remove_node(action.index)
        new_obs.add_node(action.index) #now graph is not dynamic - can use GNN, each action only removes edges
        self.actions = list(new_obs.nodes())
        self.done = nx.classes.function.is_empty(new_obs)
        self.env = to_pytorch(new_obs)
        self.observations += [self.env]
        return -1

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]
