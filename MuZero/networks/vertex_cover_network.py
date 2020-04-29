import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling
from torch_geometric.nn.models import Node2Vec

from game.game import Action
from networks.network import BaseNetwork


class VertexCoverNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(-max_value)) + 1

        class RepNet(torch.nn.Module):
            def __init__(self):
                super(RepNet, self).__init__()
                self.conv1 = SAGEConv(1, 16)
                self.pool = TopKPooling(16, ratio=0.8)
                self.conv2 = SAGEConv(16, 1)
                self.flat = torch.nn.Flatten()

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index.long())
                x = F.relu(x)
                #x, edge_index, _, batch, _, _ = self.pool(x, edge_index.long())
                #x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index.long())
                x = self.flat(x).reshape(1,-1)
                return torch.tanh(x)
                return x

        class RewardNet(torch.nn.Module):
            def __init__(self, insize, hidden_neurons):
                super(RewardNet, self).__init__()
                self.l1 = torch.nn.Linear(insize, hidden_neurons)
                self.l2 = nn.Linear(hidden_neurons, 1)

            def forward(self, data):
                x = self.l1(data)
                x = F.relu(x)
                x = self.l2(x)
                return -1*torch.sigmoid(x)

        class ValueNet(torch.nn.Module):
            def __init__(self, insize, hidden_neurons):
                super(ValueNet, self).__init__()
                self.l1 = torch.nn.Linear(insize, hidden_neurons)
                self.l2 = nn.Linear(hidden_neurons, 1)

            def forward(self, data):
                x = self.l1(data)
                x = F.relu(x)
                x = self.l2(x)
                return -10*torch.sigmoid(x)

        """ function h from the MuZero paper """
        representation_network = RepNet()

        """ function f from the MuZero paper """
        value_network = ValueNet(representation_size, hidden_neurons)
        policy_network = nn.Sequential(nn.Linear(representation_size, hidden_neurons), nn.ReLU(),
                            nn.Linear(hidden_neurons, self.action_size))

        """ function g from the MuZero paper """
        dynamic_network = nn.Sequential(nn.Linear(representation_size+self.action_size, hidden_neurons), nn.ReLU(),
                            nn.Linear(hidden_neurons, representation_size), nn.Tanh())
        reward_network = RewardNet(representation_size+self.action_size, hidden_neurons)


        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return -value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)
