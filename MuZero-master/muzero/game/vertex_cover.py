from typing import List
import networkx as nx
import gym

from game.game import Action, AbstractGame


class VertexCover(AbstractGame):
    """The Gym CartPole environment"""

    def __init__(self, vertices = 10, discount: float):
        super().__init__(discount)
        self.env = nx.generators.random_graphs.gnp_random_graph(vertices,
np.random.uniform(0,0.5))
        self.actions = list(self.env.nodes())
        self.observations = [self.env]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        self.env.remove_node(action.index)
        self.actions = list(self.env.nodes())
        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [self.env]
        self.done = nx.classes.function.is_empty(self.env)
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
