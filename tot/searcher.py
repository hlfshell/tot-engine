from abc import ABC, abstractmethod
from typing import Union

from tot.node import Node


class Searcher(ABC):
    """
    Searcher is a class that, given a parent node, will
    identify the next candidate node to be expanded upon
    based on some predetermined set of rules, such as
    rating, valid/invalid checks, and completeness, etc.

    Keywords that an implementation is expected to observe:

    1. completed - the Node is a terminal leaf node and
        should not be expanded upon
    2. invalid - the Node is marked invalid for the use
        case and should not be expanded upon
    """

    @abstractmethod
    def next(self, node: Node) -> Union[Node, None]:
        """
        next returns the next node to be expanded upon
        based on the searcher's rules.

        If there are no more nodes to be expanded upon,
        then next returns None.

        This must be implemented by the implementing
        class.
        """
        pass


class BFS(Searcher):
    def __init__(self) -> None:
        super().__init__()

        self.queue = []

    def _add_to_queue(self, node: Node) -> None:
        """
        _add_to_queue will add a node to the queue if it
        is not already in the queue.
        """
        if node in self.queue:
            return

        if node.invalid or node.completed:
            return

        if len(node.children) == 0:
            self.queue.append(node)
            return

        for child in node.children:
            self._add_to_queue(child)

    def next(self, parent_node: Node) -> Node | None:
        """
        next will take a parent node, and identify the next node to select
        based on a BFS check; specifically; the first node that can be
        expanded will be. Invalid and completed nodes are ignored. Nodes
        with children are ignored.
        """
        self._add_to_queue(parent_node)

        if len(self.queue) == 0:
            return None

        return self.queue.pop(0)
