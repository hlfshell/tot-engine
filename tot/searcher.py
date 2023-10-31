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
    def next(node: Node) -> Union[Node, None]:
        """
        next returns the next node to be expanded upon
        based on the searcher's rules.

        If there are no more nodes to be expanded upon,
        then next returns None.

        This must be implemented by the implementing
        class.
        """
        pass
