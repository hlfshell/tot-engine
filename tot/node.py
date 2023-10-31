from __future__ import annotations

from typing import List, Optional, Union
from uuid import uuid4

from tot.css import CSS

VALID = "valid"
INVALID = "invalid"


class Node:
    """
    Node is a class that encompasses:
        _id - a unique ID for the node for identification
        result - the salient string result generated by
            the LLM.
        reason - the expressed reasoning by the LLM to
            explain its result. Not strictly required in
            all instances, but useful for most complex
            tasks
        rating - a float value for its rating for weighting
            multiple nodes when branching
        children - a set of children Nodes that are branched
            from this Node
        completed - whether or not the Node is a terminal
            leaf node that completed the given task
    The goal is that nodes are an organized node that allows
    for multiple paths to be branched and compiled as the
    LLM slowly builds possible different paths off of
    existing nodes.
    """

    def __init__(
        self,
        result: str,
        reason: str = "",
        children: List[Node] = [],
        rating: Optional[float] = None,
        completed: bool = False,
        invalid: bool = False,
    ):
        self._id = uuid4()
        self.result = result
        self.reason = reason
        self.children = children
        self.rating = rating
        self.completed = completed
        self.invalid = invalid

    def clone(self) -> Node:
        clone = Node(
            self.result,
            self.reason,
            [child.clone() for child in self.children],
            self.rating,
            self.completed,
            self.invalid,
        )
        clone._id = self._id
        return clone

    def mark_completed(self, node: Node):
        """
        mark_completed will mark the target node and all parent
        nodes as completed if the child is contained within the
        given parent node.
        """
        chain = self.isolate_chain(node)
        if len(chain) == 0:
            raise ValueError("Node not found in chain")

        for node in chain:
            node.completed = True

    def isolate_chain(self, node: Node) -> List[Node]:
        """
        isolate_chain will return the chain of nodes to the desired
        node/node id as a list in order of their branching. If the
        node requested is not a leaf node, its children is not
        included. Note that these are references to the original
        nodes, so that changes to these will affect the originals.
        """
        nodes = []

        # First, we handle the case that this *is* the target node
        if self._id == node._id:
            nodes.append(self)
            return nodes

        # Then we go hunting for the node in question amongst our
        # children nodes
        for child in self.children:
            child_nodes = child.isolate_chain(node)
            if len(child_nodes) > 0:
                nodes.append(self)
                nodes += child_nodes
                return nodes

        # Finally, complete failure - this is a dead end
        return []

    def isolate_completed_chains(self) -> Node:
        """
        isolate_completed_chains will return a list of all
        completed chains within the given node, assuming
        that an exist. Note that this assumes that you've
        ran mark_completed on the parent node to mark nodes
        complete. This function is useful if the ToT engine
        is allowed to reach multiple possible completed
        states, allowing. Nodes are repeated via clone(),
        and can be represented multiple times depending on
        the branching of the completed paths.
        """
        if not self.completed:
            return []

        if len(self.children) == 0:
            return [self]

        completed_chains = []
        for child in self.children:
            if child.completed:
                clone = self.clone()
                isolated_child = child.isolate_completed_chains()
                clone.children = [isolated_child]
                completed_chains.append(clone)

        return completed_chains

    def find(self, id: str) -> Union[Node, None]:
        """
        find returns the node with given id if it exists
        within any possible chain of nodes from this node;
        it returns None otherwise
        """
        if self._id == id:
            return self
        for child in self.children:
            found = child.find(id)
            if found is not None:
                return found
        return None

    def __eq__(self, __value: object) -> bool:
        if __value is None:
            return False
        return self._id == __value._id

    def html(self, wrap_in_html: bool = True) -> str:
        """
        html creates an increasingly more nested
        collapsible html representation of a set of
        thoughts.

        If wrap_in_html is True, then the html is wrapped
        in a full html document with a style tag that
        cleans up the css for the html generated.
        """

        children_html = ""
        for child in self.children:
            children_html += child.html(wrap_in_html=False)

        html = f"""
            <details class="tot">
                <summary class="summary-node{" complete" if self.completed
                    else ""}{" invalid" if self.invalid
                    else ""}">{self.result} | {f'Rating: {self.rating}' if
                        self.rating is not None else ""}
                </summary>
                {children_html}
            </details>
        """

        if wrap_in_html:
            html = f"""
                <html>
                    <head>
                        <style>
                            {CSS}
                        </style>
                    </head>
                    <body>
                        <div id="wrapper">
                            {html}
                        </div>
                    </body>
                </html>
            """

        return html
