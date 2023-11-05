from typing import List, Optional, Union

from examples.providers import Provider
from tot.node import Node
from tot.searcher import BFS, Dijkstra
from tot.treeofthoughts import TreeOfThoughts

GENERATE_PROMPT = open("examples/prompts/24/generate_node.prompt", "r").read()
EVALUATE_PROMPT = open("examples/prompts/24/evaluate_node.prompt", "r").read()


class TwentyFourToT(TreeOfThoughts):
    def __init__(self, provider: Provider, max_steps: Optional[int] = 30):
        super().__init__(
            provider,
            Dijkstra(depth_penalty=0.75),
            False,
            ["likely", "definite"],
            next_step_fanout=0,
            max_steps=max_steps,
        )

    def generate_node_prompt(self, nodes: List[Node]) -> str:
        """
        generate_node_prompt creates a prompt that takes out current state
        of the 24 puzzle and inserts it in for the next node generation.
        """
        # Get the latest state of the equation
        node = nodes[-1]
        equation = node.result

        return GENERATE_PROMPT.replace("{equation}", equation)

    def parse_generation_response(self, response: str) -> List[Node]:
        """
        parse_generation_response separates our different suggested
        outputs
        """
        # Each step is separated by new lines, so let's split on that
        steps = response.split("\n")

        # Then check and remove any lines that are empty (for possible LLM
        # introduced spacing)
        steps = [step for step in steps if step != ""]

        # Now separate out (left: ) sections from the answer
        reasons = []
        results = []
        for step in steps:
            result_and_reason = step.split("(left: ")
            if len(result_and_reason) > 1:
                result, reason = result_and_reason
                try:
                    paren_index = reason.index(")")
                    reason = reason[:paren_index]
                except Exception:
                    # ignore the exception here
                    pass
            else:
                result = result_and_reason[0]
                reason = ""

            reasons.append(reason)
            results.append(result)

        nodes = []

        for index in range(0, len(reasons)):
            nodes.append(Node(result=results[index], reason=reasons[index]))

        return nodes

    def evaluate_node_prompt(self, nodes: List[Node]) -> str:
        """ """
        # Grab the latest node
        node = nodes[-1]

        # Grab the remaining numbers from the reason
        remaining = node.reason

        return EVALUATE_PROMPT.replace("{numbers_left}", remaining)

    def parse_evaluation_response(self, response: str) -> str:
        """
        ???
        """
        # The expected response is two lines:
        # Reasoning: <text>
        # Rating: <label>
        # Let's extract each

        lines = response.split("\n")
        # Todo - record reasoning?
        # reasoning = response[0].replace("Reasoning: ", "")
        rating = lines[1].replace("Rating: ", "")

        # if the rating is impossible, mark it as invalid
        if rating == "impossible":
            rating = "invalid"

        return rating
