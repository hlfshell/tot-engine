from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, wait
from time import time
from typing import List, Optional, Union

from retry import retry

from tot.node import Node
from tot.provider import Provider
from tot.searcher import Searcher

VALID = "valid"
INVALID = "invalid"
COMPLETED = "completed"


class TreeOfThoughts(ABC):
    """
    TreeOfThoughts is an object that is prepared to act as an reusable ToT
    prompting engine for a particular problem set. It is configured with a
    set of prompts, evaluation categories, and configuration settings to
    try to derive the desired output from the given LLM provider.



    """

    def __init__(
        self,
        provider: Provider,
        searcher: Searcher,
        evaluation_valid_or_invalid=False,
        evaluation_categories: List[str] = [],
        evaluation_category_scores: Optional[List[float]] = None,
        temperature: float = 0.7,
        next_step_fanout: int = 0,
        max_time: Optional[float] = None,
        max_steps: Optional[int] = None,
        max_workers: int = 5,
    ) -> None:
        """
        provider - An LLM provider
        searcher - Node search algorithm
        evaluation_valid_or_invalid - If True, the evaluation categories are
            limited to VALID, INVALID, and COMPLETED.
        evaluation_categories - A list of strings that are the categories the
            rating LLM may utilize. Only used if evaluation_valid_or_invalid
            is set to False. Required if evaluation_valid_or_invalid is set
            to False.
        evaluation_category_scores - Optional - a list of floats that are the
            scores associated with same-indexed categories in
            evaluation_categories. If not provided, the categories are evenly
            spaced scores from 0.0 to 1.0.
        temperature - The temperature to use when prompting the LLM
        next_step_fanout - The number of threads to use when prompting the LLM
            to generate children nodes. If set to <= 0, then it is assumed that
            when generating next nodes the LLM is generating multiple instead
            of a singular node.
        max_time - Optional - The maximum time in second to run the ToT engine
            when `execute` is called. If not provided, no timeout is utilized.
        max_steps - Optional - The maximum number of step attempts to generate
            when running execute. If set to None, no limit is utilized.
        max_workers - The maximum number of workers to use in the threadpool
            for asynchronous operations.
        """
        self.provider = provider
        self.searcher = searcher
        self._evaluation_is_valid_or_invalid = evaluation_valid_or_invalid
        self._evaluation_categories = evaluation_categories
        self._temperature = temperature
        self._next_step_fanout = next_step_fanout
        self.max_time = max_time
        self.max_steps = max_steps
        self.max_workers = max_workers
        self._per_step_timeout = 10.0
        self.total_timeout = 60.0

        # If the ToT is not limited to valid/invalid rating and aims to use
        # categorical classification, we must set the category scores per
        # the categories.
        if not self._evaluation_is_valid_or_invalid:
            # If the evaluation_category_scores is None, make it
            # a set of floats evenly spaced from 0 to 1 based on the
            # number of descriptor labels provided. IE if 3 labels
            # are provided, then the scores will be [0.0, 0.5, 1.0]
            if evaluation_category_scores is None:
                evaluation_category_scores = [
                    i / (len(evaluation_categories) - 1)
                    for i in range(0, len(evaluation_categories))
                ]
            if len(evaluation_categories) != len(evaluation_category_scores):
                raise ValueError(
                    "evaluation_categories and evaluation_category_scores must"
                    + " be the same length"
                )

            self._evaluation_category_scores = evaluation_category_scores

            # Ensure all the label categories are lower case only
            self._evaluation_categories = [
                label.lower() for label in evaluation_categories
            ]

        self.llm_threadpool = ThreadPoolExecutor(max_workers=max_workers)

    @abstractmethod
    def generate_node_prompt(self, steps: List[Node]) -> str:
        """
        generate_node_prompt takes the existing produced steps and
        returns a new prompt that is used to ask the LLM for the
        next node. Whether the ToT engine is expecting this to
        a response for multiple nodes or a singular node is determined
        by the _next_step_fanout setting; a value of <= 0 means that
        the ToT engine will generate all children nodes within a
        singular use of this prompt; higher values will create that
        many threads, asking the LLM to generate that many children
        nodes.

        Must be defined by the implementing class.
        """
        pass

    @abstractmethod
    def parse_generation_response(self, response: str) -> Union[Node, List[Node]]:
        """
        parse_generation_response takes the response from the LLM provider
        and parses it into a string that is the next step and its reasoning
        in the form of a Node. Note that this function could return either
        a singular Node or a list of Nodes depending on if the generation
        function is expected to create a singular next step or multiple
        at once. Whether or not the ToT engine is expecting multiple or
        singular is dependent upon the _next_step_fanout setting; a value
        of <= 0 means that the ToT engine will generate all children nodes
        within the same execution, whereas higher values will create that
        many threads, asking the LLM to generate that many children.

        Must be defined by the implementing class.
        """
        pass

    @abstractmethod
    def evaluate_step_prompt(self, step: Node) -> str:
        """
        evaluate_step_prompt takes a generated step and returns
        a new prompt that is used to ask the LLM to rate the step
        based on a set of predetermined categorical labels.

        Must be defined by the implementing class.
        """
        pass

    @abstractmethod
    def parse_evaluation_response(self, response: str) -> str:
        """
        parse_evaluation_response takes the response from the LLM's
        evaluation of a given step and returns a string that is the
        assigned categorical label for a given step.

        Note that there are two special cases for labels that,
        regardless of what assigned labels are, returning results
        in special behavior.

        1. invalid - the results of this node are somehow invalid
            and must be a terminal leaf node.
        2. completed - the results of this node fulfill the request
            and is to be marked as completed.

        Must be defined by the implementing class.
        """
        pass

    def rate_node(self, node: Node) -> float:
        """
        rate_node will request a prompt for a given step and request
        a labeled rating from the LLM provider. It will assign the
        score to the node. It will finally return the float value of
        the score.

        It accomplishes this by asking the ToT engine via
        `evaluate_step_prompt` to generate a prompt to ask the LLM
        provider to rate the given step. It then parses the response
        via the `parse_evaluation_response` function to get the
        resulting categorical rating.

        It can raise an error if the prompt fails after 3 retries
        or if the parsed label is not in the evaluation_categories.

        If the _evaluation_is_valid_or_invalid is set to True, then
        the ony possible expected values are "valid" and "invalid".
        In this case, the score will be 0.0 or 1.0 for each.

        If the _evaluation_is_valid_or_invalid is set to False, then
        the expected values are the _evaluation_categories. In this
        case, the score will be the associated score from the list
        of scores in _evaluation_category_scores.

        Regardless of which setting is used, an "invalid" label
        being returned will mark the node's `.invalid` property,
        preventing it from being considered at all when considering
        next steps.

        Returning a label of "completed" will set the `completed`
        flag on the node, regardless if it is an included category
        in the assigned labels.
        """
        rating_prompt = self.evaluate_step_prompt(node.result)
        rating_response = self.prompt(rating_prompt)
        label = self.parse_evaluation_response(rating_response)
        label = label.lower()

        # If the ToT engine is set to have either valid or invalid
        # labels only, then we limit ourselves to those values and
        # assign either 1.0 or 0.0. Otherwise we derive the resulting
        # score from the associated label with the rating.
        if self._evaluation_is_valid_or_invalid:
            if label == VALID:
                score = 1.0
            elif label == INVALID:
                node.invalid = True
                score = 0.0
            elif label == COMPLETED:
                node.completed = True
                score = 1.0
            else:
                raise ValueError(
                    f"Unknown label {label}: only valid or invalid is allowed"
                )

            node.score = score

            return score
        else:
            # Find the label if it exists within the category store
            label_index = self._evaluation_categories.index(label)
            if label_index == -1:
                raise ValueError(f"Unknown label {label}")

            # Handle special cases
            if label == INVALID:
                node.invalid = True
                score = 0.0
            elif label == COMPLETED:
                node.completed = True
                score = self.evaluate_category_scores[-1]

            # Map the score to the label
            score = self.evaluation_category_scores[label_index]

            node.score = score

            return score

    def rate_nodes(self, nodes: List[Node]) -> List[float]:
        """
        rate_nodes will, for a given list of nodes, asychronously
        call `rate_node` for each one given the ToT engine's
        threadpool. It will return a list of the scores for each
        node in the same order. If an exception is thrown the
        rating will be 0.0.
        """
        with self.llm_threadpool as executor:
            futures: List[Future] = []
            for node in nodes:
                future = executor.submit(self.rate_node, node)
                futures.append(future)

            wait(futures)

            scores = []

            # Skip over any exceptions; if it failed even
            # with retry we just skip it for now
            for future in futures:
                if future.exception() is not None:
                    scores.append(0.0)
                    continue

                scores.append(future.result())

            return scores

    def generate_children_nodes(self, node: Node) -> List[Node]:
        """
        generate_children_nodes will generate numerous children nodes
        (as per the `_next_step_fanout` setting) for a given node. It
        accomplishes this by asking the ToT engine to generate a prompt
        to generate the children nodes via `generate_step_prompt`. It
        then parses the response from the LLM via the
        `parse_generation_response`
        """
        prompt = self.generate_node_prompt(node)

        # We need to determine if we're going to fan out or utilize
        # a singular prompt generating steps.
        if self._next_step_fanout <= 0:
            response = self.prompt(prompt)
            nodes = self.parse_generation_response(response)
            return nodes
        else:
            nodes = []
            with self.llm_threadpool as executor:
                futures: List[Future] = []
                for _ in range(0, self._next_step_fanout):
                    future = executor.submit(self.prompt, prompt)
                    futures.append(future)

                # Wait for each future to finish - we can't
                # continue on a given thread until we have
                # scores for each anyway.
                wait(futures)

                # Skip over any exceptions; if it failed even
                # with retry we just skip it for now
                for future in futures:
                    if future.exception() is not None:
                        continue

                    node = self.parse_generation_response(future.result())
                    nodes.append(node)

            return nodes

    def multi_prompt(
        self,
        prompt: str,
        times: int = -1,
    ) -> List[Node]:
        """
        multi_prompt triggers the given prompt multiple times (the times
        parameter) at once utilizing max_workers workers in a threadpool.
        If times is -1, then the tree of thoughts default of
        self._next_step_fanout is used.
        """
        if times == -1:
            times = self._next_step_fanout

        with self.llm_threadpool as executor:
            futures: List[Future] = []
            for _ in range(0, times):
                future = executor.submit(self.prompt, prompt)
                futures.append(future)

            # Wait for each future to finish - we can't
            # continue on a given thread until we have
            # scores for each anyway.
            wait(futures)

            branches: List[Node] = []

            for future in futures:
                # Skip over any exceptions; if it failed even
                # with retry we just skip it for now
                if future.exception() is not None:
                    continue

                response = future.result()
                branches.append(Node(prompt, response))

            return branches

    @retry(tries=3)
    def prompt(self, prompt: str) -> str:
        """
        prompt will trigger a network call to the llm provider with a
        built in retry mechanism
        """
        return self.provider.prompt(prompt, self.temperature)

    def execute(self, task: str) -> Node:
        """
        execute runs the tree of thoughts engine as configured,
        producing (hopefully) a complete chain of thoughts via
        our Node class providing an answer to the question.
        """
        stop = False
        started_at = time()
        steps = 0

        root_thought = Node()
        current_thought: Union[Node, None] = None

        # Until we trigger a stop condition, we keep expanding
        # upon the tree of thoughts
        while not stop:
            # Check to see if we're taking too long to
            # generate our results; if so, break.
            if time() - started_at > self.total_timeout:
                stop = True
                break
            elif steps >= self.max_steps:
                stop = True
                break
            steps += 1

            # Pick the next target node to expand upon.
            # If it's the first time we're doing this, start
            # with the root node. Otherwise, use the searcher
            if len(root_thought.children) == 0:
                current_thought = root_thought
            else:
                current_thought = self.searcher.next(root_thought)
                # If the searcher returns None, we do not have
                # a valid path forward and we abort.
                if current_thought is None:
                    stop = True
                    break

            # Now that we have the next target thought, let's
            # generate the next set of nodes off of this thought
            nodes = self.generate_children_nodes(current_thought)

            # Set the nodes as children to the current_thought node
            current_thought.children = nodes

            # Rate each node
            self.rate_nodes(nodes)

            # Are any of the current nodes completed? If so, we can
            # stop.
            for node in nodes:
                if node.completed:
                    stop = True
                    root_thought.mark_completed(node)

        return root_thought
