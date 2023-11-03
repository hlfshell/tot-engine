from typing import List, Optional

from sudoku import Sudoku

from examples.providers import Provider
from tot.node import Node
from tot.searcher import BFS
from tot.treeofthoughts import TreeOfThoughts


GENERATE_PROMPT = open("examples/prompts/sudoku/generate_node.prompt", "r").read()


class SudokuToT(TreeOfThoughts):
    def __init__(self, provider: Provider, max_steps: Optional[int] = 30):
        super().__init__(
            provider,
            BFS(),
            False,
            ["bad", "okay", "great"],
            next_step_fanout=5,
            max_steps=max_steps,
        )

    def generate_node_prompt(self, nodes: List[Node]) -> str:
        """
        generate_node_prompt creates a prompt that takes out current state
        of the sudoku board and insert it in for next step generation.
        """
        # Get the latest node only
        node = nodes[-1]

        # From this extract the step as that's the board
        board = node.result

        # Generate the prompt replacing {board} with the board
        return GENERATE_PROMPT.replace("{board}", board)

    def parse_generation_response(self, response: str) -> Node | List[Node]:
        """
        parse_generation_response separates our reasoning and our resulting
        sudoku board after the move is made, saving it as a string.
        """

        # Isolate the text before the "+--" string representing the sudoku board
        # and after the "Reasoning:" string
        end_at = response.index("+--")
        start_at = response.index("Reasoning: ")
        reasoning = response[start_at:end_at]

        # Trim excess spacing and new lines
        reasoning = reasoning.strip()

        # Now isolate the sudoku board - everything since the "+--" string
        board = response[end_at:]

        # Create a node with the reasoning and response
        node = Node(board, reasoning)

        return node

    def rate_node(self, node: Node) -> float:
        """
        rate_node overrides the LLM prompting version since we can do our
        evaluation entirely without the LLM. rate_node will rate the given
        node, returning a set value of its "score". It will also apply the
        score upon calculation.

        It will appropriately mark nodes as invalid or completed.
        """
        board = self.parse_sudoku(node.result)

        valid = self.validate_puzzle(board)

        # If our board is invalid, mark it as such and return a 0.0
        if not valid:
            node.invalid = True
            return 0.0

        # Is our board complete? If we're valid and complete we have
        # successfully finished the puzzle
        if self.is_puzzle_complete(board):
            node.completed = True
            return 1.0

        return 1.0

    def evaluate_node_prompt(self, nodes: List[Node]) -> str:
        return ""

    def parse_evaluation_response(self, response: str) -> str:
        return ""

    def parse_sudoku(self, board: str) -> List[List[int]]:
        """
        Given a sudoku response formatted in the manner expected, such as:

        +-------+-------+-------+
        | 3   9 | 5 1   | 4   6 |
        |       |   8   |       |
        |   2   | 6     |   9 1 |
        +-------+-------+-------+
        |   8 3 |   2   |     4 |
        |       |   4   | 3   8 |
        | 2 9   | 1 3   |     5 |
        +-------+-------+-------+
        |   3 5 | 2 6   |   4 9 |
        | 9 4 2 |   5   | 6 3   |
        | 1 6   |   9 4 | 8 5 2 |
        +-------+-------+-------+

        ... convert it to an array of ints, where 0 represents
        empty space, such as:
        [
            [3,0,9,5,1,0,4,0,6],
            [0,0,0,0,8,0,0,0,0],
            [0,2,0,6,0,0,0,9,1],
            [0,8,3,0,2,0,0,0,4],
            [0,0,0,0,4,0,3,0,8],
            [2,9,0,1,3,0,0,0,5],
            [0,3,5,2,6,0,0,4,9],
            [9,4,2,0,5,0,6,3,0],
            [1,6,0,0,9,4,8,5,2],
        ]

        Note this only works for a typical 9x9 grid of Sudoku.
        """
        # Find first occurrence of '+' and remove everything prior
        start_at = board.index("+")
        board = board[start_at:]

        # Remove the first and last line
        board = board.split("\n")[1:-1]  # remove first and last line
        # Remove the 3rd and 7th line as they're spacers
        board = board[:3] + board[4:7] + board[8:]

        # Move through each line. Split by the "|" character.
        # We expect the pattern of " ""#" for each. Note that
        # if the "#" is also a space, we replace it with "0"
        for index, line in enumerate(board):
            new_line = ""
            sections = line.split("|")
            sections = sections[1:-1]

            for section in sections:
                while len(section) > 1:
                    # drop the first char - its a space
                    section = section[1:]
                    # pop the next char
                    char = section[0]
                    section = section[1:]
                    if char == " ":
                        char = "0"
                    new_line += char
            board[index] = new_line

        board = [
            [int(n) if n else 0 for seg in line for n in seg] for line in board
        ]  # convert to int

        # Sometimes the final line is a blank space and makes it to this point
        # untouched - check for empty rows and remove them.
        for index, row in enumerate(board):
            if len(row) == 0:
                board.pop(index)

        return board

    def is_puzzle_complete(self, board: List[List[int]]) -> bool:
        """
        is_puzzle_complete returns whether or not a puzzle is completed
        but not necessarily if it is correct.
        """
        for row in board:
            for value in row:
                if value == 0:
                    return False
        return True

    def validate_puzzle(self, board: List[List[int]]) -> bool:
        """
        validate_puzzle accepts a sudoku board in the form of a list of
        lists of ints, and returns True if the puzzle is in a valid form
        per the rules of sudoku, False otherwise. Note that a move can be
        valid even if it does not move us closer to solving.
        """
        return Sudoku(3, 3, board=board).validate()

    def is_puzzle_solved(self, board: List[List[int]]) -> bool:
        """
        is_puzzle_solved returns whether or not a puzzle is correctly
        solved
        """
        try:
            Sudoku(3, 3, board=board).solve(raising=True)
            return True
        except Exception:
            return False
