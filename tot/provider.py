from abc import ABC, abstractmethod
from threading import Lock
from typing import List


class Provider(ABC):
    """
    Provider is an abstract class to provide a simplified interface
    for various LLM api providers. The goal is to just present an
    object with a `prompt` function for quick usage.
    """

    def __init__(self):
        self._tokens_sent = 0
        self._tokens_returned = 0
        self._tokens_lock = Lock()

    def increment_tokens(self, sent: int, returned: int):
        """
        Thread-safe increment of sent and returned token
        counts for tracking costs/usage
        """
        with self._tokens_lock:
            self._tokens_sent += sent
            self._tokens_returned += returned

    def get_tokens_used(self) -> List[int]:
        """
        get_tokens_used returns the number of tokens used by the provider
        in the order of [sent, returned]
        """
        with self._tokens_lock:
            return [self._tokens_sent, self._tokens_returned]

    def reset_tokens_used(self):
        """
        reset_tokens_used resets the tokens used by the provider
        """
        with self._tokens_lock:
            self._tokens_sent = 0
            self._tokens_returned = 0

    @abstractmethod
    def prompt(self, prompt: str, temperature: float) -> str:
        pass
