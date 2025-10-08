from abc import ABC, abstractmethod

class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, max_length: int=100, min_length: int=30) -> str:
        """Generate a summary for the given text."""
        pass