from abc import ABCMeta, abstractmethod


class ModelSelectionBase(metaclass=ABCMeta):
    """Abstract base class for model selection"""

    def __init__(self, model: str) -> None:
        self.model = model
        self.workflow = None

    @abstractmethod
    def activate(self, *args, **kwargs) -> None:
        pass
