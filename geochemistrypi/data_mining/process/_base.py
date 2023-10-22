from abc import ABCMeta, abstractmethod


class ModelSelectionBase(metaclass=ABCMeta):
    """Abstract base class for model selection"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.workflow = None
        self.transformer_config = {}

    @abstractmethod
    def activate(self, *args, **kwargs) -> None:
        pass
