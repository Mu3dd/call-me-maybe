from typing import Any
from abc import ABC, abstractmethod


class ExecutingStage(ABC):
    """interface of json pipeline"""
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """must implement this function on all derived class

        Args:
            data: (Any): data can change between specific pipelines

        Returns:
            data: (Any): return changed data for next stage in pipeline
        """
        pass
