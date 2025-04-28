from abc import ABC, abstractmethod
from typing import List
from langchain_core.messages import BaseMessage


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, messages: List[BaseMessage]) -> str:
        pass
