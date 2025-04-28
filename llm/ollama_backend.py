from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
from config import MODEL_NAME, OLLAMA_URL
from llm.base import BaseLLM


class OllamaChat(BaseLLM):
    def __init__(self):
        self.llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_URL)

    def generate_response(self, messages: list[BaseMessage]) -> str:
        response = self.llm.invoke(messages)

        return response
