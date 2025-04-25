from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, MODEL_TEMPERATURE
from llm.base import BaseLLM


class OpenAIChat(BaseLLM):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
            temperature=MODEL_TEMPERATURE
        )

    def generate_response(self, message: str, messages: list[BaseMessage]) -> str:
        response = self.llm.invoke(messages)

        return response.content
