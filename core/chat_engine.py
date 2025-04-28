from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from config import USE_BACKEND, SYSTEM_PROMPT_VERSION
from llm.ollama_backend import OllamaChat
from llm.openai_backend import OpenAIChat
from core.preference_extractor import extract_preferences
from core.prompt_manager import PromptManager

# Session chat history
chat_history = []


# LLM backend selection
def get_llm():
    if USE_BACKEND == "ollama":
        return OllamaChat()
    elif USE_BACKEND == "openai":
        return OpenAIChat()
    else:
        raise ValueError("Unsupported backend")


llm_instance = get_llm()
prompt_mgr = PromptManager()


def get_session_id(user_info: dict) -> str:
    return f"{user_info['username']}"


def init_chat_history(user_info: dict):
    session_id = get_session_id(user_info)

    global chat_history
    if not chat_history:
        SYSTEM_PROMPT = prompt_mgr.load("chat", version=SYSTEM_PROMPT_VERSION)
        chat_history.append(SystemMessage(content=SYSTEM_PROMPT))


def reset_chat_history(user_info: dict):
    global chat_history

    # TO-DO: save chat history

    chat_history.clear()


def generate_chat_response(message: str, user_info: dict) -> str:
    # Extract preferences
    preferences = extract_preferences(llm_instance, chat_history, message)

    chat_history.append(HumanMessage(content=message))
    response = llm_instance.generate_response(chat_history)

    chat_history.append(AIMessage(content=response))

    return response
