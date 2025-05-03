from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from config import USE_BACKEND, SYSTEM_PROMPT_VERSION
from llm.ollama_backend import OllamaChat
from llm.openai_backend import OpenAIChat
from core.preference_manager import PreferenceManager
from core.prompt_manager import PromptManager
from core.logger import Logger

# Session chat history
chat_history = []
# Session logger
logger = None

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
pref_mgr = PreferenceManager(llm_instance)


def get_session_id(user_info: dict) -> str:
    return f"{user_info['username']}"


def init_chat_history(user_info: dict):
    session_id = get_session_id(user_info)

    global chat_history, logger
    if not chat_history:
        SYSTEM_PROMPT = prompt_mgr.load("chat", version=SYSTEM_PROMPT_VERSION)
        chat_history.append(SystemMessage(content=SYSTEM_PROMPT))
        logger = Logger(session_id)


def reset_chat_history(user_info: dict):
    global chat_history

    # TO-DO: save chat history

    chat_history.clear()

    logger = None


def generate_chat_response(user_msg: str, user_info: dict) -> str:
    username = get_session_id(user_info)

    # Extract and update preferences, get combined preferences
    raw, new_prefs, preferences = pref_mgr.process(
        username, chat_history, user_msg)

    # Find any existing “User preferences:” message
    pref_msg_found = False
    for idx, msg in enumerate(chat_history):
        if isinstance(msg, SystemMessage) and msg.content.startswith("User preferences:"):
            chat_history[idx] = SystemMessage(
                content=f"User preferences: {preferences}"
            )
            pref_msg_found = True
            break
    if not pref_msg_found:
        # Insert it right after general SYSTEM_PROMPT
        chat_history.insert(1, SystemMessage(
            content=f"User preferences: {preferences}"
        ))

    chat_history.append(HumanMessage(content=user_msg))
    print(chat_history)
    response = llm_instance.generate_response(chat_history)

    chat_history.append(AIMessage(content=response))

    # Log this turn
    if logger:
        logger.log_turn({
            "user_message": user_msg,
            "raw_extraction": raw,
            "new_preferences": new_prefs,
            "combined_preferences": preferences,
            "ai_response": response
        })

    return response
