import json
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from core.prompt_manager import PromptManager


def extract_preferences(llm_instance, chat_history: List[BaseMessage], last_user_message: str):
    # extraction_context = chat_history.copy()
    # extraction_context.append(HumanMessage(content=extraction_prompt))

    # # Call the LLM using extraction prompt together with dialogue context.
    # preferences = llm_instance.generate_response(
    #     extraction_prompt, extraction_context)

    # print("Extracted Preferences:", preferences)

    extraction_prompt = PromptManager().load("extract_prefs", version="v2")

    # Building a extraction context
    # Serialize chat_history as JSON, filtering out all SystemMessage entries so we don’t mistake them for user turns.
    serialized = []
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            role = "system"
            continue  # no system messages in extraction context
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "unknown"
        serialized.append({"role": role, "content": msg.content})

    extraction_context = (
        "chat_history:\n"
        f"{json.dumps(serialized, ensure_ascii=False, indent=2)}\n\n"
        f"last_user_message: {last_user_message}"
    )

    # Combine extraction prompt with context data
    extraction_msg = [
        HumanMessage(content=f"{extraction_prompt}\n\n{extraction_context}")
    ]
    print("EXTRACTION PROMT: \n", extraction_msg[0], "\n")

    # Send message
    raw = llm_instance.generate_response(extraction_msg)
    print("EXTRACTOR RESPONSE: \n", raw)

    preferences = ""
    return preferences
