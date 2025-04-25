from typing import List
from langchain_core.messages import HumanMessage

extraction_prompt = (
    "Analyze the entire conversation so far and extract any user preferences regarding explanation style. "
    "Consider whether the user prefers concise or detailed explanations, expert-level or simplified vocabulary, "
    "whether they like examples or not and any other possible preferences. Return the extracted preferences as a list. "
    "If no explicit preferences are found, please output 'No preferences found'."
)


def extract_preferences(llm_instance, chat_history):
    extraction_context = chat_history.copy()
    extraction_context.append(HumanMessage(content=extraction_prompt))

    # Call the LLM using extraction prompt together with dialogue context.
    preferences = llm_instance.generate_response(
        extraction_prompt, extraction_context)

    print("Extracted Preferences:", preferences)

    return preferences
