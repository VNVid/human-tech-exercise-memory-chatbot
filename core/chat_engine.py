import re
import json
from typing import List, Tuple
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from config import USE_BACKEND, SYSTEM_PROMPT_VERSION, EXTRACT_PREF_PROMPT_VERSION, MERGE_PREF_PROMPT_VERSION
from llm.ollama_backend import OllamaChat
from llm.openai_backend import OpenAIChat
from core.preference_manager import PreferenceManager
from core.prompt_manager import PromptManager
from core.logger import Logger
from core.picture_agent import PictureAgent
from core.exercise_db import ExerciseDB


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
# Holds session-specific state (IDs), cleared when the session ends.
picture_agent = PictureAgent(llm_instance)


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

    picture_agent.clear_state()


def merge_system_messages(chat_history: List[BaseMessage]) -> List[BaseMessage]:
    """
    Merges the block of SystemMessage(s) at the beginning of 'chat_history'
    into a single SystemMessage (order preserved).

    Returns new list [merged_system_msg, ...rest...].
    """
    if not chat_history or not isinstance(chat_history[0], SystemMessage):
        return chat_history  # nothing to do

    buffer = []
    idx = 0
    while idx < len(chat_history) and isinstance(chat_history[idx], SystemMessage):
        buffer.append(chat_history[idx].content.strip())
        idx += 1

    merged = SystemMessage(content="\n\n".join(buffer))

    # Return new list
    return [merged, *chat_history[idx:]]


def split_llm_sections(raw: str) -> Tuple[str, List[int]]:
    """
    Splits the raw LLM reply into (textual_response, [img_ids]) based on these markers:

        Textual response:
        <arbitrary text>
        Images: 
        [id1, id2, ...]

    If the pattern isn't matched or JSON fails, returns (raw.strip(), []).
    """
    # Find the "Images:" section (any prefix, then Images:, then [ ... ])
    img_pattern = re.compile(r"Images:\s*(\[.*\])", re.DOTALL)
    m_img = img_pattern.search(raw)

    if m_img:
        images_json = m_img.group(1).strip()
        im_sect_start = m_img.start()
    else:
        images_json = ""
        im_sect_start = len(raw)

    # Parse the JSON array of IDs
    img_ids = []
    if images_json:
        try:
            parsed = json.loads(images_json)
            if isinstance(parsed, list):
                img_ids = [int(x) for x in parsed]
            else:
                img_ids = []
        except json.JSONDecodeError:
            img_ids = []

    # Take everything before the Images section as the textual slice
    textual_slice = raw[:im_sect_start].strip()

    # Extract the "Textual response:" part
    text_pattern = re.compile(r"Textual\s+response:\s*(.*?)\s*$", re.DOTALL)
    m_text = text_pattern.search(textual_slice)
    if m_text:
        textual_resp = m_text.group(1).strip()
    else:
        textual_resp = textual_slice

    return textual_resp, img_ids


def generate_chat_response(user_msg: str, user_info: dict):
    username = get_session_id(user_info)

    # 1) Extract and update preferences, get combined preferences
    raw_extract, new_prefs, raw_merge, preferences = pref_mgr.process(
        username, chat_history, user_msg)

    # 2) Insert preferences into system prompt
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

    # 3) Add user message to chat history
    chat_history.append(HumanMessage(content=user_msg))

    # 4) Get candidate IDs  from PictureAgent
    candidate_ids = picture_agent.process(chat_history, user_msg)

    # 5) Create full chat prompt, merging system messages
    full_prompt = merge_system_messages(chat_history)

    # 6) If candidate IDs is not empty, add them to LLM's context
    if len(candidate_ids) > 0:
        db = picture_agent.db  # ExerciseDB instance
        exercise_rows = db.get_rows_by_ids(candidate_ids)
        reordered_rows = db.reorder_and_filter_columns(exercise_rows)

        # Build the system info prefix for the assistant
        sys_info_block = (
            "### Start of system information ###\n"
            "Here is the set of exercises to choose from:\n"
            f"{reordered_rows.to_json(orient='records', force_ascii=False, indent=2)}\n"
            "### End of system information ###\n\n"
        )

        # Make an AIMessage prefix the model must continue
        ai_prefix = AIMessage(content=sys_info_block)
        # Change full prompt
        full_prompt = merge_system_messages([*chat_history, ai_prefix])

    # 7) Get LLM's response
    response = llm_instance.generate_response(full_prompt)

    print("\n", response, "\n")

    # 8) Parse response into Text and Images sections
    text_resp, img_ids = split_llm_sections(response)

    # print("TEXT PARSED\n", text_resp, "\n")
    print("IMAGES PARSED\n", img_ids, "\n")

    if len(candidate_ids) > 0:
        response = sys_info_block + response
    chat_history.append(AIMessage(content=response))

    # Log this turn
    if logger:
        logger.log_turn({
            "SYSTEM_PROMPT_VERSION": SYSTEM_PROMPT_VERSION,
            "EXTRACT_PREF_PROMPT_VERSION": EXTRACT_PREF_PROMPT_VERSION,
            "MERGE_PREF_PROMPT_VERSION": MERGE_PREF_PROMPT_VERSION,
            "user_message": user_msg,
            "raw_extraction": raw_extract,
            "new_preferences": new_prefs,
            "raw_merge": raw_merge,
            "merged_preferences": preferences,
            "ai_response": response
        })

    # Model decided not to show images
    if not img_ids:
        return text_resp

    # Prepare images for illustration
    images_to_show = []  # List of gr.Image and str with corresponding exercise name
    for id in img_ids:
        try:
            url = db.get_url_by_id(id)
            name = db.get_name_by_id(id)

            images_to_show.append(gr.Image(value=url))
            images_to_show.append(name)
        except Exception:
            continue

    return [text_resp, *images_to_show]
