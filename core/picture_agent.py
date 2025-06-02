import re
import json
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from llm.base import BaseLLM
from core.exercise_db import ExerciseDB
from core.prompt_manager import PromptManager
from config import PICTURE_AGENT_SEARCH_PROMPT_VERSION, PICTURE_AGENT_SELECT_PROMPT_VERSION, RANDOM_SEED


class PictureAgent:
    """
    A 2-step ReAct-style agent (no loops):

     1) Decide if the user is asking for exercises, and if so
        return Dict with search parameters:
        Action: {"chat": false,
                 "search parameters": {...}}
        Otherwise short-circuit to {"chat": true}.

     2) Given the Observation (list of rows resulted from search),
        pick the best N exercises and return their IDs.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.db = ExerciseDB()
        self.prompt_mgr = PromptManager()

        self.last_ids: List[int] = []
        self.session_ids: List[List[int]] = []

        # pre‐load column‐value lists for the search prompt
        self._search_listings = self.db.all_values()

    def clear_state(self):
        self.last_ids = []
        self.session_ids = []

    def _parse_action(self, raw: str, only_json: bool = False) -> Optional[Dict]:
        """
        Parse out a JSON object from the raw text.

        There are two supported variants:
        1. When only_json is False (default), look for an 'Action:' section at the end of raw
        and parse the JSON that follows it.
        2. When only_json is True, assume that raw itself is (or contains) a JSON object and
        attempt to parse it directly.

        Returns:
            A dict if a valid JSON object is found and parsed successfully; otherwise, None.
        """

        raw = raw.strip()

        # Variant 2: raw is just a JSON
        if only_json:
            # Try to parse the JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None

        # Variant 1: find the 'Action:' keyword and parse the JSON that follows.
        m = re.search(r"Action:\s*\{", raw)
        if not m:
            return None

        # Find the index of the first '{' after 'Action:'
        start = m.start() + raw[m.start():].find("{")
        # Everything from '{' to end
        json_block = raw[start:].strip()

        # Try to parse the captured JSON
        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            return None

    def decide_on_search(self, chat_history: List[BaseMessage], last_user_message: str) -> Dict:
        # Grab a short window of the last few messages for context, filtering out SystemMessage entries.
        chat_context = []
        for msg in chat_history[-4:]:
            if isinstance(msg, SystemMessage):
                role = "system"
                continue
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            chat_context.append({"role": role, "content": msg.content})

        # Serialize into the single context string
        chat_json = json.dumps(chat_context, ensure_ascii=False, indent=2)
        # Create a context for agent's decision on recommending exercise(s)
        decision_context = (
            "chat_history:\n"
            f"{chat_json}\n\n"
            f"last_user_message: {last_user_message}"
        )

        # Prepare prompt for decision on searching exercise(s).
        # Insert column names and values into prompt.
        columns_list = ", ".join(self._search_listings.keys())
        columns_json = json.dumps(
            self._search_listings, ensure_ascii=False, indent=2)

        search_prompt = self.prompt_mgr.load(
            "picture_agent_search", PICTURE_AGENT_SEARCH_PROMPT_VERSION).format(
                columns_list=columns_list,
                columns=columns_json,
        )

        # Build HumanMessage that contains instructions + context
        instruction_msg = [
            HumanMessage(
                content=f"{search_prompt}\n\n{decision_context}")
        ]

        # Call the LLM
        raw = self.llm.generate_response(instruction_msg)

        print("\n\n Raw\n", raw, "\n")

        action = self._parse_action(raw)
        if action is None:
            action = {"chat": True}

        print("\n\n Action\n", action, "\n")

        return action

    def select_exercises(self, chat_history: List[BaseMessage], last_user_message: str,
                         matching_rows, default_set_size: int = 10, max_rows_num: int = 100) -> Dict:
        """
        From matching_rows (a DataFrame of candidate exercises),
        pick the most suitable subset and return the list of IDs.
        """

        # Grab a short window of the last few messages for context, filtering out SystemMessage entries.
        chat_context = []
        for msg in chat_history[-4:]:
            if isinstance(msg, SystemMessage):
                role = "system"
                continue
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            chat_context.append({"role": role, "content": msg.content})

        # Serialize into the single context string
        chat_json = json.dumps(chat_context, ensure_ascii=False, indent=2)

        # Define column list for selection prompt formatting and matching_rows reordering.
        # 1) Exclude any "url" columns (case-insensitive)
        filtered_cols = [c for c in list(
            matching_rows.keys()) if "url" not in c.lower()]
        # 2) Put "id" first, "name" (the exercise column) second, then the rest
        ordered_cols = ["id", "name"]
        for c in filtered_cols:
            if c not in ordered_cols:
                ordered_cols.append(c)

        # Reorder columns in canditate rows, sample max_rows_num random rows and serialize them
        reordered_df = matching_rows[ordered_cols]

        if len(reordered_df) > max_rows_num:
            # Randomly sample max_rows_num rows without replacement
            reordered_df = reordered_df.sample(
                n=max_rows_num, random_state=RANDOM_SEED)

        rows_json = reordered_df.to_json(
            orient="records", force_ascii=False, indent=2)

        # Create a context for agent's decision on selecting exercise(s)
        selection_context = (
            "chat_history:\n"
            f"{chat_json}\n\n"
            f"last_user_message: {last_user_message}\n\n"
            f"{rows_json}"
        )

        # Load and format the selection prompt.
        # Insert column names and default number of exercises to select.
        select_prompt = self.prompt_mgr.load(
            "picture_agent_select", PICTURE_AGENT_SELECT_PROMPT_VERSION
        ).format(
            columns_list=ordered_cols,
            default_set_size=default_set_size
        )

        # Build HumanMessage that contains instructions + context
        instruction_msg = [
            HumanMessage(
                content=f"{select_prompt}\n\n{selection_context}")
        ]

        # Call the LLM
        raw = self.llm.generate_response(instruction_msg)

        print("Selection raw:\n", raw)

        selected_ids = self._parse_action(raw)
        if selected_ids is None:
            selected_ids = {"IDs": []}

        print("Selected IDs:\n", selected_ids)

        return selected_ids

    def process(self, chat_history: List[BaseMessage], last_user_message: str) -> List[int]:
        """
        Run the 2-step agent. Returns {"chat": bool, "IDs": [...]}
        and sets self.last_ids.
        """

        action = self.decide_on_search(chat_history, last_user_message)

        # User not asking for exercises
        if action.get("chat", True):
            return []

        # Run the DB search tool
        params = action.get("search parameters", {})
        matching_rows = self.db.get_rows_by_dict(params)

        matching_rows.to_csv("filename.csv", index=False)  # ____    DELETE

        selected_exercises = self.select_exercises(
            chat_history, last_user_message, matching_rows)

        return selected_exercises["IDs"]
