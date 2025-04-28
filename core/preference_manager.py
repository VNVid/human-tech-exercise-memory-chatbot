import os
import re
import json
from typing import Optional, List

from langchain_core.messages import BaseMessage
from core.prompt_manager import PromptManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm.base import BaseLLM
from config import EXTRACT_PREF_PROMPT_VERSION


class PreferenceManager:
    """
    Implements the long-term memory for the chatbot.

    PreferenceManager handles extracting and updating, saving, loading and injecting user 
    preferences across conversations. It provides persistent storage per user 
    to enable the chatbot to remember preferences between sessions.

    This class represents the memory module of the project.
    """

    def __init__(self, llm: BaseLLM, storage_dir: str = "storage/preferences"):
        """
        Initialize the PreferenceManager.

        Args:
            llm: An LLM that implements generate_response(messages: List[BaseMessage]) -> str.
            storage_dir: Directory where per-user preference JSON files are stored.
        """

        self.llm = llm
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.prompt_mgr = PromptManager()

    def _path(self, username: str) -> str:
        """
        Build the file path for a user's preference file.
        """

        return os.path.join(self.storage_dir, f"{username}.json")

    def load(self, username: str) -> List[str]:
        """
        Load a user's saved preferences from disk.

        Args:
            username: The user's identifier.

        Returns:
            The saved preferences as a list of strings (or return empty list if none yet).
        """

        p = self._path(username)

        if not os.path.exists(p):
            return []

        data = json.load(open(p, "r", encoding="utf-8"))
        return data.get("current_preferences", [])

    def save(self, username: str, prefs: List[str]):
        """
        Overwrite the user's preference file with the given list.

        Args:
            username: The user's identifier.
            prefs: The list of preferences to save.
        """
        p = self._path(username)

        data = {"current_preferences": prefs}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def extract_preferences_raw(self, chat_history: List[BaseMessage], last_user_message: str) -> str:
        """
        Run the preference extraction prompt on the conversation history 
        and returns raw response of the LLM.

        Args:
            username: The user's identifier.
            chat_history: List of past conversation messages.
            last_user_message: Last user message to extract preferences from.

        Returns:
            LLM response as str.
        """

        # Prepare extraction prompt and context
        extraction_prompt = self.prompt_mgr.load(
            "extract_prefs", version=EXTRACT_PREF_PROMPT_VERSION)

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

        # Serialize into the single context string
        chat_json = json.dumps(serialized, ensure_ascii=False, indent=2)
        extraction_context = (
            "chat_history:\n"
            f"{chat_json}\n\n"
            f"last_user_message: {last_user_message}"
        )

        # Build HumanMessage that contains instructions + context
        extraction_msg = [
            HumanMessage(
                content=f"{extraction_prompt}\n\n{extraction_context}")
        ]

        # Call the LLM extractor
        raw = self.llm.generate_response(extraction_msg)
        return raw

    def parse_extraction_output(self, raw: str) -> List[str]:
        """
        Given the extractor LLM's raw string, parse the JSON under "Output: {"new_preferences": [...]}".

        Returns:
            the list value of "new_preferences" or [].
        """
        m = re.search(r"Output:\s*(\{.*\})", raw, flags=re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(1))  # returns only the JSON part
            return obj.get("new_preferences", []) or []
        except json.JSONDecodeError:
            # If failed to parse, assume nothing new
            return []

    def add_preferences(self, username: str, new_prefs: List[str]) -> List[str]:
        """
        Merge new_prefs into the user's existing list, save and return the combined list.
        """
        if not new_prefs:
            return self.load(username)

        #         TO-DO:      deduplication            TO-DO
        existing = self.load(username)
        combined = existing.copy()

        for p in new_prefs:
            if p not in combined:
                combined.append(p)

        if combined != existing:
            self.save(username, combined)

        return combined

    def process(self, username: str, chat_history: List[BaseMessage], last_user_message: str) -> List[str]:
        """
        Main entry point for preference extraction and memory update during a chat session.
        It should be called by the chat engine at each user turn to keep memory updated.

        This method runs the full pipeline: extract new preferences from the latest conversation,
        parse and merge them into the user's long-term memory.

        Args:
            username: The user's identifier.
            chat_history: Full conversation history up to this point.
            last_user_message: The last message sent by the user.

        Returns:
            The updated list of user preferences.
        """

        raw = self.extract_preferences_raw(chat_history, last_user_message)
        new_prefs = self.parse_extraction_output(raw)
        combined = self.add_preferences(username, new_prefs)

        print("EXTRACTOR RESPONSE: \n", raw)
        print("\nNEW PREFERENCES: \n", new_prefs)
        print("\COMBINED PREFERENCES: \n", combined)

        return combined
