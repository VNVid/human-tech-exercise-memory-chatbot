import os
import re
import json
from typing import Optional, List

from langchain_core.messages import BaseMessage
from core.prompt_manager import PromptManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm.base import BaseLLM
from config import EXTRACT_PREF_PROMPT_VERSION, MERGE_PREF_PROMPT_VERSION


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
        Load a user's saved preferences from storage.

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

    def merge_preferences_raw(
        self, old_prefs: List[str], new_prefs: List[str]
    ) -> str:
        """
        Call LLM to merge old and new preferences in one shot, explaining actions.

        Args:
            old_prefs: List of current preferences loaded from storage.
            new_prefs: List of newly extracted preferences.

        Returns:
            LLM response as str.
        """

        # Prepare merge prompt and context
        merge_prompt = self.prompt_mgr.load(
            "merge_prefs", version=MERGE_PREF_PROMPT_VERSION)

        merge_context = (
            f"old_preferences: {json.dumps(old_prefs, ensure_ascii=False)}\n"
            f"new_preferences: {json.dumps(new_prefs, ensure_ascii=False)}"
        )

        # Build HumanMessage that contains instructions + context
        merge_msg = [
            HumanMessage(
                content=f"{merge_prompt}\n\n{merge_context}")
        ]

        # Call the LLM merger
        raw = self.llm.generate_response(merge_msg)

        return raw

    def parse_output(self, raw: str, key: str = "new_preferences") -> List[str]:
        """
        Given the extractor or merge LLM's raw string, parse the JSON under "Output: { key : [...]}".

        Returns:
            the list value of key or [].
        """

        m = re.search(r"Output:\s*(\{.*\})", raw, flags=re.S)

        if not m:
            return []
        try:
            obj = json.loads(m.group(1))  # returns only the JSON part
            return obj.get(key, []) or []
        except json.JSONDecodeError:
            # If failed to parse, assume nothing new
            return []

    def add_preferences(self, username: str, new_prefs: List[str]):
        """
        Merge new_prefs into the user's existing list, save and return the merged list.

        This method ensures preference memory stays consistent and efficient by:
        - removing duplicates
        - replacing old preferences that contradict new ones
        - saving changes only when necessary.

        It maintains an up-to-date and non-conflicting set of user preferences stored in the memory.

        Args:
            username: The user's identifier.
            new_prefs: List of newly extracted preferences.

        Returns:
            Tuple of raw LLM output and the updated list of user preferences.
        """

        # Load existing preferences form storage
        existing = self.load(username)
        # If there are no new preferences to merge, return existing ones
        if not new_prefs:
            return "", existing

        # If there are no preferences in the storage yet, save and return new ones
        if not existing:
            self.save(username, new_prefs)
            return "", new_prefs

        # Merge via LLM
        raw_merge = self.merge_preferences_raw(existing, new_prefs)
        # Parse merged list
        merged = self.parse_output(raw_merge, key="merged_preferences")
        # Save resulting list in the storage
        self.save(username, merged)

        return raw_merge, merged

    def process(self, username: str, chat_history: List[BaseMessage], last_user_message: str):
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
            Tuple of raw LLM extraction output, list of new preferences, raw LLM merge output and 
            the updated list of user preferences.
        """

        # Extract via LLM
        raw = self.extract_preferences_raw(chat_history, last_user_message)
        # Parse extracted list
        new_prefs = self.parse_output(raw)
        # Merge new preferences with older ones
        raw_merged, merged = self.add_preferences(username, new_prefs)

        print("EXTRACTOR RESPONSE: \n", raw)
        print("\n NEW PREFERENCES: \n", new_prefs)
        print("\n MERGE RESPONSE: \n", raw_merged)
        print("\n MERGED PREFERENCES: \n", merged)

        return raw, new_prefs, raw_merged, merged
