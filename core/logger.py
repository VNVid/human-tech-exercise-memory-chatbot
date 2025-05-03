import os
import json
from datetime import datetime


class Logger:
    """
    Logger for chat sessions.

    Stores one JSONL log file per session in a user-specific directory.
    Each turn is appended as a JSON block separated by blank lines.
    """

    def __init__(self, username: str, base_dir: str = "storage/logs"):
        """
        Initialize the logger for a given user.

        Args:
            username (str): Identifier for the user (used to name the directory).
            base_dir (str): Root directory where logs are stored.
        """

        self.username = username
        user_dir = os.path.join(base_dir, username)
        # Create the directory if it does not exist
        os.makedirs(user_dir, exist_ok=True)

        # One session file, timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(user_dir, f"{ts}.jsonl")

    def log_turn(self, entry: dict):
        """
        Append a single-turn log entry as JSON.

        Args:
            entry (dict): A dictionary containing data for this turn, e.g.:
                {
                    'username': 'Nadia',
                    'user_message': 'Hello',
                    'raw_extraction': 'Reasoning: ...',
                    'new_preferences': [...],
                    'combined_preferences': [...],
                    'ai_response': 'Hi there!'
                }
        """
        # Add timestamp if not provided
        entry.setdefault("timestamp", datetime.now().isoformat())
        dumped = json.dumps(entry, ensure_ascii=False, indent=2)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(dumped)
            f.write("\n\n")   # blank line separator
