import os


class PromptManager:
    """
    A utility class for managing and loading prompt files organized in a directory structure.

    Attributes:
        base_dir (str): The base directory where prompt files (and possible subdirectories) are stored.
    """

    def __init__(self, base_dir: str = "prompts"):
        self.base_dir = base_dir

    def load(self, category: str, version: str = "v1") -> str:
        """
        Load prompts/<category>/<version>.txt
        """
        path = os.path.join(self.base_dir, category, f"{version}.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
