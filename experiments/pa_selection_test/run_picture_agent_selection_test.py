from __future__ import annotations

import argparse
import contextlib
import io
from typing import List
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, BaseMessage

project_root = Path(__file__).resolve()
while not (project_root / "core").is_dir():
    if project_root.parent == project_root:
        raise RuntimeError("Couldn’t locate project root (no core/ folder found).")
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from core.picture_agent import PictureAgent
from llm.openai_backend import OpenAIChat

def capture_stdout(fn):
    """Decorator: run *fn* and capture anything written to *stdout* (str)."""
    def wrapper(*args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    return wrapper


@capture_stdout
def _process(agent: PictureAgent, chat_history: List[BaseMessage], msg: str) -> None:
    agent.process(chat_history, msg)


def run_one(agent: PictureAgent, user_msg: str) -> str:
    """Run *agent* on *user_msg* and return everything printed to stdout."""
    chat_history: List[BaseMessage] = [HumanMessage(content=user_msg)]
    return _process(agent, chat_history, user_msg)


def main():
    parser = argparse.ArgumentParser(description="Capture PictureAgent raw selector output (one file per run)")
    parser.add_argument("messages_file", help="Text file with ONE user message per line")
    parser.add_argument("--prefix", default="run", help="Filename prefix – useful when testing different prompt versions")
    parser.add_argument("--runs", "-n", type=int, default=3, help="How many times to process EACH message (default: 3)")
    args = parser.parse_args()

    # Load messages
    messages: List[str] = [line.strip() for line in Path(args.messages_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not messages:
        raise SystemExit("[error] The messages file is empty.")

    # Save results into an 'outputs/' next to this script
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    agent = PictureAgent(OpenAIChat())

    for midx, msg in enumerate(messages, 1):
        for run in range(1, args.runs + 1):
            raw_log = run_one(agent, msg)

            fname = f"{args.prefix}_{midx}_r{run}.txt"
            (out_dir / fname).write_text(raw_log, encoding="utf-8")

            print(f"[msg {midx}/{len(messages)} · run {run}/{args.runs}] → {fname} ({len(raw_log)} chars)")


if __name__ == "__main__":
    main()
