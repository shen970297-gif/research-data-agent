import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "runs.jsonl"


def write_research_event(root: Path, event: dict):
    """索引每次研究运行；详细证据留在对应的 run 目录中。"""
    path = root / "logs" / "research_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": datetime.now().astimezone().isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def write_run_log(
    user_question: str,
    tool_name: str | None,
    tool_arguments: dict | None,
    tool_result: dict | None,
    final_answer: str | None,
    error: str | None = None,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user_question": user_question,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "tool_result": tool_result,
        "final_answer": final_answer,
        "error": error,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
