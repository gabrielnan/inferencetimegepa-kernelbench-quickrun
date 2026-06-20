from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    entry_point: str
    tests: tuple[str, ...]


def load_jsonl(path: Path) -> list[CodeTask]:
    tasks: list[CodeTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                tasks.append(
                    CodeTask(
                        task_id=str(raw["task_id"]),
                        prompt=str(raw["prompt"]),
                        entry_point=str(raw["entry_point"]),
                        tests=tuple(str(t) for t in raw["tests"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required key {exc}") from exc
    if not tasks:
        raise ValueError(f"{path} did not contain any tasks")
    return tasks
