from __future__ import annotations

import json
import os
from pathlib import Path

from inferencetimegepa.code_extract import extract_python_code


def score_contributions(rows: list[dict], method: str = "heuristic", judge_model: str | None = None) -> list[dict]:
    if method == "heuristic":
        return _score_heuristic(rows)
    if method == "openai":
        if not judge_model:
            raise ValueError("judge_model is required for openai contribution scoring")
        return _score_openai(rows, judge_model)
    if method == "anthropic":
        if not judge_model:
            raise ValueError("judge_model is required for anthropic contribution scoring")
        return _score_anthropic(rows, judge_model)
    raise ValueError(f"unknown contribution method {method!r}")


def load_dict_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_dict_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _score_heuristic(rows: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for row in rows:
        seen_valid: set[str] = set()
        new_row = {**row, "rollouts": []}
        for rollout in row["rollouts"]:
            code_key = _code_key(rollout["completion"]["text"])
            valid = bool(rollout["eval"]["valid"])
            if not valid:
                contribution = 0.0
                reason = "invalid rollout"
            elif code_key not in seen_valid:
                contribution = 1.0
                reason = "first valid solution with this code shape"
                seen_valid.add(code_key)
            else:
                contribution = 0.2
                reason = "valid but redundant solution"

            completion = {
                **rollout["completion"],
                "meta": {
                    **rollout["completion"].get("meta", {}),
                    "contribution_score": contribution,
                    "contribution_reason": reason,
                    "contribution_method": "heuristic",
                },
            }
            new_row["rollouts"].append({**rollout, "completion": completion})
        scored.append(new_row)
    return scored


def _score_openai(rows: list[dict], judge_model: str) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install with `python -m pip install -e '.[openai]'`") from exc

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )

    scored: list[dict] = []
    for row in rows:
        new_row = {**row, "rollouts": []}
        valid_codes = [
            {
                "index": idx,
                "agent_id": rollout.get("agent_id"),
                "score": rollout["eval"]["score"],
                "valid": rollout["eval"]["valid"],
                "code": extract_python_code(rollout["completion"]["text"]),
            }
            for idx, rollout in enumerate(row["rollouts"])
        ]
        prompt = (
            "Score each rollout's useful contribution to solution search from 0.0 to 1.0. "
            "Reward correct, non-redundant approaches. Penalize invalid or duplicate attempts. "
            "Return only JSON: {\"scores\": [{\"index\": int, \"score\": float, \"reason\": str}]}.\n\n"
            f"Task id: {row['task_id']}\nRollouts:\n{json.dumps(valid_codes)}"
        )
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
            by_index = {int(item["index"]): item for item in payload.get("scores", [])}
        except Exception:
            by_index = {}

        for idx, rollout in enumerate(row["rollouts"]):
            item = by_index.get(idx, {})
            contribution = float(item.get("score", 0.0))
            contribution = max(0.0, min(1.0, contribution))
            completion = {
                **rollout["completion"],
                "meta": {
                    **rollout["completion"].get("meta", {}),
                    "contribution_score": contribution,
                    "contribution_reason": str(item.get("reason", "judge did not provide a reason")),
                    "contribution_method": "openai",
                    "contribution_judge_model": judge_model,
                },
            }
            new_row["rollouts"].append({**rollout, "completion": completion})
        scored.append(new_row)
    return scored


def _score_anthropic(rows: list[dict], judge_model: str) -> list[dict]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Install with `python -m pip install -e '.[anthropic]'`") from exc

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    scored: list[dict] = []
    for row in rows:
        new_row = {**row, "rollouts": []}
        valid_codes = [
            {
                "index": idx,
                "agent_id": rollout.get("agent_id"),
                "score": rollout["eval"]["score"],
                "valid": rollout["eval"]["valid"],
                "code": extract_python_code(rollout["completion"]["text"]),
            }
            for idx, rollout in enumerate(row["rollouts"])
        ]
        prompt = (
            "Score each rollout's useful contribution to solution search from 0.0 to 1.0. "
            "Reward correct, non-redundant approaches. Penalize invalid or duplicate attempts. "
            "Return only JSON: {\"scores\": [{\"index\": int, \"score\": float, \"reason\": str}]}.\n\n"
            f"Task id: {row['task_id']}\nRollouts:\n{json.dumps(valid_codes)}"
        )
        message = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        try:
            payload = json.loads(content)
            by_index = {int(item["index"]): item for item in payload.get("scores", [])}
        except Exception:
            by_index = {}

        for idx, rollout in enumerate(row["rollouts"]):
            item = by_index.get(idx, {})
            contribution = float(item.get("score", 0.0))
            contribution = max(0.0, min(1.0, contribution))
            completion = {
                **rollout["completion"],
                "meta": {
                    **rollout["completion"].get("meta", {}),
                    "contribution_score": contribution,
                    "contribution_reason": str(item.get("reason", "judge did not provide a reason")),
                    "contribution_method": "anthropic",
                    "contribution_judge_model": judge_model,
                },
            }
            new_row["rollouts"].append({**rollout, "completion": completion})
        scored.append(new_row)
    return scored


def _code_key(text: str) -> str:
    code = extract_python_code(text)
    return " ".join(line.strip() for line in code.splitlines() if line.strip())
