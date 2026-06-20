from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptCandidate:
    candidate_id: str
    role: str
    prompt: str
    metrics: dict[str, float]
    parent_ids: tuple[str, ...] = ()
    reflection: str = ""


def load_prompt_candidates(path: Path) -> list[PromptCandidate]:
    candidates: list[PromptCandidate] = []
    if not path.exists():
        return candidates
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates.append(
                PromptCandidate(
                    candidate_id=str(row["candidate_id"]),
                    role=str(row["role"]),
                    prompt=str(row["prompt"]),
                    metrics={str(k): float(v) for k, v in row.get("metrics", {}).items()},
                    parent_ids=tuple(str(v) for v in row.get("parent_ids", ())),
                    reflection=str(row.get("reflection", "")),
                )
            )
    return candidates


def write_prompt_candidates(path: Path, candidates: list[PromptCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(asdict(candidate), sort_keys=True) + "\n")


def pareto_frontier(candidates: list[PromptCandidate], metric_names: list[str]) -> list[PromptCandidate]:
    """Return candidates not dominated on all requested metrics. Higher is better."""
    frontier: list[PromptCandidate] = []
    for candidate in candidates:
        if not any(_dominates(other, candidate, metric_names) for other in candidates if other is not candidate):
            frontier.append(candidate)
    return sorted(frontier, key=lambda c: c.candidate_id)


def build_reflection_prompt(
    role: str,
    rollout_path: Path,
    frontier: list[PromptCandidate],
    max_failures: int = 6,
) -> str:
    rows = _load_jsonl_dicts(rollout_path)
    failures = _failure_summaries(rows, max_failures=max_failures)
    frontier_summary = "\n".join(
        f"- {candidate.candidate_id}: metrics={candidate.metrics}; reflection={candidate.reflection or 'n/a'}"
        for candidate in frontier
    )
    failure_summary = "\n".join(f"- {failure}" for failure in failures) or "- No failures found in this batch."
    return (
        "You are optimizing the instructions for a Devin team that implements KernelBench kernels.\n"
        "Use GEPA-style reflection: read full traces and actionable side information, diagnose failures, "
        "and propose targeted prompt mutations.\n"
        "Do not propose RL, fine-tuning, LoRA, or weight updates. Improve the Devin team prompt text only.\n\n"
        f"Role being optimized: {role}\n\n"
        "Current Pareto frontier:\n"
        f"{frontier_summary or '- No frontier candidates yet.'}\n\n"
        "Observed failures and weak traces:\n"
        f"{failure_summary}\n\n"
        "Return JSON with keys: candidate_id, role, prompt, parent_ids, reflection. The prompt must be directly usable as a system prompt."
    )


def candidate_from_json(text: str) -> PromptCandidate:
    row = json.loads(text)
    return PromptCandidate(
        candidate_id=str(row["candidate_id"]),
        role=str(row["role"]),
        prompt=str(row["prompt"]),
        metrics={str(k): float(v) for k, v in row.get("metrics", {}).items()},
        parent_ids=tuple(str(v) for v in row.get("parent_ids", ())),
        reflection=str(row.get("reflection", "")),
    )


def _dominates(left: PromptCandidate, right: PromptCandidate, metric_names: list[str]) -> bool:
    left_values = [left.metrics.get(metric, 0.0) for metric in metric_names]
    right_values = [right.metrics.get(metric, 0.0) for metric in metric_names]
    return all(l >= r for l, r in zip(left_values, right_values)) and any(
        l > r for l, r in zip(left_values, right_values)
    )


def _load_jsonl_dicts(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _failure_summaries(rows: list[dict], max_failures: int) -> list[str]:
    failures: list[str] = []
    for row in rows:
        task_id = row.get("task_id", "unknown")
        for rollout in row.get("rollouts", []):
            evaluation = rollout.get("eval", {})
            if evaluation.get("valid") and float(evaluation.get("score", 0.0)) >= 1.0:
                continue
            completion = rollout.get("completion", {})
            text = str(completion.get("text", "")).replace("\n", "\\n")
            error = evaluation.get("error") or evaluation.get("stderr") or "low score"
            failures.append(f"{task_id}: {error}; completion_prefix={text[:240]}")
            if len(failures) >= max_failures:
                return failures
    return failures
