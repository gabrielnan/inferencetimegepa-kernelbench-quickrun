from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from inferencetimegepa.evaluator import EvalResult, evaluate_completion
from inferencetimegepa.samplers import Completion, Sampler
from inferencetimegepa.tasks import CodeTask


@dataclass(frozen=True)
class RolloutResult:
    completion: Completion
    eval: EvalResult
    agent_id: str = "agent_0"


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    k: int
    best_score: float
    mean_score: float
    valid_rate: float
    rollouts: tuple[RolloutResult, ...]


def run_single_agent(
    tasks: list[CodeTask],
    sampler: Sampler,
    k: int,
    seed: int,
    timeout_s: float,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for task_idx, task in enumerate(tasks):
        completions = sampler.sample(task, k=k, seed=seed + task_idx * 100_000)
        rollouts = tuple(
            RolloutResult(
                completion=completion,
                eval=evaluate_completion(task, completion.text, timeout_s=timeout_s),
            )
            for completion in completions
        )
        scores = [r.eval.score for r in rollouts]
        valid = [r.eval.valid for r in rollouts]
        results.append(
            TaskResult(
                task_id=task.task_id,
                k=k,
                best_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                valid_rate=sum(valid) / len(valid) if valid else 0.0,
                rollouts=rollouts,
            )
        )
    return results


def run_agent(
    tasks: list[CodeTask],
    sampler: Sampler,
    agent_id: str,
    k: int,
    seed: int,
    timeout_s: float,
    system_prompt: str | None = None,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for task_idx, task in enumerate(tasks):
        completions = sampler.sample(
            task,
            k=k,
            seed=seed + task_idx * 100_000,
            system_prompt=system_prompt,
        )
        rollouts = tuple(
            RolloutResult(
                completion=completion,
                eval=evaluate_completion(task, completion.text, timeout_s=timeout_s),
                agent_id=agent_id,
            )
            for completion in completions
        )
        scores = [r.eval.score for r in rollouts]
        valid = [r.eval.valid for r in rollouts]
        results.append(
            TaskResult(
                task_id=task.task_id,
                k=k,
                best_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                valid_rate=sum(valid) / len(valid) if valid else 0.0,
                rollouts=rollouts,
            )
        )
    return results


def run_routed_agents(
    tasks: list[CodeTask],
    samplers_by_agent: dict[str, Sampler],
    prompts_by_agent: dict[str, str],
    router,
    rollouts_per_agent: int,
    seed: int,
    timeout_s: float,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for task_idx, task in enumerate(tasks):
        decision = router(task)
        if decision.agent_id not in samplers_by_agent:
            raise ValueError(f"router selected unknown agent {decision.agent_id!r}")
        completions = samplers_by_agent[decision.agent_id].sample(
            task,
            k=rollouts_per_agent,
            seed=seed + task_idx * 100_000,
            system_prompt=prompts_by_agent.get(decision.agent_id),
        )
        rollouts = tuple(
            RolloutResult(
                completion=Completion(
                    text=completion.text,
                    seed=completion.seed,
                    meta={
                        **completion.meta,
                        "router_reason": decision.reason,
                    },
                ),
                eval=evaluate_completion(task, completion.text, timeout_s=timeout_s),
                agent_id=decision.agent_id,
            )
            for completion in completions
        )

        scores = [r.eval.score for r in rollouts]
        valid = [r.eval.valid for r in rollouts]
        results.append(
            TaskResult(
                task_id=task.task_id,
                k=len(rollouts),
                best_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                valid_rate=sum(valid) / len(valid) if valid else 0.0,
                rollouts=tuple(rollouts),
            )
        )
    return results


def merge_task_results(paths: list[Path]) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_task.setdefault(row["task_id"], []).extend(row["rollouts"])

    merged: list[dict] = []
    for task_id, rollouts in sorted(by_task.items()):
        scores = [float(r["eval"]["score"]) for r in rollouts]
        valid = [bool(r["eval"]["valid"]) for r in rollouts]
        merged.append(
            {
                "task_id": task_id,
                "k": len(rollouts),
                "best_score": max(scores) if scores else 0.0,
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "valid_rate": sum(valid) / len(valid) if valid else 0.0,
                "rollouts": rollouts,
            }
        )
    return merged


def write_dict_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_task_result_to_dict(result), sort_keys=True) + "\n")


def summarize(results: list[TaskResult]) -> dict[str, float | int]:
    if not results:
        return {
            "tasks": 0,
            "mean_best_score": 0.0,
            "mean_score": 0.0,
            "mean_valid_rate": 0.0,
        }
    return {
        "tasks": len(results),
        "mean_best_score": sum(r.best_score for r in results) / len(results),
        "mean_score": sum(r.mean_score for r in results) / len(results),
        "mean_valid_rate": sum(r.valid_rate for r in results) / len(results),
    }


def summarize_dict_rows(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {
            "tasks": 0,
            "mean_best_score": 0.0,
            "mean_score": 0.0,
            "mean_valid_rate": 0.0,
        }
    return {
        "tasks": len(rows),
        "mean_best_score": sum(float(r["best_score"]) for r in rows) / len(rows),
        "mean_score": sum(float(r["mean_score"]) for r in rows) / len(rows),
        "mean_valid_rate": sum(float(r["valid_rate"]) for r in rows) / len(rows),
    }


def _task_result_to_dict(result: TaskResult) -> dict:
    return {
        "task_id": result.task_id,
        "k": result.k,
        "best_score": result.best_score,
        "mean_score": result.mean_score,
        "valid_rate": result.valid_rate,
        "rollouts": [
            {
                "completion": asdict(rollout.completion),
                "eval": asdict(rollout.eval),
                "agent_id": rollout.agent_id,
            }
            for rollout in result.rollouts
        ],
    }
