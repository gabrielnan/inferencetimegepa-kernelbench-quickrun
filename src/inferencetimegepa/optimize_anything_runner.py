from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROLE_FILES = {
    "kernel_author": Path("prompts/roles/kernel_author.md"),
    "correctness_reviewer": Path("prompts/roles/correctness_reviewer.md"),
    "benchmark_engineer": Path("prompts/roles/benchmark_engineer.md"),
    "gepa_reflector": Path("prompts/roles/gepa_reflector.md"),
}


def load_seed_prompt_pack(repo_root: Path) -> dict[str, str]:
    return {role: (repo_root / path).read_text(encoding="utf-8") for role, path in ROLE_FILES.items()}


def write_prompt_pack(repo_root: Path, prompt_pack: dict[str, str], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for role, text in prompt_pack.items():
        path = out_dir / f"{role}.md"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        written[role] = path
    return written


def run_optimize_anything(
    repo_root: Path,
    dataset_path: Path,
    out_dir: Path,
    max_metric_calls: int,
    max_candidate_proposals: int,
    reflection_lm: str | None,
    devin_command: str | None,
    compare_cmd: str | None,
    kernelbench_timeout_s: float,
) -> Any:
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, log, optimize_anything

    _load_env(repo_root / ".env")
    dataset = _load_jsonl(dataset_path)
    if not dataset:
        raise ValueError(f"dataset is empty: {dataset_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_candidate = load_seed_prompt_pack(repo_root)

    def evaluator(candidate: dict[str, str], example: dict[str, Any] | None = None) -> float:
        task = example or {}
        if not devin_command:
            raise ValueError("--devin-command is required")
        if not compare_cmd:
            raise ValueError("--compare-cmd is required")
        summary = _evaluate_devin_candidate(
            repo_root=repo_root,
            candidate=candidate,
            task=task,
            out_dir=out_dir,
            devin_command=devin_command,
            compare_cmd=compare_cmd,
            kernelbench_timeout_s=kernelbench_timeout_s,
            gepa_log=log,
        )
        return float(summary.get("score", 0.0))

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        dataset=dataset,
        valset=dataset,
        objective=(
            "Improve Devin team prompts for KernelBench kernel implementation. "
            "Maximize correctness and accepted speed/energy improvements without relying on hidden tests, "
            "scorer internals, comments, filenames, or benchmark artifacts."
        ),
        background=(
            "The candidate is a prompt pack for Devin roles. Devin implements kernels; an external KernelBench "
            "scoring function evaluates correctness, latency, and optional Zeus energy."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(out_dir / "gepa_run"),
                max_metric_calls=max_metric_calls,
                max_candidate_proposals=max_candidate_proposals,
                parallel=False,
                capture_stdio=True,
                display_progress_bar=False,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
            merge=None,
        ),
    )
    _write_result(out_dir, result)
    return result


def run_codex_reflection_packet(
    repo_root: Path,
    dataset_path: Path,
    out_dir: Path,
    devin_command: str,
    compare_cmd: str,
    kernelbench_timeout_s: float,
    team_size: int = 1,
    router_command: str | None = None,
) -> Path:
    _load_env(repo_root / ".env")
    dataset = _load_jsonl(dataset_path)
    if not dataset:
        raise ValueError(f"dataset is empty: {dataset_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_pack = load_seed_prompt_pack(repo_root)
    eval_summaries: list[dict[str, Any]] = []
    for task in dataset:
        router_plan = ""
        router_plan_path = ""
        router_summary: dict[str, Any] | None = None
        if router_command and team_size > 1:
            router_summary = _run_devin_router(
                repo_root=repo_root,
                candidate=prompt_pack,
                task=task,
                out_dir=out_dir,
                router_command=router_command,
            )
            router_plan = str(router_summary.get("router_plan", ""))
            router_plan_path = str(router_summary.get("router_plan_path", ""))
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(max(1, team_size)):
            attempts.append(
                _evaluate_devin_candidate(
                    repo_root=repo_root,
                    candidate=prompt_pack,
                    task=task,
                    out_dir=out_dir,
                    devin_command=devin_command,
                    compare_cmd=compare_cmd,
                    kernelbench_timeout_s=kernelbench_timeout_s,
                    gepa_log=None,
                    attempt_index=attempt_index,
                    team_size=max(1, team_size),
                    team_role=_team_role(attempt_index),
                    router_plan=router_plan,
                    router_plan_path=router_plan_path,
                )
            )
        best = dict(max(attempts, key=_summary_rank))
        best["team_attempts"] = attempts
        if router_summary:
            best["router"] = router_summary
        eval_summaries.append(best)
    team_path = out_dir / "team_attempts.json"
    team_path.write_text(json.dumps(eval_summaries, indent=2, sort_keys=True), encoding="utf-8")
    packet_path = out_dir / "codex_reflection_packet.md"
    packet_path.write_text(_format_codex_packet(prompt_pack, eval_summaries), encoding="utf-8")
    return packet_path


def _summary_rank(summary: dict[str, Any]) -> tuple[float, float, float]:
    row = summary.get("kernelbench_row", {})
    return (
        1.0 if row.get("accepted") else 0.0,
        1.0 if row.get("correct") else 0.0,
        float(row.get("score", summary.get("score", 0.0)) or 0.0),
    )


def _team_role(attempt_index: int) -> str:
    roles = [
        "correctness_first_simple_cuda",
        "memory_coalescing_and_layout",
        "tiling_and_shared_memory",
    ]
    return roles[attempt_index % len(roles)]


def _run_devin_router(
    repo_root: Path,
    candidate: dict[str, str],
    task: dict[str, Any],
    out_dir: Path,
    router_command: str,
) -> dict[str, Any]:
    task_id = str(task.get("task_id", "unknown"))
    router_dir = out_dir / "evals" / _safe_name(task_id) / f"router_{int(time.time() * 1000)}"
    prompt_dir = router_dir / "prompts"
    write_prompt_pack(repo_root, candidate, prompt_dir)
    router_dir.mkdir(parents=True, exist_ok=True)
    plan_path = router_dir / "router_plan.md"
    trace_path = router_dir / "devin_router_trace.log"
    command = router_command.format(
        task_id=task_id,
        prompt_dir=str(prompt_dir),
        eval_dir=str(router_dir),
        router_plan_path=str(plan_path),
        baseline=task.get("baseline_path", ""),
        entry_point=task.get("entry_point", ""),
    )
    proc = subprocess.run(
        command,
        shell=True,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60 * 60,
    )
    trace_path.write_text(proc.stdout, encoding="utf-8")
    router_plan = plan_path.read_text(encoding="utf-8") if plan_path.exists() else proc.stdout[-4000:]
    return {
        "task_id": task_id,
        "router_dir": str(router_dir),
        "router_plan_path": str(plan_path),
        "router_trace": str(trace_path),
        "router_returncode": proc.returncode,
        "router_plan": router_plan,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _evaluate_devin_candidate(
    repo_root: Path,
    candidate: dict[str, str],
    task: dict[str, Any],
    out_dir: Path,
    devin_command: str,
    compare_cmd: str,
    kernelbench_timeout_s: float,
    gepa_log,
    attempt_index: int = 0,
    team_size: int = 1,
    team_role: str = "solo",
    router_plan: str = "",
    router_plan_path: str = "",
) -> dict[str, Any]:
    def emit(*parts: Any) -> None:
        message = " ".join(str(part) for part in parts)
        if gepa_log:
            gepa_log(message)

    task_id = str(task.get("task_id", "unknown"))
    eval_dir = out_dir / "evals" / _safe_name(task_id) / f"agent_{attempt_index}_{int(time.time() * 1000)}"
    prompt_dir = eval_dir / "prompts"
    written = write_prompt_pack(repo_root, candidate, prompt_dir)
    emit("Task:", task_id)
    emit("Candidate roles:", ", ".join(sorted(candidate)))
    emit("Prompt files:", json.dumps({role: str(path) for role, path in written.items()}, sort_keys=True))
    emit("Black-box contract: candidate sees task spec and public docs, not hidden tests or scorer internals.")

    trace_path = eval_dir / "devin_trace.log"
    eval_dir.mkdir(parents=True, exist_ok=True)
    command = devin_command.format(
        task_id=task_id,
        prompt_dir=str(prompt_dir),
        eval_dir=str(eval_dir),
        baseline=task.get("baseline_path", ""),
        entry_point=task.get("entry_point", ""),
        attempt_index=attempt_index,
        team_size=team_size,
        team_role=team_role,
        router_plan=router_plan,
        router_plan_path=router_plan_path,
    )
    emit("Devin command:", command)
    proc = subprocess.run(
        command,
        shell=True,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60 * 60,
    )
    trace_path.write_text(proc.stdout, encoding="utf-8")
    emit("Devin return code:", proc.returncode)
    emit("Devin trace tail:", proc.stdout[-4000:])
    summary: dict[str, Any] = {
        "task_id": task_id,
        "eval_dir": str(eval_dir),
        "devin_returncode": proc.returncode,
        "devin_trace": str(trace_path),
        "team_attempt_index": attempt_index,
        "team_size": team_size,
        "team_role": team_role,
        "score": 0.0,
    }
    if proc.returncode != 0:
        summary["error"] = "devin_failed"
        return summary

    candidate_path = eval_dir / "candidate.py"
    if not candidate_path.exists():
        emit("Missing candidate:", candidate_path)
        summary["error"] = "missing_candidate"
        return summary

    kernel_tasks = eval_dir / "kernel_tasks.jsonl"
    kernel_tasks.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "baseline_path": task.get("baseline_path"),
                "candidate_path": str(candidate_path),
                "entry_point": task.get("entry_point"),
                "metadata": task.get("metadata", {}),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results_path = eval_dir / "kernelbench_results.jsonl"
    kb_cmd = [
        "inferencetimegepa",
        "run-kernelbench",
        "--tasks",
        str(kernel_tasks),
        "--compare-cmd",
            compare_cmd,
            "--timeout-s",
            str(kernelbench_timeout_s),
            "--zeus",
            "--out",
        str(results_path),
    ]
    proc = subprocess.run(kb_cmd, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    emit("KernelBench command:", " ".join(kb_cmd))
    emit("KernelBench output:", proc.stdout[-4000:])
    rows = _load_jsonl(results_path) if results_path.exists() else []
    summary["kernelbench_stdout"] = proc.stdout[-4000:]
    summary["kernelbench_results"] = str(results_path)
    if not rows:
        summary["error"] = "missing_kernelbench_results"
        return summary
    row = rows[0]
    emit("KernelBench result:", json.dumps(row, sort_keys=True))
    summary["score"] = float(row.get("score", 0.0))
    summary["kernelbench_row"] = row
    _track_eval(out_dir, summary)
    return summary


def _format_codex_packet(prompt_pack: dict[str, str], eval_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Codex GEPA Reflection Packet",
        "",
        "Goal: mutate the Devin team prompt pack using GEPA-style reflection.",
        "",
        "Rules:",
        "- Do not expose or infer hidden tests/scorer internals.",
        "- Optimize prompts only.",
        "- Preserve successful behavior and target recurring failures.",
        "- Candidate kernels are judged only by black-box KernelBench correctness, latency, and Zeus energy summaries.",
        "",
        "## Current Prompt Pack",
    ]
    for role, prompt in prompt_pack.items():
        lines.extend(["", f"### {role}", "", "```text", prompt.rstrip(), "```"])
    lines.extend(["", "## Evaluation Summaries"])
    for summary in eval_summaries:
        lines.extend(["", "```json", json.dumps(summary, indent=2, sort_keys=True), "```"])
    lines.extend(
        [
            "",
            "## Requested Mutation",
            "",
            "Return revised prompt text for any roles that should change, plus a short reflection explaining the failure pattern addressed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_result(out_dir: Path, result: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "to_dict"):
        (out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    best = getattr(result, "best_candidate", None)
    if isinstance(best, dict):
        best_dir = out_dir / "best_prompts"
        best_dir.mkdir(parents=True, exist_ok=True)
        for role, text in best.items():
            (best_dir / f"{role}.md").write_text(str(text).rstrip() + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:120]


def _track_eval(out_dir: Path, summary: dict[str, Any]) -> None:
    row = summary.get("kernelbench_row", {})
    record = {
        "time": time.time(),
        "task_id": summary.get("task_id"),
        "score": float(summary.get("score", 0.0)),
        "accepted": bool(row.get("accepted", False)),
        "correct": bool(row.get("correct", False)),
        "improved": bool(row.get("improved", False)),
        "reward": row.get("reward"),
        "baseline_latency_ms": row.get("baseline_latency_ms"),
        "candidate_latency_ms": row.get("candidate_latency_ms"),
        "latency_delta_ms": row.get("latency_delta_ms"),
        "baseline_energy_j": row.get("baseline_energy_j"),
        "candidate_energy_j": row.get("candidate_energy_j"),
        "energy_delta_j": row.get("energy_delta_j"),
        "error": row.get("error") or summary.get("error"),
        "eval_dir": summary.get("eval_dir"),
    }
    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _write_leaderboard(out_dir, _load_jsonl(metrics_path))
    _wandb_log(record, out_dir)


def _write_leaderboard(out_dir: Path, records: list[dict[str, Any]]) -> None:
    best = sorted(records, key=lambda r: (float(r.get("score") or 0.0), bool(r.get("accepted"))), reverse=True)
    lines = [
        "# Run Leaderboard",
        "",
        f"Total evals: {len(records)}",
        "",
        "| rank | task | score | accepted | correct | cand ms | base ms | cand J | error |",
        "|---:|---|---:|:---:|:---:|---:|---:|---:|---|",
    ]
    for idx, record in enumerate(best[:25], start=1):
        lines.append(
            "| {rank} | {task} | {score:.4g} | {accepted} | {correct} | {cand_ms} | {base_ms} | {cand_j} | {error} |".format(
                rank=idx,
                task=record.get("task_id", ""),
                score=float(record.get("score") or 0.0),
                accepted="yes" if record.get("accepted") else "no",
                correct="yes" if record.get("correct") else "no",
                cand_ms=_fmt(record.get("candidate_latency_ms")),
                base_ms=_fmt(record.get("baseline_latency_ms")),
                cand_j=_fmt(record.get("candidate_energy_j")),
                error=record.get("error") or "",
            )
        )
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wandb_log(record: dict[str, Any], out_dir: Path) -> None:
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_DEV_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_DEV_KEY"]
    if not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb

        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "inferencetimegepa"),
                name=os.environ.get("WANDB_RUN_NAME"),
                config={"tracker": "inferencetimegepa"},
            )
        wandb.log({k: v for k, v in record.items() if isinstance(v, (int, float, bool)) or v is None})
        for path in [
            out_dir / "metrics.jsonl",
            out_dir / "leaderboard.md",
            Path(str(record.get("eval_dir", ""))) / "candidate.py",
            Path(str(record.get("eval_dir", ""))) / "kernelbench_results.jsonl",
            Path(str(record.get("eval_dir", ""))) / "devin_trace.log",
        ]:
            if path.exists():
                wandb.save(str(path), base_path=str(out_dir))
    except Exception:
        return


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
