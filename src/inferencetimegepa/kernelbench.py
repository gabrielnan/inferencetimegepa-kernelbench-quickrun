from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelTask:
    task_id: str
    baseline_path: str
    candidate_path: str | None = None
    entry_point: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class KernelEvalResult:
    task_id: str
    correct: bool
    improved: bool
    baseline_energy_j: float | None
    candidate_energy_j: float | None
    energy_delta_j: float | None
    baseline_latency_ms: float | None
    candidate_latency_ms: float | None
    latency_delta_ms: float | None
    accepted: bool
    score: float
    reward: float
    error: str | None


def load_kernel_tasks(path: Path) -> list[KernelTask]:
    tasks: list[KernelTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                KernelTask(
                    task_id=row["task_id"],
                    baseline_path=row["baseline_path"],
                    candidate_path=row.get("candidate_path"),
                    entry_point=row.get("entry_point"),
                    metadata=row.get("metadata"),
                )
            )
    return tasks


def write_kernel_results(path: Path, results: list[KernelEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def run_kernelbench_eval(
    tasks: list[KernelTask],
    benchmark_cmd: str | None,
    zeus_enabled: bool,
    timeout_s: float,
    compare_cmd: str | None = None,
    allow_mock: bool = False,
) -> list[KernelEvalResult]:
    return [
        evaluate_kernel_task(
            task=task,
            benchmark_cmd=benchmark_cmd,
            compare_cmd=compare_cmd,
            zeus_enabled=zeus_enabled,
            timeout_s=timeout_s,
            allow_mock=allow_mock,
        )
        for task in tasks
    ]


def evaluate_kernel_task(
    task: KernelTask,
    benchmark_cmd: str | None,
    compare_cmd: str | None,
    zeus_enabled: bool,
    timeout_s: float,
    allow_mock: bool = False,
) -> KernelEvalResult:
    if not task.candidate_path:
        return KernelEvalResult(
            task_id=task.task_id,
            correct=True,
            improved=False,
            baseline_energy_j=None,
            candidate_energy_j=None,
            energy_delta_j=None,
            baseline_latency_ms=None,
            candidate_latency_ms=None,
            latency_delta_ms=None,
            accepted=False,
            score=0.0,
            reward=-1.0,
            error="missing_candidate",
        )

    if compare_cmd:
        measured = _run_compare_command(task, compare_cmd, zeus_enabled, timeout_s)
        baseline = _baseline_from_compare(measured)
        candidate = _candidate_from_compare(measured)
    else:
        baseline = _run_kernelbench_command(task, task.baseline_path, benchmark_cmd, zeus_enabled, timeout_s, allow_mock)
        candidate = _run_kernelbench_command(task, task.candidate_path, benchmark_cmd, zeus_enabled, timeout_s, allow_mock)
    if baseline.get("error") or candidate.get("error"):
        return KernelEvalResult(
            task_id=task.task_id,
            correct=False,
            improved=False,
            baseline_energy_j=baseline.get("energy_j"),
            candidate_energy_j=candidate.get("energy_j"),
            energy_delta_j=_delta(candidate.get("energy_j"), baseline.get("energy_j")),
            baseline_latency_ms=baseline.get("latency_ms"),
            candidate_latency_ms=candidate.get("latency_ms"),
            latency_delta_ms=_delta(candidate.get("latency_ms"), baseline.get("latency_ms")),
            accepted=False,
            score=0.0,
            reward=-1.0,
            error=baseline.get("error") or candidate.get("error"),
        )

    correct = bool(candidate.get("correct", False))
    baseline_energy = baseline.get("energy_j")
    candidate_energy = candidate.get("energy_j")
    baseline_latency = baseline.get("latency_ms")
    candidate_latency = candidate.get("latency_ms")
    improved = correct and _is_improved(
        baseline_energy=baseline_energy,
        candidate_energy=candidate_energy,
        baseline_latency=baseline_latency,
        candidate_latency=candidate_latency,
    )
    score = _score(correct, improved, baseline_energy, candidate_energy, baseline_latency, candidate_latency)
    accepted = correct and improved
    return KernelEvalResult(
        task_id=task.task_id,
        correct=correct,
        improved=improved,
        baseline_energy_j=baseline_energy,
        candidate_energy_j=candidate_energy,
        energy_delta_j=_delta(candidate_energy, baseline_energy),
        baseline_latency_ms=baseline_latency,
        candidate_latency_ms=candidate_latency,
        latency_delta_ms=_delta(candidate_latency, baseline_latency),
        accepted=accepted,
        score=score,
        reward=_reward(correct, baseline_energy, candidate_energy, baseline_latency, candidate_latency),
        error=None,
    )


def _run_kernelbench_command(
    task: KernelTask,
    kernel_path: str,
    benchmark_cmd: str | None,
    zeus_enabled: bool,
    timeout_s: float,
    allow_mock: bool,
) -> dict:
    if not benchmark_cmd:
        if not allow_mock:
            return {"correct": False, "error": "missing_benchmark_cmd"}
        return _mock_kernelbench_result(kernel_path)

    cmd = _format_command(
        benchmark_cmd,
        task=task,
        kernel=kernel_path,
        baseline=task.baseline_path,
        candidate=task.candidate_path or "",
    )
    return _run_json_command(cmd, zeus_enabled, timeout_s)


def _run_compare_command(
    task: KernelTask,
    compare_cmd: str,
    zeus_enabled: bool,
    timeout_s: float,
) -> dict:
    cmd = _format_command(
        compare_cmd,
        task=task,
        kernel=task.candidate_path or "",
        baseline=task.baseline_path,
        candidate=task.candidate_path or "",
    )
    return _run_json_command(cmd, zeus_enabled, timeout_s)


def _run_json_command(cmd: str, zeus_enabled: bool, timeout_s: float) -> dict:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return {"correct": False, "error": "timeout"}

    elapsed_ms = (time.monotonic() - started) * 1000
    if proc.returncode != 0:
        return {"correct": False, "latency_ms": elapsed_ms, "error": (stderr or stdout)[-2000:]}
    parsed = _parse_benchmark_stdout(stdout)
    parsed.setdefault("latency_ms", elapsed_ms)
    if zeus_enabled:
        parsed.setdefault("energy_j", None)
    return parsed


def _format_command(command: str, task: KernelTask, kernel: str, baseline: str, candidate: str) -> str:
    replacements = {
        "kernel": kernel,
        "baseline": baseline,
        "candidate": candidate,
        "task_id": task.task_id,
        "entry_point": task.entry_point or "",
    }
    if task.metadata:
        replacements.update({str(key): str(value) for key, value in task.metadata.items()})
    formatted = command
    for key, value in replacements.items():
        formatted = formatted.replace("{" + key + "}", value)
    return formatted


def _parse_benchmark_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        return {
            "correct": bool(row.get("correct", row.get("valid", False))),
            "energy_j": row.get("energy_j"),
            "latency_ms": row.get("latency_ms"),
            "baseline_energy_j": row.get("baseline_energy_j"),
            "candidate_energy_j": row.get("candidate_energy_j"),
            "baseline_latency_ms": row.get("baseline_latency_ms"),
            "candidate_latency_ms": row.get("candidate_latency_ms"),
            "baseline_correct": row.get("baseline_correct"),
            "candidate_correct": row.get("candidate_correct"),
        }
    return {"correct": True}


def _mock_kernelbench_result(kernel_path: str) -> dict:
    text = Path(kernel_path).read_text(encoding="utf-8") if Path(kernel_path).exists() else kernel_path
    latency = 1.0 if "fast" in text else 2.0
    energy = 1.0 if "efficient" in text else 2.0
    return {"correct": "incorrect" not in text, "latency_ms": latency, "energy_j": energy}


def _baseline_from_compare(row: dict) -> dict:
    return {
        "correct": bool(row.get("baseline_correct", True)),
        "energy_j": row.get("baseline_energy_j"),
        "latency_ms": row.get("baseline_latency_ms"),
        "error": row.get("baseline_error") or row.get("error"),
    }


def _candidate_from_compare(row: dict) -> dict:
    return {
        "correct": bool(row.get("candidate_correct", row.get("correct", row.get("valid", False)))),
        "energy_j": row.get("candidate_energy_j", row.get("energy_j")),
        "latency_ms": row.get("candidate_latency_ms", row.get("latency_ms")),
        "error": row.get("candidate_error") or row.get("error"),
    }


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _is_improved(
    baseline_energy: float | None,
    candidate_energy: float | None,
    baseline_latency: float | None,
    candidate_latency: float | None,
) -> bool:
    if baseline_energy is not None and candidate_energy is not None:
        return candidate_energy < baseline_energy
    if baseline_latency is not None and candidate_latency is not None:
        return candidate_latency < baseline_latency
    return False


def _score(
    correct: bool,
    improved: bool,
    baseline_energy: float | None,
    candidate_energy: float | None,
    baseline_latency: float | None,
    candidate_latency: float | None,
) -> float:
    if not correct:
        return 0.0
    if not improved:
        return 0.5
    if baseline_energy and candidate_energy:
        return max(0.5, baseline_energy / max(candidate_energy, 1e-9))
    if baseline_latency and candidate_latency:
        return max(0.5, baseline_latency / max(candidate_latency, 1e-9))
    return 1.0


def _reward(
    correct: bool,
    baseline_energy: float | None,
    candidate_energy: float | None,
    baseline_latency: float | None,
    candidate_latency: float | None,
) -> float:
    if not correct:
        return -1.0
    if baseline_energy and candidate_energy:
        relative_delta = (baseline_energy - candidate_energy) / max(baseline_energy, 1e-9)
        if relative_delta > 0:
            return 1.0 + min(relative_delta, 2.0)
        return max(0.05, 0.25 + max(relative_delta, -0.2))
    if baseline_latency and candidate_latency:
        relative_delta = (baseline_latency - candidate_latency) / max(baseline_latency, 1e-9)
        if relative_delta > 0:
            return 0.5 + min(relative_delta, 1.0)
        return max(0.05, 0.2 + max(relative_delta, -0.15))
    return 0.1
