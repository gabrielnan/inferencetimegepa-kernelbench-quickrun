from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from inferencetimegepa.code_extract import extract_python_code
from inferencetimegepa.tasks import CodeTask


@dataclass(frozen=True)
class EvalResult:
    score: float
    valid: bool
    tests_passed: int
    total_tests: int
    elapsed_s: float
    error: str | None


def evaluate_completion(task: CodeTask, completion: str, timeout_s: float = 5.0) -> EvalResult:
    code = extract_python_code(completion)
    harness = _build_harness(code, task.tests)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="inferencetime_eval_") as tmp:
        path = Path(tmp) / "candidate_test.py"
        path.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(
                score=0.0,
                valid=False,
                tests_passed=0,
                total_tests=len(task.tests),
                elapsed_s=time.monotonic() - started,
                error="timeout",
            )

    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        return EvalResult(
            score=0.0,
            valid=False,
            tests_passed=0,
            total_tests=len(task.tests),
            elapsed_s=elapsed,
            error=(proc.stderr or proc.stdout).strip()[-2000:] or f"exit {proc.returncode}",
        )

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return EvalResult(
            score=0.0,
            valid=False,
            tests_passed=0,
            total_tests=len(task.tests),
            elapsed_s=elapsed,
            error=f"invalid evaluator output: {proc.stdout[-500:]}",
        )

    passed = int(payload["passed"])
    total = int(payload["total"])
    score = passed / total if total else 0.0
    return EvalResult(
        score=score,
        valid=passed == total,
        tests_passed=passed,
        total_tests=total,
        elapsed_s=elapsed,
        error=payload.get("error"),
    )


def _build_harness(code: str, tests: tuple[str, ...]) -> str:
    tests_literal = repr(list(tests))
    return (
        "import json\n"
        "import traceback\n"
        "from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple\n\n"
        f"{code}\n\n"
        f"tests = {tests_literal}\n"
        "passed = 0\n"
        "error = None\n"
        "for test in tests:\n"
        "    try:\n"
        "        exec(test, globals(), globals())\n"
        "        passed += 1\n"
        "    except Exception:\n"
        "        error = traceback.format_exc(limit=2)\n"
        "        break\n\n"
        'print(json.dumps({"passed": passed, "total": len(tests), "error": error}))\n'
    )
