from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    baseline_fn = _load_symbol(Path(args.baseline), args.entry_point)
    candidate_fn = _load_symbol(Path(args.candidate), args.entry_point)
    cases = _cases(args.entry_point)

    try:
        for case in cases:
            expected = baseline_fn(*case)
            actual = candidate_fn(*case)
            if actual != expected:
                print(
                    json.dumps(
                        {
                            "baseline_correct": True,
                            "candidate_correct": False,
                            "candidate_error": f"mismatch for args={case!r}: expected {expected!r}, got {actual!r}",
                        }
                    )
                )
                return
        baseline_latency = _time_fn(baseline_fn, cases, args.repeats)
        candidate_latency = _time_fn(candidate_fn, cases, args.repeats)
        print(
            json.dumps(
                {
                    "baseline_correct": True,
                    "candidate_correct": True,
                    "baseline_latency_ms": baseline_latency,
                    "candidate_latency_ms": candidate_latency,
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "baseline_correct": True,
                    "candidate_correct": False,
                    "candidate_error": repr(exc),
                }
            )
        )


def _load_symbol(path: Path, entry_point: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"kernel_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry_point)


def _cases(entry_point: str) -> list[tuple[Any, ...]]:
    if entry_point == "kernel_fn":
        return [([1, 2, 3],), ([],), ([0, -1, 5, 9],), ([42],)]
    return [(1,), (2,), (3,)]


def _time_fn(fn, cases: list[tuple[Any, ...]], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        for i in range(repeats):
            fn(*cases[i % len(cases)])
        samples.append((time.perf_counter() - started) * 1000 / repeats)
    return statistics.median(samples)


if __name__ == "__main__":
    main()
