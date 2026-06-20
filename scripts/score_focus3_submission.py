from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FOCUS3_TASKS = {
    "mlp_small": "benchmarks/kernelbench_focus3/mlp_small.py",
    "matmul_gelu_softmax_small": "benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py",
    "mingpt_causal_attention_small": "benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a Focus3 submission with the controlled leaderboard protocol. "
            "The output JSON separates KernelBench time-track latency from the "
            "warmed repeated-forward energy-track window."
        )
    )
    parser.add_argument("--task", choices=sorted(FOCUS3_TASKS), required=True)
    parser.add_argument("--baseline", help="Override the baseline path for --task.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--entry-point", default="ModelNew")
    parser.add_argument("--host", default=os.environ.get("PRIME_HOST", "ubuntu@209.20.158.160"))
    parser.add_argument("--key", default=os.environ.get("PRIME_SSH_KEY", str(Path.home() / ".ssh/primeintellect_ed25519")))
    parser.add_argument("--remote-root", default=os.environ.get("PRIME_REMOTE_ROOT", "/home/ubuntu/inferencetimegepa"))
    parser.add_argument("--kernelbench-root", default=os.environ.get("PRIME_KERNELBENCH_ROOT", "/home/ubuntu/KernelBench"))
    parser.add_argument("--output", type=Path, help="Write normalized JSON here.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--measure-iters", type=int, default=200_000)
    parser.add_argument(
        "--energy-orders",
        default="candidate,baseline;baseline,candidate",
        help="Semicolon-separated measurement orders for energy scoring.",
    )
    parser.add_argument("--skip-time", action="store_true", help="Skip KernelBench run_and_check latency scoring.")
    parser.add_argument("--skip-energy", action="store_true", help="Skip warmed repeated-forward energy scoring.")
    parser.add_argument(
        "--allow-energy-after-time-fail",
        action="store_true",
        help="Run the energy phase even if the KernelBench time/correctness phase fails.",
    )
    parser.add_argument("--fail-on-invalid", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    baseline = Path(args.baseline or FOCUS3_TASKS[args.task])
    candidate = Path(args.candidate)
    if not baseline.exists():
        parser.error(f"baseline does not exist: {baseline}")
    if not candidate.exists():
        parser.error(f"candidate does not exist: {candidate}")

    started = time.time()
    row: dict[str, Any] = {
        "schema_version": 1,
        "task": args.task,
        "baseline_path": str(baseline),
        "candidate_path": str(candidate),
        "entry_point": args.entry_point,
        "host": args.host,
        "kernelbench_root": args.kernelbench_root,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracks": {
            "time": {
                "measurement_scope": "kernelbench_run_and_check_cuda_event_latency",
                "lower_is_better": True,
                "skipped": bool(args.skip_time),
            },
            "energy": {
                "measurement_scope": "official_kernel_only_repeated_forward_pair",
                "lower_is_better": True,
                "warmup_iters": args.warmup_iters,
                "measure_iters": args.measure_iters,
                "orders": args.energy_orders,
                "skipped": bool(args.skip_energy),
            },
        },
    }

    if not args.skip_time:
        row["tracks"]["time"].update(run_time_track(args, baseline, candidate))
    time_failed = bool(
        not args.skip_time
        and not row["tracks"]["time"].get("candidate_correct")
    )
    if args.skip_energy:
        pass
    elif time_failed and not args.allow_energy_after_time_fail:
        row["tracks"]["energy"].update(
            {
                "skipped": True,
                "skip_reason": "time_track_failed",
                "candidate_correct": False,
            }
        )
    else:
        row["tracks"]["energy"].update(run_energy_track(args, baseline, candidate))

    row["elapsed_s"] = time.time() - started
    row["valid"] = is_valid(row)
    row["accepted_for_time_track"] = bool(
        row["tracks"]["time"].get("candidate_correct")
        and row["tracks"]["time"].get("candidate_latency_ms") is not None
    )
    row["accepted_for_energy_track"] = bool(
        row["tracks"]["energy"].get("candidate_correct")
        and row["tracks"]["energy"].get("candidate_energy_j") is not None
    )

    output = json.dumps(row, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(json.dumps(row, sort_keys=True))

    if args.fail_on_invalid and not row["valid"]:
        raise SystemExit(2)


def run_time_track(args: argparse.Namespace, baseline: Path, candidate: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/prime_kernelbench_pair.py",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--entry-point",
        args.entry_point,
        "--host",
        args.host,
        "--key",
        args.key,
        "--remote-root",
        args.remote_root,
        "--kernelbench-root",
        args.kernelbench_root,
        "--num-correct-trials",
        str(args.num_correct_trials),
        "--num-perf-trials",
        str(args.num_perf_trials),
        "--timeout",
        str(args.timeout),
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    parsed = last_json(proc.stdout) or {}
    return {
        "command": command,
        "returncode": proc.returncode,
        "candidate_correct": bool(parsed.get("candidate_correct")) and proc.returncode == 0,
        "baseline_latency_ms": parsed.get("baseline_latency_ms"),
        "candidate_latency_ms": parsed.get("candidate_latency_ms"),
        "torch_compile_latency_ms": parsed.get("torch_compile_latency_ms"),
        "candidate_energy_j_diagnostic_subprocess_window": parsed.get("candidate_energy_j"),
        "baseline_energy_j_diagnostic_subprocess_window": parsed.get("baseline_energy_j"),
        "stdout_tail": proc.stdout[-4000:],
    }


def run_energy_track(args: argparse.Namespace, baseline: Path, candidate: Path) -> dict[str, Any]:
    task_dir = f"{args.remote_root}/runs/focus3_submission_score/{safe_name(args.task + '_' + candidate.stem)}"
    remote_baseline = f"{task_dir}/baseline.py"
    remote_candidate = f"{task_dir}/candidate.py"
    remote_runner = f"{task_dir}/score_energy.py"

    try:
        ssh(args, f"mkdir -p {shlex.quote(task_dir)}")
        scp(args, baseline, remote_baseline)
        scp(args, candidate, remote_candidate)
        write_remote_energy_runner(args, remote_runner)
    except subprocess.CalledProcessError as exc:
        return {
            "candidate_correct": False,
            "error": f"remote_setup_failed:{exc.returncode}",
            "stdout_tail": str(exc.stdout or exc.output or "")[-4000:],
        }

    command = (
        f"cd {shlex.quote(args.kernelbench_root)} && "
        "PYBIND_INC=$(python3 -c 'import pybind11; print(pybind11.get_include())') && "
        "PATH=$HOME/.local/bin:$PATH "
        f"PYTHONPATH={shlex.quote(args.kernelbench_root)}/src "
        "CPATH=$PYBIND_INC CPLUS_INCLUDE_PATH=$PYBIND_INC "
        f"python3 {shlex.quote(remote_runner)} "
        f"--baseline {shlex.quote(remote_baseline)} "
        f"--candidate {shlex.quote(remote_candidate)} "
        f"--entry-point {shlex.quote(args.entry_point)} "
        f"--warmup-iters {args.warmup_iters} "
        f"--measure-iters {args.measure_iters} "
        f"--orders {shlex.quote(args.energy_orders)}"
    )
    proc = ssh(args, command, check=False)
    parsed = last_json(proc.stdout) or {}
    parsed["command"] = command
    parsed["returncode"] = proc.returncode
    parsed["stdout_tail"] = proc.stdout[-4000:]
    if proc.returncode != 0:
        parsed["candidate_correct"] = False
        parsed.setdefault("error", f"remote_energy_scorer_failed:{proc.returncode}")
    return parsed


def write_remote_energy_runner(args: argparse.Namespace, remote: str) -> None:
    source = textwrap.dedent(
        r'''
        from __future__ import annotations

        import argparse
        import importlib.util
        import json
        import math
        import subprocess
        import sys
        import time
        from pathlib import Path

        import torch


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--baseline", required=True)
            parser.add_argument("--candidate", required=True)
            parser.add_argument("--entry-point", required=True)
            parser.add_argument("--warmup-iters", type=int, required=True)
            parser.add_argument("--measure-iters", type=int, required=True)
            parser.add_argument("--orders", required=True)
            args = parser.parse_args()

            try:
                out = run(args)
            except Exception as exc:
                out = {
                    "candidate_correct": False,
                    "error": repr(exc),
                }
            print(json.dumps(out, sort_keys=True))


        def run(args) -> dict:
            torch.manual_seed(0)
            torch.set_grad_enabled(False)
            baseline_mod = load_module(Path(args.baseline), "focus3_score_baseline")
            candidate_mod = load_module(Path(args.candidate), "focus3_score_candidate")

            baseline_model = baseline_mod.Model(*baseline_mod.get_init_inputs()).cuda().eval()
            candidate_cls = getattr(candidate_mod, args.entry_point)
            candidate_model = candidate_cls(*baseline_mod.get_init_inputs()).cuda().eval()
            candidate_model.load_state_dict(baseline_model.state_dict(), strict=False)
            inputs = [
                item.cuda(non_blocking=True) if torch.is_tensor(item) else item
                for item in baseline_mod.get_inputs()
            ]

            expected = baseline_model(*inputs)
            actual = candidate_model(*inputs)
            torch.cuda.synchronize()
            max_abs_error = float((expected - actual).abs().max().item())
            max_rel_error = float(((expected - actual).abs() / expected.abs().clamp_min(1.0e-12)).max().item())
            candidate_correct = bool(torch.allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4))
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,pci.bus_id,power.limit",
                    "--format=csv,noheader",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            ).stdout.strip()
            if not candidate_correct:
                return {
                    "candidate_correct": False,
                    "max_abs_error": max_abs_error,
                    "max_rel_error": max_rel_error,
                    "gpu": gpu,
                }

            models = {
                "baseline": baseline_model,
                "candidate": candidate_model,
            }
            measurements = []
            for order_text in args.orders.split(";"):
                order = [part.strip() for part in order_text.split(",") if part.strip()]
                for name in order:
                    stats = measure_forward(models[name], inputs, args.warmup_iters, args.measure_iters, f"{name}_forward")
                    stats["name"] = name
                    stats["order"] = order_text
                    measurements.append(stats)
            candidate_stats = aggregate_measurements(measurements, "candidate")
            baseline_stats = aggregate_measurements(measurements, "baseline")
            return {
                "candidate_correct": True,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
                "gpu": gpu,
                "baseline_latency_ms": baseline_stats["latency_ms"],
                "candidate_latency_ms": candidate_stats["latency_ms"],
                "baseline_energy_j": baseline_stats["energy_j"],
                "candidate_energy_j": candidate_stats["energy_j"],
                "baseline_energy_per_iter_j": baseline_stats["energy_per_iter_j"],
                "candidate_energy_per_iter_j": candidate_stats["energy_per_iter_j"],
                "baseline_avg_power_w_cuda_time": baseline_stats["avg_power_w_cuda_time"],
                "candidate_avg_power_w_cuda_time": candidate_stats["avg_power_w_cuda_time"],
                "baseline_avg_power_w_wall_time": baseline_stats["avg_power_w_wall_time"],
                "candidate_avg_power_w_wall_time": candidate_stats["avg_power_w_wall_time"],
                "baseline_iters_per_second": baseline_stats["iters_per_second"],
                "candidate_iters_per_second": candidate_stats["iters_per_second"],
                "measurements": measurements,
                "orders": args.orders,
                "zeus_available": bool(candidate_stats["zeus_available"] and baseline_stats["zeus_available"]),
            }


        def measure_forward(model, inputs, warmup_iters: int, measure_iters: int, label: str) -> dict:
            for _ in range(warmup_iters):
                model(*inputs)
            torch.cuda.synchronize()

            monitor = None
            zeus_available = False
            try:
                from zeus.monitor import ZeusMonitor

                monitor = ZeusMonitor(gpu_indices=[0])
                zeus_available = True
            except Exception:
                monitor = None

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if monitor is not None:
                monitor.begin_window(label)
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(measure_iters):
                model(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            energy_j = None
            if monitor is not None:
                measurement = monitor.end_window(label)
                energy_j = sum(float(v) for v in measurement.gpu_energy.values())
            elapsed_ms = float(start_event.elapsed_time(end_event))
            total_cuda_event_s = elapsed_ms / 1000.0
            return {
                "latency_ms": elapsed_ms / measure_iters,
                "energy_j": energy_j,
                "energy_per_iter_j": None if energy_j is None else energy_j / measure_iters,
                "total_cuda_event_s": total_cuda_event_s,
                "wall_s": wall_seconds,
                "avg_power_w_cuda_time": (
                    None if energy_j is None or total_cuda_event_s <= 0 else energy_j / total_cuda_event_s
                ),
                "avg_power_w_wall_time": (
                    None if energy_j is None or wall_seconds <= 0 else energy_j / wall_seconds
                ),
                "iters_per_second": measure_iters / wall_seconds if wall_seconds > 0 else math.inf,
                "zeus_available": zeus_available,
            }


        def aggregate_measurements(measurements: list[dict], name: str) -> dict:
            rows = [row for row in measurements if row["name"] == name]
            if not rows:
                raise RuntimeError(f"missing measurements for {name}")
            keys = [
                "latency_ms",
                "energy_j",
                "energy_per_iter_j",
                "total_cuda_event_s",
                "wall_s",
                "avg_power_w_cuda_time",
                "avg_power_w_wall_time",
                "iters_per_second",
            ]
            out = {}
            for key in keys:
                values = [row[key] for row in rows if row.get(key) is not None]
                out[key] = None if not values else sum(float(v) for v in values) / len(values)
            out["zeus_available"] = all(bool(row.get("zeus_available")) for row in rows)
            out["repeats"] = len(rows)
            return out


        def load_module(path: Path, name: str):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module


        if __name__ == "__main__":
            main()
        '''
    ).lstrip()
    subprocess.run(
        ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", args.host, f"cat > {shlex.quote(remote)}"],
        input=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def ssh(args: argparse.Namespace, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", args.host, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def scp(args: argparse.Namespace, local: Path, remote: str) -> None:
    subprocess.run(
        ["scp", "-i", args.key, "-o", "IdentitiesOnly=yes", str(local), f"{args.host}:{remote}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def is_valid(row: dict[str, Any]) -> bool:
    time_track = row["tracks"]["time"]
    energy_track = row["tracks"]["energy"]
    time_ok = bool(time_track.get("skipped") or time_track.get("candidate_correct"))
    energy_ok = bool(energy_track.get("skipped") or energy_track.get("candidate_correct"))
    return time_ok and energy_ok


def last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:120]


if __name__ == "__main__":
    main()
