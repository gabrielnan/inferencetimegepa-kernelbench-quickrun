from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--entry-point", default="ModelNew")
    parser.add_argument("--host", default=os.environ.get("PRIME_HOST", "ubuntu@209.20.158.160"))
    parser.add_argument("--key", default=os.environ.get("PRIME_SSH_KEY", str(Path.home() / ".ssh/primeintellect_ed25519")))
    parser.add_argument("--remote-root", default=os.environ.get("PRIME_REMOTE_ROOT", "/home/ubuntu/inferencetimegepa"))
    parser.add_argument("--kernelbench-root", default=os.environ.get("PRIME_KERNELBENCH_ROOT", "/home/ubuntu/KernelBench"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=5)
    parser.add_argument("--warmup-compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    task_dir = f"{args.remote_root}/runs/prime_eval/{_safe_name(Path(args.candidate).stem)}"
    _ssh(args, f"mkdir -p {shlex.quote(task_dir)}")
    remote_baseline = f"{task_dir}/baseline.py"
    remote_candidate = f"{task_dir}/candidate.py"
    remote_baseline_candidate = f"{task_dir}/baseline_candidate.py"
    remote_runner = f"{task_dir}/run_with_zeus.py"
    _scp(args, args.baseline, remote_baseline)
    _scp(args, args.candidate, remote_candidate)
    _scp_baseline_candidate(args, args.baseline, remote_baseline_candidate)
    _write_remote_runner(args, remote_runner)

    command = (
        f"cd {shlex.quote(args.kernelbench_root)} && "
        "PYBIND_INC=$(python3 -c 'import pybind11; print(pybind11.get_include())') && "
        "PATH=$HOME/.local/bin:$PATH "
        f"PYTHONPATH={shlex.quote(args.kernelbench_root)}/src "
        "CPATH=$PYBIND_INC CPLUS_INCLUDE_PATH=$PYBIND_INC "
        f"python3 {shlex.quote(remote_runner)} "
        f"--kernelbench-root {shlex.quote(args.kernelbench_root)} "
        f"--baseline {shlex.quote(remote_baseline)} "
        f"--candidate {shlex.quote(remote_candidate)} "
        f"--baseline-candidate {shlex.quote(remote_baseline_candidate)} "
        f"--num-correct-trials {args.num_correct_trials} "
        f"--num-perf-trials {args.num_perf_trials} "
        f"--timeout {args.timeout} "
        f"{'--warmup-compile' if args.warmup_compile else '--no-warmup-compile'}"
    )
    proc = _ssh(args, command, check=False)
    row = _parse_kernelbench_stdout(proc.stdout)
    row["stdout_tail"] = proc.stdout[-4000:]
    if proc.returncode != 0:
        row["candidate_correct"] = False
        row["candidate_error"] = f"remote_scorer_failed:{proc.returncode}"
    print(json.dumps(row, sort_keys=True))


def _ssh(args, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", args.host, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def _scp(args, local: str, remote: str) -> None:
    subprocess.run(
        ["scp", "-i", args.key, "-o", "IdentitiesOnly=yes", local, f"{args.host}:{remote}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _scp_baseline_candidate(args, baseline: str, remote: str) -> None:
    source = Path(baseline).read_text(encoding="utf-8").rstrip() + "\n\nModelNew = Model\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        tmp_path = f.name
    try:
        _scp(args, tmp_path, remote)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _write_remote_runner(args, remote: str) -> None:
    source = textwrap.dedent(
        r'''
        from __future__ import annotations

        import argparse
        import contextlib
        import io
        import json
        import os
        import re
        import subprocess
        import sys


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--kernelbench-root", required=True)
            parser.add_argument("--baseline", required=True)
            parser.add_argument("--candidate", required=True)
            parser.add_argument("--baseline-candidate", required=True)
            parser.add_argument("--num-correct-trials", type=int, required=True)
            parser.add_argument("--num-perf-trials", type=int, required=True)
            parser.add_argument("--timeout", type=int, required=True)
            parser.add_argument("--warmup-compile", action=argparse.BooleanOptionalAction, default=True)
            args = parser.parse_args()

            def command_for(kernel_path: str, clear_cache: bool, check_kernel: bool) -> list[str]:
                return [
                sys.executable,
                "scripts/run_and_check.py",
                "ref_origin=local",
                f"ref_arch_src_path={args.baseline}",
                f"kernel_src_path={kernel_path}",
                "eval_mode=local",
                "gpu_arch=['Hopper']",
                f"num_correct_trials={args.num_correct_trials}",
                f"num_perf_trials={args.num_perf_trials}",
                f"timeout={args.timeout}",
                f"check_kernel={check_kernel}",
                f"clear_cache={clear_cache}",
                ]

            env = dict(os.environ)
            env["PYTHONHASHSEED"] = "0"
            warmup_stdout = ""
            if args.warmup_compile:
                baseline_warmup = subprocess.run(command_for(args.baseline_candidate, True, False), cwd=args.kernelbench_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                candidate_warmup = subprocess.run(command_for(args.candidate, True, True), cwd=args.kernelbench_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                warmup_stdout = "[BASELINE_WARMUP]\n" + baseline_warmup.stdout + "\n[CANDIDATE_WARMUP]\n" + candidate_warmup.stdout
                if baseline_warmup.returncode != 0 or candidate_warmup.returncode != 0:
                    baseline = parse_stdout(baseline_warmup.stdout)
                    baseline["energy_j"] = None
                    baseline["stdout"] = baseline_warmup.stdout
                    baseline["returncode"] = baseline_warmup.returncode
                    candidate = parse_stdout(candidate_warmup.stdout)
                    candidate["energy_j"] = None
                    candidate["stdout"] = candidate_warmup.stdout
                    candidate["returncode"] = candidate_warmup.returncode
                    out = {
                        "baseline": baseline,
                        "candidate": candidate,
                        "stdout": warmup_stdout,
                        "returncode": candidate_warmup.returncode or baseline_warmup.returncode,
                        "warmup_compile": args.warmup_compile,
                        "warmup_failed": True,
                    }
                    print(json.dumps(out, sort_keys=True))
                    return

            def measured_run(label: str, kernel_path: str, check_kernel: bool) -> dict:
                from zeus.monitor import ZeusMonitor
                monitor = ZeusMonitor(gpu_indices=[0])
                monitor.begin_window(label)
                proc = subprocess.run(command_for(kernel_path, False, check_kernel), cwd=args.kernelbench_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                measurement = monitor.end_window(label)
                energy_j = sum(float(v) for v in measurement.gpu_energy.values())
                parsed = parse_stdout(proc.stdout)
                parsed["energy_j"] = energy_j
                parsed["stdout"] = proc.stdout
                parsed["returncode"] = proc.returncode
                return parsed

            try:
                baseline = measured_run("baseline", args.baseline_candidate, False)
                candidate = measured_run("candidate", args.candidate, True)
            except Exception as exc:
                proc = subprocess.run(command_for(args.candidate, False, True), cwd=args.kernelbench_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                candidate = parse_stdout(proc.stdout)
                candidate["energy_j"] = None
                candidate["stdout"] = proc.stdout + f"\n[ZEUS_ERROR] {exc!r}\n"
                candidate["returncode"] = proc.returncode
                baseline = {}

            out = {
                "baseline": baseline,
                "candidate": candidate,
                "stdout": (
                    "[WARMUP_COMPILE_OUTPUT]\n"
                    + warmup_stdout[-4000:]
                    + "\n[BASELINE_MEASURED_OUTPUT]\n"
                    + str(baseline.get("stdout", ""))[-4000:]
                    + "\n[CANDIDATE_MEASURED_OUTPUT]\n"
                    + str(candidate.get("stdout", ""))[-4000:]
                ),
                "returncode": candidate.get("returncode", 1),
                "warmup_compile": args.warmup_compile,
            }
            print(json.dumps(out, sort_keys=True))


        def parse_stdout(stdout: str) -> dict:
            return {
                "compiled": "compiled=True" in stdout,
                "correct": "correctness=True" in stdout,
                "baseline_latency_ms": extract_float(stdout, r"PyTorch Reference Eager exec time:\s*([0-9.]+)"),
                "candidate_latency_ms": extract_float(stdout, r"Custom Kernel exec time:\s*([0-9.]+)"),
                "torch_compile_latency_ms": extract_float(stdout, r"PyTorch Reference torch\.compile time:\s*([0-9.]+)"),
            }


        def extract_float(text: str, pattern: str) -> float | None:
            match = re.search(pattern, text)
            if not match:
                return None
            return float(match.group(1))


        if __name__ == "__main__":
            main()
        '''
    ).lstrip()
    proc = subprocess.run(
        ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", args.host, f"cat > {shlex.quote(remote)}"],
        input=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _parse_kernelbench_stdout(stdout: str) -> dict:
    parsed = _last_json(stdout)
    if parsed:
        baseline = parsed.get("baseline", {})
        candidate = parsed.get("candidate", parsed)
        compiled = bool(candidate.get("compiled"))
        correct = bool(candidate.get("correct"))
        returncode = int(candidate.get("returncode", parsed.get("returncode", 1)) or 0)
        warmup_failed = bool(parsed.get("warmup_failed"))
        custom = candidate.get("candidate_latency_ms")
        eager = candidate.get("baseline_latency_ms") or baseline.get("baseline_latency_ms")
        compiled_ref = candidate.get("torch_compile_latency_ms")
        baseline_energy = baseline.get("energy_j")
        candidate_energy = candidate.get("energy_j")
        stdout_tail = str(parsed.get("stdout", stdout))[-4000:]
    else:
        compiled = "compiled=True" in stdout
        correct = "correctness=True" in stdout
        custom = _extract_float(stdout, r"Custom Kernel exec time:\s*([0-9.]+)")
        eager = _extract_float(stdout, r"PyTorch Reference Eager exec time:\s*([0-9.]+)")
        compiled_ref = _extract_float(stdout, r"PyTorch Reference torch\.compile time:\s*([0-9.]+)")
        baseline_energy = None
        candidate_energy = None
        stdout_tail = stdout[-4000:]
        returncode = 0
        warmup_failed = False
    return {
        "baseline_correct": True,
        "candidate_correct": bool(compiled and correct and returncode == 0 and not warmup_failed),
        "baseline_latency_ms": eager,
        "candidate_latency_ms": custom,
        "torch_compile_latency_ms": compiled_ref,
        "baseline_energy_j": baseline_energy,
        "candidate_energy_j": candidate_energy,
        "energy_j": candidate_energy,
        "stdout_tail": stdout_tail,
    }


def _extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _last_json(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:80]


if __name__ == "__main__":
    main()
