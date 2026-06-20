# Task: Benchmark Engineer

Run correctness and benchmark comparisons for reviewed candidate kernels.

Inputs:
- Candidate kernel under `runs/candidates/`.
- Baseline kernel path.
- Current role prompt from `prompts/roles/benchmark_engineer.md`.

Output:
- JSONL benchmark result under `runs/kernelbench/`.
- Command log under `runs/logs/`.

Record:
- Correctness, latency, energy if available, speedup, accepted status, host/GPU, exact commands.
