# Devin Team Instructions

This repo uses Devin agents to implement and review KernelBench kernels. GEPA is used only to optimize these team prompts and task instructions.

Do not implement RL, GRPO, LoRA, fine-tuning, or model-weight optimization in this repo.

## Team Roles

- `kernel_author`: implements candidate KernelBench kernels.
- `correctness_reviewer`: reviews implementation correctness before benchmarking.
- `benchmark_engineer`: runs KernelBench checks and records reproducible results.
- `gepa_reflector`: reviews traces/results and proposes improved prompts for the team.

## Operating Contract

- Keep kernel changes task-scoped.
- Prefer correctness before latency.
- Report exact commands and result files.
- Store benchmark outputs under `runs/`.
- Store prompt candidates in `prompts/candidates.jsonl`.
- Keep accepted prompt frontier output in `prompts/frontier.jsonl`.

## GEPA Loop

1. Sample team behavior on KernelBench tasks.
2. Collect implementation traces, review notes, benchmark JSONL, and failures.
3. Reflect in natural language on what instruction changes would have helped.
4. Propose new role prompts only.
5. Evaluate the new prompts on held-out or repeated tasks.
6. Retain non-dominated prompts on correctness, accepted rate, speedup, and clarity.
