# inferencetimegepa

Repo for using a Devin CLI team to implement KernelBench kernels, with GEPA-style prompt optimization applied to the team's instructions.

This setup does not use RL, GRPO, LoRA, fine-tuning, or model-weight optimization. Devin agents do the kernel implementation work. GEPA is the outer loop that reflects on their traces and benchmark outcomes, then proposes better prompts for the Devin team.

Current recommended setup is a 4-agent Devin team: 1 router plus 3 implementation subagents. For this team structure, the best first optimization loop is a reduced KernelBench subset, not the broader 8-task train split.

## Local Setup

```bash
cp .env.example .env
bash scripts/bootstrap_quickrun.sh
```

If you can authenticate Devin CLI, do that locally before launching team work:

```bash
devin auth
```

`devin` is expected to run from this repo root so it can read `AGENTS.md`, `.devin/tasks`, `prompts/roles`, and the KernelBench harness.

The repo is safe to publish as-is: `.env`, `runs/`, `wandb/`, `tmp/`, and the local `third_party/KernelBench` clone are ignored. Fill your local `.env` with Prime SSH settings and optional W&B credentials after cloning.

## Quickrun

For the current recommended reduced subset and 4-agent team:

```bash
cp .env.example .env
devin auth
bash scripts/bootstrap_quickrun.sh
bash scripts/quickrun_5h_small.sh
```

This launches the 5-hour router + 3 subagent loop on:

- `benchmarks/kernelbench_prime_train_small/tasks.jsonl`
- `benchmarks/kernelbench_prime_val_small/tasks.jsonl`

Override any remote/scorer setting with `.env`:

```bash
PRIME_HOST=ubuntu@your-box
PRIME_SSH_KEY=~/.ssh/your_key
PRIME_REMOTE_ROOT=/home/ubuntu/inferencetimegepa
PRIME_KERNELBENCH_ROOT=/home/ubuntu/KernelBench
```

## Focus3 Run

Current recommended public replication target for this specific experiment:

`https://github.com/sjbaebae/kernelbench-focus3-quickrun`

```bash
cp .env.example .env
devin auth
bash scripts/bootstrap_quickrun.sh
bash scripts/quickrun_5h_focus3.sh
```

This launches the exact 5-hour `focus3` run:

- cheap: `benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`
- mid: `benchmarks/kernelbench_focus3/mlp_small.py`
- expensive: `benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py`
- held-out validation: `benchmarks/kernelbench_focus3/layernorm_small.py`

Task manifests:

- `benchmarks/kernelbench_focus3/train_tasks.jsonl`
- `benchmarks/kernelbench_focus3/val_tasks.jsonl`

The standalone public runner repo above is the easiest way for other people to reproduce this exact `3` task train + `1` task validation setup with baseline energy tracking.

## Remote GPU

Current Prime Intellect SSH target:

```bash
ssh ubuntu@209.20.158.160 -p 22
```

Use this first for connectivity and KernelBench smoke tests. H100 launch comes later after the team/prompt loop is proven.

Current scorer wrapper: `scripts/prime_kernelbench_pair.py` copies a baseline/candidate pair to Prime, runs the official KernelBench `scripts/run_and_check.py` on the H100, and emits JSON for `inferencetimegepa run-kernelbench`.

Verified smoke:

```bash
inferencetimegepa run-kernelbench \
  --tasks tmp/prime_smoke/tasks.jsonl \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point} --num-correct-trials 1 --num-perf-trials 3 --timeout 120' \
  --zeus \
  --timeout-s 240 \
  --out runs/prime_smoke/results.jsonl
```

This reports correctness, latency, and Zeus GPU energy from the Prime H100 scorer. The wrapper performs an unmeasured compile/warmup pass by default, then starts Zeus for the measured pass so CUDA extension compilation energy is not counted. Use a longer `--timeout-s` because the whole remote command still includes both passes.

## Workflow

1. Give Devin team members the task briefs in `.devin/tasks`.
2. Router writes a task-specific implementation plan from the public baseline/interface only.
3. Three implementation subagents produce independent candidate kernels from that plan.
4. KernelBench / Prime scores the candidates and keeps the best attempt for reflection.
5. Either `optimize_anything` or Codex reads traces/results and proposes prompt mutations.
6. Keep prompt candidates in `prompts/candidates.jsonl` and compute the Pareto frontier.

Useful local commands:

```bash
inferencetimegepa pre-gpu-smoke --config configs/agents_2xh200.json --out-dir runs/pre_gpu_smoke
inferencetimegepa run-kernelbench \
  --tasks runs/kernel_tasks.jsonl \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point}' \
  --zeus \
  --out runs/kernelbench/results.jsonl
inferencetimegepa gepa-frontier --candidates prompts/candidates.jsonl --out prompts/frontier.jsonl
```

Generate a reflection prompt from rollout traces:

```bash
inferencetimegepa gepa-reflect-prompt \
  --role gepa_reflector \
  --rollouts runs/pre_gpu_smoke/merged.jsonl \
  --out runs/gepa/reflection_prompt.md
```

## GEPA Optimize Anything

Use this when GEPA should automatically mutate the Devin team prompt pack:

```bash
inferencetimegepa optimize-anything \
  --dataset benchmarks/kernelbench_real/tasks.jsonl \
  --out-dir runs/gepa_optimize_anything \
  --devin-command 'devin --permission-mode dangerous -p -- "$(cat .devin/tasks/kernel_author.md; printf \"\nUse prompts from {prompt_dir}. Implement task {task_id} with baseline {baseline}. Write the final candidate to {eval_dir}/candidate.py.\n\")"' \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point}'
```

The candidate prompt pack never sees hidden tests or scoring internals. The evaluator only returns black-box KernelBench correctness, latency, optional Zeus energy, and traces.

## Codex As Mutator

Use this when Codex should inspect traces and manually mutate prompts:

```bash
inferencetimegepa codex-reflection-packet \
  --dataset benchmarks/kernelbench_real/tasks.jsonl \
  --out-dir runs/codex_reflection \
  --devin-command 'devin --permission-mode dangerous -p -- "$(cat .devin/tasks/kernel_author.md; printf \"\nUse prompts from {prompt_dir}. Implement task {task_id} with baseline {baseline}. Write the final candidate to {eval_dir}/candidate.py.\n\")"' \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point}'
```

Then inspect `runs/codex_reflection/codex_reflection_packet.md` and update `prompts/roles/*.md`.

For an autonomous Codex-CLI hill climb:

```bash
ITERATIONS=3 DATASET=benchmarks/kernelbench_prime_smoke/tasks.jsonl \
  bash scripts/run_prime_codex_hillclimb.sh
```

This repeatedly runs Devin + Prime KernelBench scoring, writes a reflection packet, then calls `codex exec` to mutate only the Devin team prompt artifacts.

Dataset subset details are in `docs/kernelbench_subset.md`.

Time-track candidate submissions for the focused `matmul_gelu_softmax_small`
task are stored under
`submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/`.

For the 4-agent GLM5.2 team, prefer the reduced subset first. One train task now means 1 router run, 3 implementation attempts, and up to 3 Prime scores, so the broad 8-task split is too slow for dense GEPA or RL feedback.

Use the smaller mixed subset for both GEPA and the RL comparison, then run the broader train/final sets only for checkpointed validation or end-of-run evaluation.

Concrete reduced-set files:

- `benchmarks/kernelbench_prime_train_small/tasks.jsonl`
- `benchmarks/kernelbench_prime_val_small/tasks.jsonl`

Tracking:

- Every evaluator attempt appends to `<run_dir>/metrics.jsonl`.
- A live Markdown leaderboard is written to `<run_dir>/leaderboard.md`.
- If `.env` contains `WANDB_API_KEY` or `WANDB_DEV_KEY`, metrics are also logged to W&B.
- Optional install: `python -m pip install -e '.[tracking]'`.
