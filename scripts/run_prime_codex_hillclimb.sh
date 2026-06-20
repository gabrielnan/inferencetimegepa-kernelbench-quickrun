#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-benchmarks/kernelbench_prime_smoke/tasks.jsonl}"
VAL_DATASET="${VAL_DATASET:-benchmarks/kernelbench_prime_val/tasks.jsonl}"
OUT_ROOT="${OUT_ROOT:-runs/prime_codex_hillclimb_$(date +%Y%m%d_%H%M%S)}"
MAX_HOURS="${MAX_HOURS:-3}"
MAX_SECONDS="${MAX_SECONDS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-999999}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.1-codex-max}"
DEVIN_MODEL_NAME="${DEVIN_MODEL_NAME:-}"
DEVIN_MODEL_ARG=()
if [[ -n "$DEVIN_MODEL_NAME" ]]; then
  DEVIN_MODEL_ARG=(--model "$DEVIN_MODEL_NAME")
fi
DEVIN_SUBAGENTS="${DEVIN_SUBAGENTS:-3}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VAL_EVERY="${VAL_EVERY:-5}"

mkdir -p "$OUT_ROOT"
START_TS="$(date +%s)"
if [[ -z "$MAX_SECONDS" ]]; then
  MAX_SECONDS="$(python - <<PY
print(int(float("$MAX_HOURS") * 3600))
PY
)"
fi

iter=1
while (( iter <= MAX_ITERATIONS )); do
  NOW_TS="$(date +%s)"
  ELAPSED=$((NOW_TS - START_TS))
  if (( ELAPSED >= MAX_SECONDS )); then
    echo "Time budget reached after ${ELAPSED}s. Stopping before iteration ${iter}."
    break
  fi
  ITER_DIR="$OUT_ROOT/iter_${iter}"
  mkdir -p "$ITER_DIR"

  echo "== Iteration ${iter}: Devin + Prime KernelBench (elapsed ${ELAPSED}s / budget ${MAX_SECONDS}s) =="
  inferencetimegepa codex-reflection-packet \
    --dataset "$DATASET" \
    --out-dir "$ITER_DIR/eval" \
    --kernelbench-timeout-s 1200 \
    --team-size "$DEVIN_SUBAGENTS" \
    --router-command "devin --permission-mode dangerous ${DEVIN_MODEL_ARG[*]} -p -- \"\$(cat .devin/tasks/kernel_author.md; printf \"\\nYou are the router for a 4-agent KernelBench team. Task: {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Read prompts from {prompt_dir}. Write a concise plan to {router_plan_path} for three subagents: correctness_first_simple_cuda, memory_coalescing_and_layout, and tiling_and_shared_memory. Include public tensor shapes/interfaces from the baseline only. Do not inspect scorer internals, hidden tests, or write candidate.py.\\n\")\"" \
    --devin-command "devin --permission-mode dangerous ${DEVIN_MODEL_ARG[*]} -p -- \"\$(cat .devin/tasks/kernel_author.md; printf \"\\nYou are implementation subagent {attempt_index} of {team_size}, role: {team_role}. Use prompts from {prompt_dir}. Read the router plan at {router_plan_path}. Implement task {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Write only the final candidate Python file to {eval_dir}/candidate.py. The candidate must define class ModelNew(nn.Module) compatible with the baseline Model. Do not inspect or rely on scorer internals.\\n\")\"" \
    --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point} --num-correct-trials 3 --num-perf-trials 20 --timeout 420'

  PACKET="$ITER_DIR/eval/codex_reflection_packet.md"
  CODEX_LOG="$ITER_DIR/codex_last_message.md"

  echo "== Iteration ${iter}: Codex prompt mutation =="
  codex exec \
    --cd "$(pwd)" \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$CODEX_MODEL" \
    --output-last-message "$CODEX_LOG" \
    "You are the GEPA prompt mutator for this repo.

Read ${PACKET}.

Task:
- Mutate only Devin team prompt/instruction artifacts: prompts/roles/*.md and, if useful, AGENTS.md or .devin/tasks/*.md.
- Do not edit KernelBench baselines, scorer wrappers, candidate outputs, benchmark results, or source harness code.
- Do not expose or infer hidden tests/scorer internals.
- Use GEPA-style reflection: preserve prompt behavior that worked, target concrete recurring failures, and improve future Devin kernel implementations.
- Append one JSONL row to prompts/candidates.jsonl summarizing the prompt candidate, parent, targeted failure pattern, and metrics if available.
- Keep changes small enough that the next iteration can attribute effects.

After editing, run python -m pytest and report changed files."

  if [[ "$RUN_VALIDATION" == "1" ]] && (( iter % VAL_EVERY == 0 )); then
    VAL_DIR="$ITER_DIR/val"
    echo "== Iteration ${iter}: held-out validation =="
    inferencetimegepa codex-reflection-packet \
      --dataset "$VAL_DATASET" \
      --out-dir "$VAL_DIR" \
      --kernelbench-timeout-s 1200 \
      --team-size "$DEVIN_SUBAGENTS" \
      --router-command "devin --permission-mode dangerous ${DEVIN_MODEL_ARG[*]} -p -- \"\$(cat .devin/tasks/kernel_author.md; printf \"\\nYou are the router for a 4-agent held-out validation KernelBench team. Task: {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Read prompts from {prompt_dir}. Write a concise plan to {router_plan_path} for three subagents: correctness_first_simple_cuda, memory_coalescing_and_layout, and tiling_and_shared_memory. Include public tensor shapes/interfaces from the baseline only. Do not inspect scorer internals, hidden tests, or write candidate.py.\\n\")\"" \
      --devin-command "devin --permission-mode dangerous ${DEVIN_MODEL_ARG[*]} -p -- \"\$(cat .devin/tasks/kernel_author.md; printf \"\\nYou are held-out validation implementation subagent {attempt_index} of {team_size}, role: {team_role}. Use prompts from {prompt_dir}. Read the router plan at {router_plan_path}. Implement task {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Write only the final candidate Python file to {eval_dir}/candidate.py. The candidate must define class ModelNew(nn.Module) compatible with the baseline Model. Do not inspect or rely on scorer internals.\\n\")\"" \
      --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point} --num-correct-trials 3 --num-perf-trials 20 --timeout 420'
    python - <<PY
import json
from pathlib import Path
records=[]
for p in Path("$VAL_DIR").glob("evals/*/*/kernelbench_results.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
summary={
    "iteration": $iter,
    "tasks": len(records),
    "correct": sum(1 for r in records if r.get("correct")),
    "accepted": sum(1 for r in records if r.get("accepted")),
    "mean_score": sum(float(r.get("score", 0.0)) for r in records) / len(records) if records else 0.0,
    "mean_reward": sum(float(r.get("reward", 0.0)) for r in records) / len(records) if records else 0.0,
}
path=Path("$OUT_ROOT")/"validation_history.jsonl"
with path.open("a") as f:
    f.write(json.dumps(summary, sort_keys=True)+"\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  fi

  echo "== Iteration ${iter}: done =="
  iter=$((iter + 1))
done

echo "Hill-climb complete. Output root: $OUT_ROOT"
