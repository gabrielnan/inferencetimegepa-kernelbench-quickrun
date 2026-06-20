#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-benchmarks/kernelbench_prime_l1_small/tasks.jsonl}"
OUT_DIR="${OUT_DIR:-runs/prime_gepa_autonomous_$(date +%Y%m%d_%H%M%S)}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-24}"
MAX_CANDIDATE_PROPOSALS="${MAX_CANDIDATE_PROPOSALS:-8}"
REFLECTION_LM="${REFLECTION_LM:-openai/gpt-5.1}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for fully autonomous GEPA reflection_lm=${REFLECTION_LM}." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

inferencetimegepa optimize-anything \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --max-metric-calls "$MAX_METRIC_CALLS" \
  --max-candidate-proposals "$MAX_CANDIDATE_PROPOSALS" \
  --reflection-lm "$REFLECTION_LM" \
  --kernelbench-timeout-s 1200 \
  --devin-command 'devin --permission-mode dangerous -p -- "$(cat .devin/tasks/kernel_author.md; printf "\nUse prompts from {prompt_dir}. Implement task {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Write only the final candidate Python file to {eval_dir}/candidate.py. The candidate must define class ModelNew(nn.Module) compatible with the baseline Model. Prefer simple, correct CUDA/Triton/PyTorch-extension code and include no evaluator-specific hacks.\n")"' \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point} --num-correct-trials 5 --num-perf-trials 50 --timeout 420'
