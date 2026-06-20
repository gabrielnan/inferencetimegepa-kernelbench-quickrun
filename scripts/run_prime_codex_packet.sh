#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-benchmarks/kernelbench_prime_smoke/tasks.jsonl}"
OUT_DIR="${OUT_DIR:-runs/prime_codex_packet_$(date +%Y%m%d_%H%M%S)}"

inferencetimegepa codex-reflection-packet \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --kernelbench-timeout-s 900 \
  --devin-command 'devin --permission-mode dangerous -p -- "$(cat .devin/tasks/kernel_author.md; printf "\nUse prompts from {prompt_dir}. Implement task {task_id}. Baseline path: {baseline}. Entry point: {entry_point}. Write only the final candidate Python file to {eval_dir}/candidate.py. The candidate must define class ModelNew(nn.Module) compatible with the baseline Model.\n")"' \
  --compare-cmd 'python scripts/prime_kernelbench_pair.py --baseline {baseline} --candidate {candidate} --entry-point {entry_point} --num-correct-trials 3 --num-perf-trials 20 --timeout 300'
