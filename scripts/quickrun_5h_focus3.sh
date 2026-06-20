#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${PRIME_HOST:?Set PRIME_HOST in .env or environment}"
: "${PRIME_SSH_KEY:?Set PRIME_SSH_KEY in .env or environment}"

MAX_HOURS="${MAX_HOURS:-5}"
VAL_EVERY="${VAL_EVERY:-5}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
DEVIN_MODEL_NAME="${DEVIN_MODEL_NAME:-glm-5.2}"
DEVIN_SUBAGENTS="${DEVIN_SUBAGENTS:-3}"

exec env \
  MAX_HOURS="$MAX_HOURS" \
  VAL_EVERY="$VAL_EVERY" \
  DATASET="benchmarks/kernelbench_focus3/train_tasks.jsonl" \
  VAL_DATASET="benchmarks/kernelbench_focus3/val_tasks.jsonl" \
  RUN_VALIDATION="$RUN_VALIDATION" \
  DEVIN_MODEL_NAME="$DEVIN_MODEL_NAME" \
  DEVIN_SUBAGENTS="$DEVIN_SUBAGENTS" \
  bash scripts/run_prime_codex_hillclimb.sh
