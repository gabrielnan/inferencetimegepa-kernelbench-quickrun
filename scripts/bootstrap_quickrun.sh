#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KERNELBENCH_REPO="${KERNELBENCH_REPO:-https://github.com/ScalingIntelligence/KernelBench.git}"
KERNELBENCH_DIR="${KERNELBENCH_DIR:-third_party/KernelBench}"

if [[ ! -d "$KERNELBENCH_DIR/.git" ]]; then
  mkdir -p "$(dirname "$KERNELBENCH_DIR")"
  git clone "$KERNELBENCH_REPO" "$KERNELBENCH_DIR"
fi

python -m pip install -e '.[dev,tracking]'
python -m pytest

printf 'Bootstrap complete.\n'
printf 'Fill .env from .env.example, authenticate Devin with `devin auth`, then run scripts/quickrun_5h_small.sh\n'
