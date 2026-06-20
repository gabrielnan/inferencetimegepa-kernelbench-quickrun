#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e '.[dev,openai,gepa]'
python -m pytest
inferencetimegepa pre-gpu-smoke --config configs/agents_2xh200.json --out-dir runs/pre_gpu_smoke
inferencetimegepa gepa-frontier --candidates prompts/candidates.jsonl --out prompts/frontier.jsonl
