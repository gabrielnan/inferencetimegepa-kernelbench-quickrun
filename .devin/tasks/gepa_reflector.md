# Task: GEPA Reflector

Reflect on Devin team traces and benchmark outcomes to improve role prompts.

Inputs:
- `runs/` traces, reviews, notes, and benchmark JSONL.
- `prompts/candidates.jsonl`.
- Current role prompts under `prompts/roles/`.

Output:
- New prompt candidate rows for `prompts/candidates.jsonl`.
- Reflection note explaining which failures the prompt change targets.

Constraints:
- Optimize prompts only.
- Do not implement kernels directly.
- Do not propose RL, GRPO, LoRA, fine-tuning, or model-weight updates.
