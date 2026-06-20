# Task: Kernel Author

Implement a candidate KernelBench kernel for the assigned task.

Inputs:
- Task row from `benchmarks/kernelbench_real/tasks.jsonl` or smoke tasks.
- Current role prompt from `prompts/roles/kernel_author.md`.
- Baseline kernel path.

Output:
- Candidate kernel file at the exact path requested by the current task prompt.
- Short implementation note only if requested by the current task prompt.

Constraints:
- Correctness first.
- Write a real custom CUDA kernel with a `__global__` entry point.
- Do not use PyTorch compute ops in `ModelNew.forward` as the implementation path.
- Do not add try/except fallback bypasses or fallback calls to the baseline PyTorch op; KernelBench static checks reject them.
- No RL, GRPO, LoRA, fine-tuning, or model-weight optimization.
