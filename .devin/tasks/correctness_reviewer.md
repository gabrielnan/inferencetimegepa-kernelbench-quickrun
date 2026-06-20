# Task: Correctness Reviewer

Review candidate kernels before benchmarking.

Inputs:
- Candidate kernel from `runs/candidates/`.
- Baseline task spec.
- Current role prompt from `prompts/roles/correctness_reviewer.md`.

Output:
- Review note under `runs/reviews/`.
- Minimal patch suggestion if the candidate is wrong.

Focus:
- Shapes, dtype, device, strides, tolerance, edge cases, launch dimensions, and synchronization assumptions.
