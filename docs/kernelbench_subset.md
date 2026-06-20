# KernelBench Subset

We are not hill-climbing on the full KernelBench suite. The autonomous loop uses a deliberately small mixed subset so each prompt mutation gets fast feedback while still covering the major kernel patterns Devin needs to learn.

For the current 4-agent setup, that means a smaller subset than the original broad train/val split. One task now expands into 1 router run plus 3 implementation attempts, so iteration time grows quickly and both GEPA and RL get better signal from a tighter task set.

## Recommended Small Subset

Recommended first optimization split for both GEPA and the RL comparison:

- Train: 4 tasks
- Validation: 2 tasks
- Final pass: broader 8-16 task evaluation only after prompt quality improves

Concrete files:

- `benchmarks/kernelbench_prime_train_small/tasks.jsonl`
- `benchmarks/kernelbench_prime_val_small/tasks.jsonl`

Recommended train mix:

- `kb_l1_19_relu`
- `kb_l1_40_layernorm`
- `kb_l2_1_conv2d_relu_biasadd`
- `kb_l3_1_mlp`

Recommended held-out validation mix:

- `kb_l1_48_mean_reduction`
- `kb_l2_40_matmul_scaling_residualadd`

This gives one simple elementwise kernel, one normalization/reduction-heavy pattern, one Level 2 fused operator, and one small model-level task. That is enough diversity to pressure the prompts without stretching one GEPA iteration into a multi-hour pass.

If matmul-specific prompt behavior becomes the main failure mode, swap one of the simpler train tasks for `kb_l1_2_matmul` and keep the rest of the small split fixed.

## Broader Train Split

`benchmarks/kernelbench_prime_train/tasks.jsonl` has 8 tasks:

- Level 1: matmul, ReLU, softmax, layer norm, sum reduction
- Level 2: Conv2D + ReLU + bias fusion, GEMM + multiply + LeakyReLU fusion
- Level 3: small MLP

This gives broad coverage of elementwise, matmul, reduction, normalization, convolution/fusion, and small-model structure, but it is better used after the small subset starts producing stable correct kernels.

## Broader Validation Split

`benchmarks/kernelbench_prime_val/tasks.jsonl` has 4 held-out tasks:

- batched matmul
- mean reduction
- matmul + residual fusion
- LeNet5

Use these to detect whether prompt changes generalize beyond the training tasks.

## Final Pass

`benchmarks/kernelbench_prime_final/tasks.jsonl` has 16 tasks: train + validation plus harder held-outs such as max reduction, MinGPT GELU, and additional Level 2 fusion tasks.

This should be run after the time-budgeted hill climb to estimate final prompt quality, not as the main optimization set.

## Metrics

Each task tracks correctness, accepted status, latency, Zeus GPU energy for baseline and candidate, reward, and score. First-run CUDA extension compilation is excluded from Zeus measurement.
