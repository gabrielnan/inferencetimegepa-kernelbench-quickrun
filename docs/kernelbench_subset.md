# KernelBench Subset

We are not hill-climbing on the full KernelBench suite. The active main run uses the `focus3` subset: one cheap task, one mid-cost task, one expensive task, and one held-out validation task.

## Focus3 Main Run

Standalone replication repo:

- `https://github.com/sjbaebae/kernelbench-focus3-quickrun`

Active train manifest:

- `benchmarks/kernelbench_focus3/train_tasks.jsonl`

Active validation manifest:

- `benchmarks/kernelbench_focus3/val_tasks.jsonl`

Task shape:

- cheap: fused `matmul + gelu + softmax`
- mid: `MLP`
- expensive: reduced `MinGPT` causal attention
- held-out validation: reduced `LayerNorm`

Concrete benchmark files:

- `benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`
- `benchmarks/kernelbench_focus3/mlp_small.py`
- `benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py`
- `benchmarks/kernelbench_focus3/layernorm_small.py`

This is the main run because it preserves kernel diversity while keeping router + 3 subagent iteration cost manageable.

## Alternate Small Split

If we need a broader but still reduced training mix later, the alternate small split is:

- train: `benchmarks/kernelbench_prime_train_small/tasks.jsonl`
- val: `benchmarks/kernelbench_prime_val_small/tasks.jsonl`

That split is useful for expansion after the `focus3` run starts producing stable correct kernels.

## Broader Splits

Broader train:

- `benchmarks/kernelbench_prime_train/tasks.jsonl`

Broader validation:

- `benchmarks/kernelbench_prime_val/tasks.jsonl`

Final pass:

- `benchmarks/kernelbench_prime_final/tasks.jsonl`

Those are for later checkpointed evaluation or final assessment, not for the current main optimization loop.

## Metrics

Each task tracks:

- correctness
- accepted status
- baseline latency
- candidate latency
- baseline Zeus energy
- candidate Zeus energy
- reward
- score

First-run CUDA extension compilation is excluded from Zeus measurement.
