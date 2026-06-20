# Focus3 Evaluation Contract

This document defines the leaderboard and submission metrics for
`benchmarks/kernelbench_focus3`.

Use `scripts/score_focus3_submission.py` for leaderboard scoring. It emits one
normalized JSON object with separate `time` and `energy` track payloads.

```bash
python3 scripts/score_focus3_submission.py \
  --task matmul_gelu_softmax_small \
  --candidate submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/candidate_v5_cublaslt_time_best.py \
  --entry-point ModelNew \
  --host "$PRIME_HOST" \
  --key "$PRIME_SSH_KEY" \
  --remote-root "$PRIME_REMOTE_ROOT" \
  --kernelbench-root "$PRIME_KERNELBENCH_ROOT" \
  --output runs/leaderboard_scores/matmul_v5.json
```

The repository also has a generic evaluation entry point,
`inferencetimegepa run-kernelbench`, which runs a configurable compare command
and parses the final JSON line. That remains useful for GEPA automation, but
leaderboard rows should use the controlled scorer above so energy rows are not
mixed across different measurement scopes.

## Tracks

### Time Track

The time track ranks KernelBench CUDA-event latency in milliseconds per forward.
Use the upstream KernelBench correctness/performance path through
`scripts/run_and_check.py`. The controlled scorer runs this phase through the
Prime wrapper:

```bash
python3 scripts/prime_kernelbench_pair.py \
  --baseline <baseline.py> \
  --candidate <candidate.py> \
  --entry-point ModelNew \
  --num-correct-trials 1 \
  --num-perf-trials 5 \
  --timeout 300
```

Rank by `candidate_latency_ms`, with lower values better. Each candidate row
must include the same-run `baseline_latency_ms`.

### Energy Track

The energy track ranks a warmed repeated-forward Zeus GPU energy window. It is
not the full `run_and_check.py` subprocess-window energy.

For each candidate, the controlled scorer:

1. Runs KernelBench correctness/static checks first.
2. Instantiates the PyTorch baseline and candidate in one Python process.
3. Copies the PyTorch baseline weights into `ModelNew`.
4. Warms up both models.
5. Measures the candidate and paired baseline with separate Zeus windows around
   repeated `model(*inputs)` forwards.
6. Records both CUDA-event latency per forward and Zeus energy for the full
   repeated-forward window.

Rank by `candidate_energy_j`, with lower values better. Candidate rows must
include:

- `measurement_scope=official_kernel_only_repeated_forward_pair`
- `warmup_iters`
- `measure_iters`
- `candidate_energy_j`
- `baseline_energy_j`
- `candidate_latency_ms`
- `baseline_latency_ms`
- correctness status and tolerance
- GPU model and host/team

For `matmul_gelu_softmax_small`, the current reference run uses `200` warmup
forwards and `200,000` measured forwards on an NVIDIA H100 80GB HBM3. The
controlled scorer measures both `candidate,baseline` and `baseline,candidate`
orders and reports aggregate values, which reduces order and power-state bias.

## Diagnostic Results

The older `scripts/prime_kernelbench_pair.py` Zeus energy value wraps a full
KernelBench subprocess. That includes import, extension build/load, correctness
trials, cache clearing, and scorer overhead. It may be useful for debugging, but
it is not an energy-track leaderboard metric.

Diagnostic rows must not be ranked against kernel-only energy rows.

## Submission Rules

Submissions must define `ModelNew` with the same public constructor and forward
behavior as the target baseline. They may not:

- import the baseline or scorer
- inspect hidden tests or scorer internals
- cache outputs for the fixed public input
- change inputs, weights, tolerances, or benchmark settings
- rely on one-time compile/setup work inside the measured forward window

Rows that fail correctness, static checks, or traceability requirements should
be listed only in notes, not in the ranked tables.
