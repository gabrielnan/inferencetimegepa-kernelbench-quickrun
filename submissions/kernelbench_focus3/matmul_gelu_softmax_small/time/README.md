# matmul_gelu_softmax_small Time-Track Submissions

These submissions target:

`benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`

Reference behavior: `nn.Linear(1024, 1024)` on batch `64`, exact GELU, then row-wise softmax over dimension `1`.

All files in this directory define `ModelNew` and were verified as correct in
paired H100 KernelBench runs. They are included because they beat the paired
PyTorch baseline on KernelBench CUDA-event latency.

Energy numbers below use the corrected warmed repeated-forward protocol. Older
full-subprocess Zeus numbers around `1000 J` are diagnostic only and are not
used for the energy status.

## Results

| Submission | Correct | Time-track candidate latency | Time-track paired baseline latency | Speedup | Corrected candidate energy | Corrected paired baseline energy | Track status |
|---|---:|---:|---:|---:|---:|---:|---|
| `candidate_v7_tf32_cublaslt_probe.py` | yes | `0.0187 ms` | `0.0243 ms` | `1.30x` | `559.458 J` | `1286.071 J` | time winner, energy winner |
| `candidate_v5_cublaslt_time_best.py` | yes | `0.0191 ms` | `0.0250 ms` | `1.31x` | `572.503 J` | `1254.859 J` | time winner, energy winner |
| `candidate_v8_cublas_negative_control.py` | yes | `0.0213 ms` | `0.0255 ms` | `1.20x` | not rerun | not rerun | time winner, energy pending |
| `candidate_v2_aten_linear_epilogue.py` | yes | `0.0214 ms` | `0.0247 ms` | `1.15x` | `126.586 J` | `111.403 J` | time winner, energy loss |
| `candidate_v3_cublas_epilogue.py` | yes | `0.0214 ms` | `0.0252 ms` | `1.18x` | `92.709 J` | `111.595 J` | time winner, energy winner |
| `candidate_v6_vec4_epilogue.py` | yes | `0.0216 ms` | `0.0250 ms` | `1.16x` | `115.665 J` | `109.136 J` | time winner, energy loss |

## Reproduction

Run the time-track check with:

```bash
python3 scripts/prime_kernelbench_pair.py \
  --baseline benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py \
  --candidate submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/<candidate>.py \
  --entry-point ModelNew \
  --num-correct-trials 1 \
  --num-perf-trials 5 \
  --timeout 300
```

Candidate notes:

- `candidate_v7_tf32_cublaslt_probe.py` is the fastest controlled scorer time-track candidate in this batch and has the best long-window energy result so far.
- `candidate_v5_cublaslt_time_best.py` is the second-best long-window energy result so far.
- `candidate_v8_cublas_negative_control.py` is kept as a diagnostic negative-control style submission.
- Corrected energy rows for `candidate_v5_cublaslt_time_best.py` and `candidate_v7_tf32_cublaslt_probe.py` use `measurement_scope=official_kernel_only_repeated_forward_pair`, `warmup_iters=200`, `measure_iters=200000`, and the mean across `candidate,baseline` plus `baseline,candidate` measurement orders.
