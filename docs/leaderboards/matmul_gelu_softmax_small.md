# matmul_gelu_softmax_small Leaderboard

Target: `benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`

Reference behavior: `nn.Linear(1024, 1024)` on batch `64`, exact GELU, then row-wise softmax over dimension `1`.

Hardware for the results below: NVIDIA H100 80GB HBM3 on Prime Intellect / Datacrunch.

The benchmark has two tracks:

- Energy track: lower warmed repeated-forward Zeus GPU energy is better. This is the primary H100 track for energy-aware optimization.
- Time track: lower KernelBench CUDA-event latency is better. This tracks steady-state runtime against the same paired PyTorch baseline.

Each candidate row is paired with a same-run PyTorch reference measurement.
Energy results should not be compared across hardware or scorer settings without
re-running the baseline. The old full-subprocess Zeus numbers around `1000 J`
are diagnostic only and are intentionally excluded from the energy ranking.

## Energy Track

| Rank | Submission | Correct | Candidate energy | Paired baseline energy | Delta vs paired baseline | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | `candidate_v7.py` | yes | `559.458 J` | `1286.071 J` | `-726.613 J` | Best controlled scorer energy run so far; mean of both measurement orders over `200,000` forwards |
| 2 | `candidate_v5.py` | yes | `572.503 J` | `1254.859 J` | `-682.356 J` | Controlled scorer energy win; mean of both measurement orders over `200,000` forwards |
| 3 | `candidate_v3.py` | yes | `92.709 J` | `111.595 J` | `-18.886 J` | Corrected kernel-only energy win |
| 4 | PyTorch baseline | yes | `109.136-134.779 J` | n/a | n/a | Same H100 scorer range observed across paired runs |
| 5 | `candidate_v6.py` | yes | `115.665 J` | `109.136 J` | `+6.529 J` | Latency win, corrected energy regression |
| 6 | `candidate_v2.py` | yes | `126.586 J` | `111.403 J` | `+15.183 J` | Latency win, corrected energy regression |

Top rows above use `measurement_scope=official_kernel_only_repeated_forward_pair`,
`warmup_iters=200`, and `measure_iters=200000` with both `candidate,baseline`
and `baseline,candidate` orders. Older rows were measured with shorter windows
and should be refreshed before final ranking.

## Time Track

| Rank | Submission | Correct | Candidate latency | Paired baseline latency | Speedup vs paired baseline | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | `candidate_v7.py` | yes | `0.0187 ms` | `0.0243 ms` | `1.30x` | Best controlled scorer KernelBench latency result so far |
| 2 | `candidate_v5.py` | yes | `0.0191 ms` | `0.0250 ms` | `1.31x` | Controlled scorer time and energy winner |
| 3 | `candidate_v8.py` | yes | `0.0213 ms` | `0.0255 ms` | `1.20x` | Diagnostic/negative-control submission |
| 4 | `candidate_v2.py` | yes | `0.0214 ms` | `0.0247 ms` | `1.15x` | Time win |
| 5 | `candidate_v3.py` | yes | `0.0214 ms` | `0.0252 ms` | `1.18x` | Time and corrected energy win |
| 6 | `candidate_v6.py` | yes | `0.0216 ms` | `0.0250 ms` | `1.16x` | Time win |
| 7 | PyTorch baseline | yes | `0.0241-0.0299 ms` | n/a | `1.00x` | Paired baseline range observed in the runs above |
| 8 | `candidate_v1.py` | yes | `0.2870 ms` | `0.0248 ms` | `0.09x` | Naive scalar matmul, rejected for latency |

`candidate_v4.py` is excluded from the ranked tables because it failed the KernelBench static check for this scorer path: it did not contain a visible CUDA `__global__` kernel definition.

## Reproduction

Run the controlled scorer before adding or updating rows:

```bash
python3 scripts/score_focus3_submission.py \
  --task matmul_gelu_softmax_small \
  --candidate <candidate.py> \
  --entry-point ModelNew \
  --output runs/leaderboard_scores/matmul_gelu_softmax_small.json \
  --num-correct-trials 1 \
  --num-perf-trials 5 \
  --warmup-iters 200 \
  --measure-iters 200000 \
  --timeout 300
```

For comparison, the upstream KernelBench "Kernelsseum" public leaderboard ranks per-task speedup over Torch on NVIDIA L40S. That is useful context, but it is not the same track as this H100 energy leaderboard.

The scorer runs the KernelBench time-track check and the corrected warmed
repeated-forward energy protocol described in `docs/leaderboards/evaluation.md`.
Do not use full-subprocess Zeus energy for the ranked energy table.
