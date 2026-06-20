# matmul_gelu_softmax_small Leaderboard

Target: `benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`

Reference behavior: `nn.Linear(1024, 1024)` on batch `64`, exact GELU, then row-wise softmax over dimension `1`.

Hardware for the results below: NVIDIA H100 80GB HBM3 on Prime Intellect / Datacrunch.

The benchmark has two tracks:

- Energy track: lower warmed repeated-forward Zeus GPU board energy is better. This is the primary H100 track for energy-aware optimization.
- Time track: lower KernelBench CUDA-event latency is better. This tracks steady-state runtime against the same paired PyTorch baseline.

Each candidate row is paired with a same-run PyTorch reference measurement.
Energy results should not be compared across hardware or scorer settings without
re-running the baseline. The energy table reports total energy for the
repeated-forward measurement window plus per-forward energy in millijoules. A
`1000 J`-scale total is plausible for a `200,000`-forward H100 board-energy
window; it is not the energy for one forward. The old full-subprocess Zeus
numbers are diagnostic only and are intentionally excluded from the energy
ranking.

## Energy Track

| Rank | Submission | Correct | Candidate window energy | Candidate energy/forward | Paired baseline window energy | Paired baseline energy/forward | Delta vs paired baseline | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `candidate_v7.py` | yes | `559.458 J` | `2.797 mJ` | `1286.071 J` | `6.430 mJ` | `-726.613 J` | Best controlled scorer energy run so far; mean of both measurement orders over `200,000` forwards |
| 2 | `candidate_v5.py` | yes | `572.503 J` | `2.863 mJ` | `1254.859 J` | `6.274 mJ` | `-682.356 J` | Controlled scorer energy win; mean of both measurement orders over `200,000` forwards |
| 3 | `candidate_v3.py` | yes | `92.709 J` | n/a | `111.595 J` | n/a | `-18.886 J` | Corrected kernel-only energy win |
| 4 | PyTorch baseline | yes | `109.136-134.779 J` | n/a | n/a | n/a | n/a | Same H100 scorer range observed across paired runs |
| 5 | `candidate_v6.py` | yes | `115.665 J` | n/a | `109.136 J` | n/a | `+6.529 J` | Latency win, corrected energy regression |
| 6 | `candidate_v2.py` | yes | `126.586 J` | n/a | `111.403 J` | n/a | `+15.183 J` | Latency win, corrected energy regression |

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

## External Runtime Baselines

These rows are diagnostic cross-checks against public runtime-oriented
implementations. They are not ranked unless they satisfy the same submission and
static-check rules as local candidates.

| Source | Public task | Public runtime result | Correct under this scorer | H100 time result | H100 energy result | Notes |
|---|---|---:|---:|---:|---:|---|
| [Kernelsseum `gpt-o1`](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/5c36dadcac0846f0a4bc95a39760ad9d.py) | `Level 2: 99_Matmul_GELU_Softmax` | `0.7604x` vs Torch on public L40S leaderboard | yes | `0.0259 ms` candidate vs `0.0242 ms` paired baseline | `1358.847 J` candidate vs `1257.811 J` paired baseline over `200,000` forwards; `6.794 mJ/forward` candidate vs `6.289 mJ/forward` baseline | Uses PyTorch `nn.Linear` and `F.softmax`; custom part is approximate tanh-GELU only, so this is provenance/diagnostic rather than a clean custom-kernel submission |

## Strong Baseline Ladder

These baselines are intended to separate ordinary fusion wins from more
meaningful kernel improvements. Rows use the same focused forward-only energy
scorer as the leaderboard rows. Diagnostic rows that fail KernelBench static
checks are not eligible submissions, but they are still useful context.

| Baseline | Eligible submission | Correct | Focused energy | Paired baseline energy | Focused latency | Time-track latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| PyTorch eager equivalent | no | yes | `6.265 mJ/fwd` | `6.317 mJ/fwd` | `0.0278 ms` | n/a | Diagnostic only; static check rejects PyTorch compute ops |
| `torch.compile` equivalent | no | yes | `9.651 mJ/fwd` | `6.266 mJ/fwd` | `0.0570 ms` | n/a | Diagnostic only; slower and higher energy for this shape |
| cuBLASLt GEMM + separate custom GELU + separate custom softmax | yes | yes | `3.235 mJ/fwd` | `6.437 mJ/fwd` | `0.0184 ms` | `0.0212 ms` | Strong comparator for "custom kernels but no GELU+softmax fusion" |
| cuBLASLt bias epilogue + fused custom GELU/softmax | yes | yes | `2.863 mJ/fwd` | `6.209 mJ/fwd` | `0.0159 ms` | `0.0193 ms` | Strong comparator for using cuBLASLt bias epilogue before the fused post-op |
| `candidate_v5.py` | yes | yes | `2.371 mJ/fwd` | `6.703 mJ/fwd` | `0.0157 ms` | `0.0191 ms` | Current best focused-energy row |
| `candidate_v7.py` | yes | yes | `2.419 mJ/fwd` | `6.729 mJ/fwd` | `0.0160 ms` | `0.0187 ms` | Current best time-track row |

CUTLASS/CuTe was not measured in this pass because the active H100 image did
not have a CUTLASS checkout or Python package available. A future CUTLASS/CuTe
row should use the same scorer before it is compared against the rows above.

The current public Kernelsseum data has one entry for the exact corresponding
task. Adjacent GEMM, matmul, GELU, and softmax leaderboard entries can still be
used as implementation references, but they must be ported to the exact
`nn.Linear(1024, 1024) -> exact GELU -> softmax(dim=1)` behavior before they are
eligible for this leaderboard.

The scorer runs the KernelBench time-track check and the corrected warmed
repeated-forward energy protocol described in `docs/leaderboards/evaluation.md`.
Do not use full-subprocess Zeus energy for the ranked energy table.
