# matmul_gelu_softmax_small Results Brief

This document is a detailed source brief for making slides about our
`matmul_gelu_softmax_small` KernelBench Focus3 work. It is written for a second
agent or teammate who needs the story, methodology, caveats, and measured
numbers without reconstructing the entire thread.

## Executive Summary

We optimized the Focus3 task:

```text
Linear(1024, 1024), batch=64
  -> exact GELU
  -> row-wise softmax over dim=1
```

The PyTorch reference is:

```python
x = self.linear(x)
x = torch.nn.functional.gelu(x)
x = torch.nn.functional.softmax(x, dim=1)
```

The main result is that our best candidates reduce focused forward-only H100 GPU
board energy by roughly 55-57% versus the paired PyTorch baseline, while also
improving the KernelBench CUDA-event time track.

The best focused-energy result is currently `candidate_v7.py`:

- Focused energy: `2.797 mJ/forward`
- Paired PyTorch baseline energy: `6.430 mJ/forward`
- Energy reduction: about `56.5%`
- Focused latency: `0.01567 ms/forward`
- Paired PyTorch baseline focused latency: `0.02855 ms/forward`
- KernelBench time-track latency: `0.0187 ms` vs paired baseline `0.0243 ms`

The best result is not just beating a weak PyTorch eager baseline. We added a
stronger baseline ladder:

- PyTorch eager equivalent: `6.265 mJ/forward`
- `torch.compile` equivalent: `9.651 mJ/forward`
- public Kernelsseum exact-task solution: `6.794 mJ/forward`
- cuBLASLt GEMM + separate custom GELU + separate custom softmax:
  `3.235 mJ/forward`
- cuBLASLt bias epilogue + fused custom GELU/softmax:
  `2.863 mJ/forward`
- our `candidate_v7.py`: `2.797 mJ/forward`

The important scientific interpretation is:

1. Plain PyTorch and public runtime-leader code are not competitive on this H100
   focused-energy metric.
2. A straightforward custom-kernel baseline with separate post-GEMM GELU and
   softmax is a major improvement over PyTorch, reaching `3.235 mJ/forward`.
3. Folding bias into the cuBLASLt GEMM epilogue and then fusing GELU+softmax is
   stronger, reaching `2.863 mJ/forward`.
4. Our best candidates are still slightly better than that strong epilogue
   baseline, but the remaining gap is much smaller than the PyTorch comparison.
5. Therefore, the credible claim is not "we invented fusion"; it is "we found a
   specialized, energy-efficient composition for this fixed shape that beats
   both PyTorch and stronger cuBLASLt/custom-kernel baselines."

## Repository And PR Context

Primary repository:

- `sjbaebae/inferencetimegepa-kernelbench-quickrun`
- Local PR worktree used for leaderboard docs: `/tmp/kb-pr1`
- Local submission PR worktree used for candidate code: `/tmp/kb-pr2`

Relevant PRs:

- PR #1: Focus3 leaderboards and scorer contract
  - `https://github.com/sjbaebae/inferencetimegepa-kernelbench-quickrun/pull/1`
  - Branch: `codex/matmul-gelu-softmax-leaderboard`
- PR #2: time-track submissions
  - `https://github.com/sjbaebae/inferencetimegepa-kernelbench-quickrun/pull/2`
  - Branch: `codex/matmul-gelu-softmax-time-submissions`

Important docs:

- `docs/leaderboards/README.md`
- `docs/leaderboards/evaluation.md`
- `docs/leaderboards/matmul_gelu_softmax_small.md`
- This brief:
  `docs/leaderboards/matmul_gelu_softmax_small_slide_brief.md`

Important scorer:

- `scripts/score_focus3_submission.py`

Important candidate files:

- `/tmp/kb-pr2/submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/candidate_v5_cublaslt_time_best.py`
- `/tmp/kb-pr2/submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/candidate_v7_tf32_cublaslt_probe.py`

Important local diagnostic artifacts:

- `/tmp/kb-pr1/runs/leaderboard_scores/matmul_v5_stress.jsonl`
- `/tmp/kb-pr1/runs/leaderboard_scores/matmul_v7_stress.jsonl`
- `/tmp/kb-pr1/runs/leaderboard_scores/alt_cublaslt_separate_gelu_softmax_focused.json`
- `/tmp/kb-pr1/runs/leaderboard_scores/alt_cublaslt_bias_epilogue_fused_gelu_softmax_focused.json`
- `/tmp/kb-pr1/runs/leaderboard_scores/alt_kernelsseum_l2p99_gpt_o1_focused.json`
- `/tmp/kb-pr1/runs/leaderboard_scores/alt_pytorch_eager_equivalent_focused.json`
- `/tmp/kb-pr1/runs/leaderboard_scores/alt_torch_compile_equivalent_focused.json`

Note: `runs/` is ignored by git. The measured numbers should be copied into
tracked docs when needed.

## Benchmark Definition

Task file:

- `benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py`

Reference behavior:

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

batch_size = 64
in_features = 1024
out_features = 1024
```

Shape:

- Input: `[64, 1024]`, float32
- Weight: `[1024, 1024]`, float32
- Bias: `[1024]`, float32
- Output: `[64, 1024]`, float32
- Softmax axis: row-wise over the `1024` output columns

Mathematical operation:

```text
logits = x @ weight.T + bias
gelu = 0.5 * logits * (1 + erf(logits / sqrt(2)))
out[row, col] = exp(gelu[row, col] - max(gelu[row, :])) / sum_j exp(gelu[row, j] - max(gelu[row, :]))
```

Correctness target:

- Match PyTorch reference behavior.
- The focused energy scorer records `max_abs_error`, `max_rel_error`, and
  `torch.allclose(..., rtol=1e-4, atol=1e-4)`.
- Stress checks for `v5` and `v7` passed 25/25 correctness trials across
  multiple input distributions.

## Measurement Methodology

There are two metrics. They are intentionally different because they answer
different questions.

### Time Track

The time track uses the upstream KernelBench `scripts/run_and_check.py` path via
the Prime wrapper:

```bash
python3 scripts/prime_kernelbench_pair.py \
  --baseline benchmarks/kernelbench_focus3/matmul_gelu_softmax_small.py \
  --candidate <candidate.py> \
  --entry-point ModelNew \
  --num-correct-trials 1 \
  --num-perf-trials 5 \
  --timeout 300
```

This returns:

- KernelBench correctness
- KernelBench static-check status
- CUDA-event runtime for the custom kernel candidate
- CUDA-event runtime for the PyTorch reference
- PyTorch `torch.compile` reference runtime reported by KernelBench

The time track is good for compatibility with KernelBench-style runtime
reporting. It is not the primary energy metric.

### Focused Energy Track

The energy track measures the focused forward call, not the whole
`run_and_check.py` subprocess.

The scorer:

1. Runs KernelBench correctness/static checks first.
2. Instantiates the PyTorch baseline and candidate in one Python process.
3. Copies the PyTorch baseline weights into `ModelNew`.
4. Builds inputs from the task's public `get_inputs()`.
5. Warms up both models.
6. Measures repeated `model(*inputs)` calls under Zeus GPU board-energy windows.
7. Measures both orders:
   - `candidate,baseline`
   - `baseline,candidate`
8. Averages candidate windows and baseline windows separately.

Current primary settings:

- `measurement_scope=official_kernel_only_repeated_forward_pair`
- `warmup_iters=200`
- `measure_iters=200000`
- `orders=candidate,baseline;baseline,candidate`
- GPU: NVIDIA H100 80GB HBM3

Why use repeated forwards?

- One forward is roughly `0.015-0.03 ms`, which is too short for stable
  energy-window measurement.
- Repeating the exact same forward produces a multi-second measurement window.
- The result is reported both as total window joules and per-forward
  millijoules:

```text
energy_per_forward = window_energy_j / measure_iters
```

Important unit sanity check:

- A `200000`-forward PyTorch baseline window can consume around `1250 J`.
- That is not energy for one forward.
- It corresponds to roughly `6.25 mJ/forward`.
- With about `5.5 s` total CUDA-event time, `1250 J / 5.5 s` implies around
  `225 W`, which is plausible for this H100 workload.

### What We Explicitly Do Not Count As Energy Track

We do not rank full-subprocess Zeus energy around `run_and_check.py`.

That full-subprocess energy includes:

- Python process startup
- imports
- extension compile/load
- static checks
- correctness checks
- KernelBench profiler overhead
- cache clearing
- reference measurement

Those can be useful as diagnostics, but they are not the focused forward-only
energy result.

## Hardware And Infrastructure

Known active Prime Intellect H100 pods during the run:

| Pod name | Team | Provider | GPU | SSH endpoint | Notes |
|---|---|---|---|---|---|
| `kb-matmul-gelu-softmax-v1` | `naka3` | Datacrunch | `H100_80GB x1` | `root@86.38.238.67` | Main known-good scorer host |
| `kb-matmul-rl-scorer-h100b` | `naka7` | Lambda Labs | `H100_80GB x1` | `ubuntu@209.20.159.129` | Repo present |
| `kb-matmul-rl-scorer-h100c` | `naka7` | Lambda Labs | `H100_80GB x1` | `ubuntu@192.222.52.104` | Repo present; initially missing Python deps |
| `kb-matmul-rl-scorer-h100d` | `naka6` | Datacrunch | `H100_80GB x1` | `root@86.38.238.163` | Repo partially present under `/home/ubuntu` |

For consistency, the headline comparison rows were measured on the known-good
Datacrunch H100 host:

```text
root@86.38.238.67
GPU: NVIDIA H100 80GB HBM3
Power limit shown by nvidia-smi: 700 W
```

The idle Lambda host was missing `litellm` initially; it was bootstrapped with:

```bash
python3 -m pip install --user -q litellm pybind11 zeus-ml
```

Cross-host comparisons should always be treated carefully. The strongest claims
below rely on paired candidate/baseline measurements from the same scorer run.

## Candidate And Baseline Taxonomy

### PyTorch Eager Equivalent

Purpose:

- Diagnostic sanity check.
- Verifies that the focused energy scorer gives nearly the same answer for a
  candidate that is semantically identical to the PyTorch baseline.

Implementation:

```python
self.linear = nn.Linear(in_features, out_features)
x = self.linear(x)
x = F.gelu(x)
x = F.softmax(x, dim=1)
```

Eligibility:

- Not an eligible custom-kernel submission.
- KernelBench static check rejects it because it uses PyTorch compute ops and
  has no visible CUDA `__global__` kernel.

Interpretation:

- It lands around `6.265 mJ/forward`, very close to the paired baseline
  `6.317 mJ/forward`.
- This is a useful sanity check for the energy scorer.

### torch.compile Equivalent

Purpose:

- Diagnostic modern-framework compiler baseline.

Implementation:

- Same PyTorch module as eager, with `torch.compile(..., mode="reduce-overhead")`
  around the forward implementation.

Eligibility:

- Not an eligible custom-kernel submission.
- KernelBench static check rejects it for PyTorch compute ops/no visible CUDA
  kernel.

Interpretation:

- It is worse for this tiny fixed-shape task: `9.651 mJ/forward` and
  `0.0570 ms/forward` focused latency.
- Likely overhead/compiled graph behavior is not favorable for this specific
  microbenchmark.

### Public Kernelsseum L2P99 Candidate

Source:

- Public KernelBench/Kernelsseum exact corresponding task:
  `Level 2: 99_Matmul_GELU_Softmax`
- Public result: `0.7604x` vs Torch on public L40S leaderboard
- Raw public solution:
  `https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/5c36dadcac0846f0a4bc95a39760ad9d.py`

Implementation:

- PyTorch `nn.Linear`
- Custom CUDA approximate tanh-GELU kernel
- PyTorch `F.softmax`

Eligibility:

- It passes correctness under our scorer.
- It has static warnings because it uses PyTorch `nn.Linear` and `F.softmax`.
- It is best treated as provenance/diagnostic rather than a clean custom-kernel
  baseline.

Interpretation:

- It is not competitive on our H100 focused energy metric:
  `6.794 mJ/forward` vs paired baseline `6.289 mJ/forward`.
- It is also slower in focused latency:
  `0.0318 ms` vs paired baseline `0.0276 ms`.

### cuBLASLt + Separate GELU + Separate Softmax

Purpose:

- Strong custom-kernel baseline.
- Tests whether "just leaving PyTorch" and using normal custom post-op kernels
  explains the gains.

Implementation:

```text
cuBLASLt GEMM
  -> custom add_bias_gelu_kernel
  -> custom softmax_rows_kernel
```

Key properties:

- Real CUDA kernels.
- No PyTorch compute ops in forward.
- Uses cuBLASLt for GEMM.
- Uses separate kernels for GELU and softmax, so it intentionally does not
  fuse the post-GEMM GELU+softmax into one kernel.

Measured result:

- `3.235 mJ/forward`
- Paired PyTorch baseline: `6.437 mJ/forward`
- Focused latency: `0.0184 ms`
- Time-track latency: `0.0212 ms`

Interpretation:

- This is a major improvement over PyTorch.
- It demonstrates that a large fraction of the win comes from replacing generic
  PyTorch postprocessing with custom kernels and avoiding some framework
  overhead.
- It is still meaningfully worse than the best fused candidates.

### cuBLASLt Bias Epilogue + Fused GELU/Softmax

Purpose:

- Stronger custom-kernel baseline.
- Tests whether using the cuBLASLt bias epilogue before fused GELU+softmax
  matches or beats our best candidates.

Implementation:

```text
cuBLASLt GEMM with CUBLASLT_EPILOGUE_BIAS
  -> custom fused exact GELU + row-wise softmax kernel
```

Implementation note:

- The first row-major descriptor attempt failed with
  `CUBLAS_STATUS_NOT_SUPPORTED`.
- The successful version uses the column-major descriptor pattern from the
  `v7` probe, while preserving the contiguous PyTorch output memory layout.

Measured result:

- `2.863 mJ/forward`
- Paired PyTorch baseline: `6.209 mJ/forward`
- Focused latency: `0.0159 ms`
- Time-track latency: `0.0193 ms`

Interpretation:

- This is a very strong engineering baseline.
- It is close to `v5`, and not far from `v7`.
- It suggests the main advantage is the shape-specialized post-GEMM path and
  careful cuBLASLt composition, not an exotic new algorithm.

### candidate_v5

File:

- `/tmp/kb-pr2/submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/candidate_v5_cublaslt_time_best.py`

Implementation:

```text
cuBLASLt GEMM
  -> custom fused bias + exact GELU + row-wise softmax kernel
```

Key details:

- Uses cuBLASLt matmul with TF32 fast compute mode.
- GEMM writes logits.
- Fused post-op kernel reads logits and bias, computes exact GELU, does row max
  reduction, exponentiation, row sum reduction, and writes normalized softmax.
- Softmax rows are length `1024`, which maps naturally to a one-row-per-block
  reduction pattern.

Measured 200k-forward stress result:

- Focused energy: `2.863 mJ/forward`
- Paired baseline: `6.274 mJ/forward`
- Focused latency: `0.01587 ms`
- Average candidate power during CUDA-event window: about `180.38 W`
- Average paired baseline power: about `228.34 W`
- Correctness: passed 25/25 stress trials at `rtol=atol=1e-4`
- Worst max absolute error: `4.3869949877262115e-06`
- Worst max relative error: `0.0007526325643993914`

Time-track result:

- Candidate: `0.0191 ms`
- Paired baseline: `0.0250 ms`
- Speedup: about `1.31x`

Interpretation:

- Strong focused-energy win over PyTorch.
- Similar to the bias-epilogue comparator in 200k-focused energy.
- Earlier shorter-window artifacts showed lower mJ/forward, but the 200k stress
  result is the more conservative number for slide claims.

### candidate_v7

File:

- `/tmp/kb-pr2/submissions/kernelbench_focus3/matmul_gelu_softmax_small/time/candidate_v7_tf32_cublaslt_probe.py`

Implementation:

```text
cuBLASLt or cuBLAS TF32 GEMM path
  -> custom fused bias + exact GELU + row-wise softmax kernel
```

Key details:

- Uses a cached cuBLASLt plan when available.
- Falls back to cuBLAS GEMM if the plan path is not available.
- Uses TF32 tensor-op math for the GEMM.
- Keeps exact GELU and softmax in the custom post-op kernel.
- Uses the column-major descriptor pattern for cuBLASLt while mapping back to
  the same row-major contiguous PyTorch tensor memory.

Measured 200k-forward stress result:

- Focused energy: `2.797 mJ/forward`
- Paired baseline: `6.430 mJ/forward`
- Focused latency: `0.01567 ms`
- Average candidate power during CUDA-event window: about `178.51 W`
- Average paired baseline power: about `225.25 W`
- Correctness: passed 25/25 stress trials at `rtol=atol=1e-4`
- Worst max absolute error: `4.3869949877262115e-06`
- Worst max relative error: `0.0007526325643993914`

Time-track result:

- Candidate: `0.0187 ms`
- Paired baseline: `0.0243 ms`
- Speedup: about `1.30x`

Interpretation:

- Best measured focused energy among current rows.
- Best time-track latency among current rows.
- The remaining gain over the strong bias-epilogue comparator is modest, so it
  should be presented as an incremental engineering improvement over a strong
  baseline, not a dramatic algorithmic breakthrough.

## Results Table: Focused Energy

All rows below use focused forward-only energy. Lower is better. Candidate and
baseline are paired within the same scorer run.

| Row | Eligible submission | Correct | Candidate mJ/fwd | Paired baseline mJ/fwd | Reduction vs paired baseline | Focused latency ms | Candidate avg W | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch eager equivalent | no | yes | `6.265` | `6.317` | `0.8%` | `0.02777` | `225.6` | Diagnostic sanity check |
| `torch.compile` equivalent | no | yes | `9.651` | `6.266` | `-54.0%` | `0.05696` | `169.5` | Worse energy due to much longer runtime |
| Kernelsseum L2P99 public | warning | yes | `6.794` | `6.289` | `-8.0%` | `0.03181` | `213.6` | Public exact-task solution, but uses PyTorch ops |
| cuBLASLt + separate GELU + separate softmax | yes | yes | `3.235` | `6.437` | `49.7%` | `0.01840` | `175.9` | Strong non-fused post-op custom baseline |
| cuBLASLt bias epilogue + fused GELU/softmax | yes | yes | `2.863` | `6.209` | `53.9%` | `0.01587` | `180.4` | Strong epilogue comparator |
| `candidate_v5` 200k stress | yes | yes | `2.863` | `6.274` | `54.4%` | `0.01587` | `180.4` | Conservative 200k stress number |
| `candidate_v7` 200k stress | yes | yes | `2.797` | `6.430` | `56.5%` | `0.01567` | `178.5` | Current best focused-energy row |

Notes:

- "Reduction vs paired baseline" is computed as
  `1 - candidate_mJ / baseline_mJ`.
- Negative reduction means the candidate used more energy than its paired
  PyTorch baseline.
- `candidate_v5` and the bias-epilogue comparator are effectively tied in the
  conservative 200k numbers; both are slightly behind `candidate_v7`.
- Older 20k-window artifacts for `v5/v7` showed lower candidate mJ/forward.
  For slide claims, prefer the 200k stress numbers above.

## Results Table: KernelBench Time Track

Time track uses KernelBench `run_and_check.py` CUDA-event latency. Lower is
better.

| Row | Eligible submission | Correct | Candidate ms | Paired baseline ms | Speedup vs paired baseline | Notes |
|---|---:|---:|---:|---:|---:|---|
| Kernelsseum L2P99 public | warning | yes | `0.0259` | `0.0242` | `0.93x` | Slower than baseline |
| cuBLASLt + separate GELU + separate softmax | yes | yes | `0.0212` | `0.0253` | `1.19x` | Strong custom baseline |
| cuBLASLt bias epilogue + fused GELU/softmax | yes | yes | `0.0193` | `0.0249` | `1.29x` | Strong epilogue comparator |
| `candidate_v5` | yes | yes | `0.0191` | `0.0250` | `1.31x` | Current energy-oriented submission |
| `candidate_v7` | yes | yes | `0.0187` | `0.0243` | `1.30x` | Current time leader |

Diagnostic PyTorch eager and `torch.compile` rows are omitted from this table
because the KernelBench static checker rejects them as custom-kernel
submissions.

## Correctness Summary

Headline correctness for `v5` and `v7`:

- 25/25 stress trials passed.
- Tolerance: `rtol=1e-4`, `atol=1e-4`.
- Worst max absolute error: `4.3869949877262115e-06`.
- Worst max relative error: `0.0007526325643993914`.
- Worst row-sum absolute error: `2.384185791015625e-07`.

Why errors are nonzero:

- GEMM uses TF32/fast float32 tensor-op paths.
- GELU and softmax are exact-formula in the post-op kernel, but inputs to the
  post-op may differ slightly because of GEMM math mode and accumulation.
- Observed errors remain well within the scoring tolerance.

Cheating/validity checks:

- Candidate inputs are generated by the baseline task.
- Baseline weights are copied into `ModelNew`.
- Candidate does not inspect hidden tests.
- Candidate does not cache fixed outputs.
- Candidate must define `ModelNew` compatible with the baseline constructor.
- Valid rows pass KernelBench correctness/static checks.
- Diagnostic rows that fail static checks are explicitly marked non-eligible.

## Intuition For The Improvement

The basic optimization is standard kernel fusion and specialization, not a new
mathematical algorithm.

PyTorch baseline path:

```text
nn.Linear
  -> PyTorch GELU
  -> PyTorch softmax
```

Likely costs:

- Multiple framework dispatches.
- Multiple CUDA kernel launches.
- Intermediate tensors materialized between stages.
- Extra global memory traffic:
  - write linear output
  - read linear output for GELU
  - write GELU output
  - read GELU output for softmax
  - write softmax output
- Generic kernels that are not specialized only for `[64, 1024]`.

Our leading path:

```text
cuBLASLt/TF32 GEMM
  -> shape-specialized fused post-op kernel
       bias
       exact GELU
       row max
       exp
       row sum
       normalize
```

Why energy drops:

- The active GPU time is much shorter.
- The post-GEMM path avoids extra PyTorch dispatch overhead.
- GELU and softmax are fused into a row-local reduction kernel.
- Intermediate memory traffic is reduced.
- The row size is fixed at `1024`, which is convenient for one-row-per-block
  reductions.
- Power is also lower for candidate windows, but the main effect is shorter
  runtime:
  - Baseline focused latency is around `0.027-0.029 ms`.
  - Best candidate focused latency is around `0.0157 ms`.

The strong baseline ladder clarifies the story:

- Moving from PyTorch to custom kernels is a big win:
  `6.3 -> 3.235 mJ/forward`.
- Folding bias and fusing GELU+softmax is another win:
  `3.235 -> 2.863 mJ/forward`.
- The current best candidate is a smaller additional improvement:
  `2.863 -> 2.797 mJ/forward`.

This is why the slide story should avoid overclaiming novelty. A fair phrasing:

```text
We did not invent GELU+softmax fusion. The result is a strong, shape-specific
engineering optimization: cuBLASLt/TF32 GEMM plus a specialized fused row-wise
postprocess that beats PyTorch, public exact-task code, and stronger custom
cuBLASLt baselines on focused H100 energy.
```

## What Is Actually Innovative

Not innovative:

- Fusing elementwise ops.
- Fusing GELU with a downstream row-wise softmax.
- Using cuBLASLt for GEMM.
- Using TF32 tensor cores for float32 GEMM.

Potentially meaningful/interesting:

- The measurement-driven composition for this exact shape.
- Demonstrating that a strong custom-kernel baseline still leaves some room.
- The GEPA/autoresearch workflow that generated and verified candidate variants.
- The focus on energy rather than only CUDA-event runtime.
- The explicit strong-baseline ladder to avoid self-deception.

For slide language:

- Avoid: "novel fused GELU-softmax kernel."
- Prefer: "shape-specialized fused post-GEMM path."
- Prefer: "energy-optimized composition of cuBLASLt/TF32 GEMM and a custom
  row-wise postprocess."
- Prefer: "beats strong custom baselines, not just PyTorch."

## What The Current Result Does Not Prove

The result does not prove:

- General superiority on other shapes.
- General superiority on other GPUs.
- A new algorithm for softmax, GELU, or GEMM.
- A production-ready universal kernel.
- Superiority over a carefully implemented CUTLASS/CuTe kernel.

The result does support:

- For this exact Focus3 task and H100 scorer, `v7` is currently the best
  focused-energy candidate we measured.
- The candidate beats PyTorch by a large margin.
- The candidate beats a public exact-task solution.
- The candidate beats a straightforward cuBLASLt + separate custom post-op
  baseline.
- The candidate is competitive with and slightly better than a strong cuBLASLt
  bias-epilogue + fused post-op comparator.

## CUTLASS/CuTe Status

CUTLASS/CuTe was not measured in this pass.

Reason:

- The active H100 host did not have a CUTLASS checkout under `/root` or
  `/usr/local`.
- `import cutlass` was not available.
- Triton was installed, but a Triton baseline is not the same as CUTLASS/CuTe.

Recommended next step:

1. Install or clone CUTLASS on an H100 host.
2. Implement a CUTLASS/CuTe comparator for:
   - fixed `M=64`, `N=1024`, `K=1024`
   - FP32 input/output
   - TF32 tensor-op GEMM if allowed
   - bias handling
   - exact GELU
   - row-wise softmax
3. Run the same `score_focus3_submission.py` protocol.
4. Add it to the strong baseline ladder before making a stronger novelty claim.

## Suggested Slide Structure

### Slide 1: Problem And Metric

Title:

```text
Energy-focused optimization of Linear -> GELU -> Softmax on H100
```

Bullets:

- Task: `Linear(1024,1024)` on batch `64`, exact GELU, row-wise softmax.
- Primary metric: focused H100 GPU board energy around repeated forward calls.
- Secondary metric: KernelBench CUDA-event latency.
- Goal: generate competitive custom-kernel submissions, not just optimize
  PyTorch code.

Visual:

```text
PyTorch baseline:
Linear -> GELU -> Softmax

Candidate:
cuBLASLt/TF32 GEMM -> fused row-wise postprocess
```

### Slide 2: Measurement Discipline

Bullets:

- Correctness/time from KernelBench `run_and_check.py`.
- Energy from Zeus windows around `model(*inputs)` forwards only.
- `200` warmups, `200000` measured forwards.
- Two orders: candidate then baseline, baseline then candidate.
- Report `mJ/forward`, not just total window joules.

Speaker note:

```text
We explicitly do not rank full subprocess energy because that includes import,
compile, static checks, correctness checks, and profiler overhead.
```

### Slide 3: Baseline Ladder

Use this table:

| Row | Focused energy |
|---|---:|
| PyTorch eager | `6.265 mJ/fwd` |
| public Kernelsseum exact-task solution | `6.794 mJ/fwd` |
| cuBLASLt + separate GELU + separate softmax | `3.235 mJ/fwd` |
| cuBLASLt bias epilogue + fused GELU/softmax | `2.863 mJ/fwd` |
| our `candidate_v7` | `2.797 mJ/fwd` |

Message:

```text
The right comparison is not only against PyTorch. Strong custom-kernel baselines
close most of the gap, but our best candidate remains slightly better.
```

### Slide 4: Main Result

Use this table:

| Candidate | Energy | Baseline | Reduction | Time-track |
|---|---:|---:|---:|---:|
| `candidate_v5` | `2.863 mJ/fwd` | `6.274 mJ/fwd` | `54.4%` | `0.0191 ms` |
| `candidate_v7` | `2.797 mJ/fwd` | `6.430 mJ/fwd` | `56.5%` | `0.0187 ms` |

Message:

```text
Both candidates roughly halve focused H100 energy. v7 is the current best in
both focused energy and KernelBench time among the conservative 200k-window rows.
```

### Slide 5: Why It Works

Bullets:

- Keep GEMM on optimized cuBLASLt/TF32.
- Specialize post-GEMM processing to fixed `64 x 1024` rows.
- Fuse bias/GELU/softmax-related row reductions where possible.
- Reduce framework dispatch and intermediate memory traffic.
- Shorten active GPU time.

Suggested graphic:

```text
PyTorch:
GEMM write -> GELU read/write -> softmax read/write

Candidate:
GEMM write -> fused row kernel write final output
```

### Slide 6: Honesty And Caveats

Bullets:

- This is standard fusion/specialization, not a new algorithm.
- Shape-specific result: `M=64`, `N=1024`, `K=1024`.
- Needs CUTLASS/CuTe comparator before broader claims.
- Cross-host energy comparisons must be paired and rerun.
- Full-subprocess energy is diagnostic only.

### Slide 7: Next Experiments

Bullets:

- Add CUTLASS/CuTe baseline.
- Try a one-pass custom matmul+GELU+softmax kernel, knowing beating cuBLASLt is
  hard.
- Sweep measurement windows and report variance.
- Test related shapes and batch sizes.
- Explore whether final writes/intermediate logits can be further reduced.

## Suggested Short Talk Track

```text
We focused on a small but representative fused operation: Linear -> GELU ->
Softmax. The naive PyTorch reference spends energy not only on arithmetic but on
multiple framework-dispatched kernels and intermediate tensor traffic. Our
autoresearch loop found candidates that keep GEMM on cuBLASLt/TF32 and specialize
the post-GEMM computation for exactly 64 rows of 1024 columns.

The primary metric is forward-only GPU board energy on H100. We measure it with
Zeus around repeated forward calls, not around the entire KernelBench process.
This matters because the full process includes compile, imports, correctness
checks, and profiler overhead.

Against PyTorch, the best candidate reduces focused energy from around 6.43
mJ/forward to 2.80 mJ/forward, about a 56 percent reduction. But the more
important question is whether that is just a standard fusion win. So we added
stronger baselines. A cuBLASLt GEMM plus separate custom GELU and softmax gets to
3.24 mJ/forward. A cuBLASLt bias epilogue plus fused GELU/softmax gets to 2.86
mJ/forward. Our best candidate is still slightly better at 2.80 mJ/forward.

The honest conclusion is that this is not a novel GELU-softmax algorithm. It is
a shape-specific, energy-optimized kernel composition that beats PyTorch, public
exact-task code, and strong cuBLASLt/custom-kernel baselines under a focused H100
energy metric.
```

## Open Questions

1. How much variance remains across repeated runs on the same H100?
2. Does the ranking hold on Lambda H100 vs Datacrunch H100?
3. Can a CUTLASS/CuTe implementation beat the current candidates?
4. Can a custom one-pass matmul+GELU+softmax kernel reduce writes enough to beat
   cuBLASLt despite weaker GEMM engineering?
5. How robust is the result across:
   - batch size
   - output width
   - input distribution
   - TF32 enabled/disabled
   - exact FP32 GEMM vs TF32 tensor-op GEMM
6. Is the remaining `2.797 -> 2.863 mJ/fwd` gap over the bias-epilogue baseline
   stable, or mostly measurement noise?

## Recommended Claims For Slides

Strong, supported:

- "Focused H100 forward energy drops by about 56% versus paired PyTorch
  baseline."
- "The best candidate beats a strong cuBLASLt + separate custom post-op
  baseline."
- "The best candidate is slightly better than a cuBLASLt bias-epilogue + fused
  post-op comparator."
- "The improvement comes from shape-specific composition and fusion, not a new
  softmax/GELU algorithm."

Avoid or qualify:

- "Novel kernel algorithm."
- "State of the art generally."
- "Beats all public implementations."
- "Energy is 2.8 mJ everywhere."
- "The PyTorch comparison alone proves innovation."

Better phrasing:

```text
On the Focus3 `matmul_gelu_softmax_small` task, we found a shape-specialized
H100 implementation that reduces focused forward GPU board energy by about 56%
relative to paired PyTorch and remains ahead of stronger custom cuBLASLt
baselines. The win is best understood as careful kernel composition and fusion
validated by energy-focused measurement.
```

