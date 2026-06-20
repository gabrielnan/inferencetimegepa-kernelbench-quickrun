# mingpt_causal_attention_small Leaderboard

Target: `benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py`

Reference behavior: batch `16`, sequence length `256`, embedding size `512`, `8` heads, causal self-attention, softmax, attention projection, and zero dropout.

Hardware for future scored rows: NVIDIA H100 80GB HBM3 on Prime Intellect / Datacrunch.

The benchmark has two tracks:

- Energy track: lower warmed repeated-forward Zeus GPU energy is better.
- Time track: lower KernelBench CUDA-event latency is better.

No verified custom submissions have been recorded for this task yet. The PyTorch baseline is included so the table shape is ready for the first paired H100 result.

## Energy Track

| Rank | Submission | Correct | Candidate energy | Paired baseline energy | Delta vs paired baseline | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | PyTorch baseline | yes | pending first paired H100 run | n/a | n/a | Reference implementation in `benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py` |

## Time Track

| Rank | Submission | Correct | Candidate latency | Paired baseline latency | Speedup vs paired baseline | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | PyTorch baseline | yes | pending first paired H100 run | n/a | `1.00x` | Reference implementation in `benchmarks/kernelbench_focus3/mingpt_causal_attention_small.py` |

## Reproduction

Run the controlled scorer before adding or updating rows:

```bash
python3 scripts/score_focus3_submission.py \
  --task mingpt_causal_attention_small \
  --candidate <candidate.py> \
  --entry-point ModelNew \
  --output runs/leaderboard_scores/mingpt_causal_attention_small.json \
  --num-correct-trials 1 \
  --num-perf-trials 5 \
  --warmup-iters 200 \
  --measure-iters 200000 \
  --timeout 300
```

Only add candidate rows after recording correctness, candidate latency, paired
baseline latency, candidate energy, and paired baseline energy from the same
scorer run. Full-subprocess Zeus energy is diagnostic only.
