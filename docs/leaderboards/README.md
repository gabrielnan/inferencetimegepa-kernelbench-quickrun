# Focus3 Leaderboards

These leaderboards cover the three training tasks in `benchmarks/kernelbench_focus3/train_tasks.jsonl`.

Each task has two tracks:

- Energy track: lower warmed repeated-forward Zeus GPU energy is better.
- Time track: lower KernelBench CUDA-event latency is better.

Every leaderboard includes the PyTorch baseline. Candidate rows should use
paired measurements from the same scorer run so hardware, driver, power state,
and benchmark settings are comparable. The full scoring contract is in
`docs/leaderboards/evaluation.md`.

Use `scripts/score_focus3_submission.py` to produce the normalized JSON record
for a candidate before adding it to a ranked table.

## Tasks

| Task | Category | Leaderboard |
|---|---|---|
| `kb_focus3_mlp_small` | model | `docs/leaderboards/mlp_small.md` |
| `kb_focus3_matmul_gelu_softmax_small` | fusion | `docs/leaderboards/matmul_gelu_softmax_small.md` |
| `kb_focus3_mingpt_causal_attention_small` | attention | `docs/leaderboards/mingpt_causal_attention_small.md` |

The held-out validation task, `kb_focus3_layernorm_small`, is intentionally not included in this three-task training leaderboard set.
