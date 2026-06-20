# Architecture

Codex or GEPA owns the prompt-optimization loop. Devin owns implementation attempts.

```text
Codex/GEPA proposer
  -> writes/updates prompts/candidates.jsonl and prompts/roles/*.md
  -> Devin CLI team executes .devin/tasks/*
  -> KernelBench correctness/perf outputs go to runs/
  -> Codex or optimize_anything reads traces as ASI
  -> Codex/GEPA proposes new team prompts
  -> Pareto frontier retains non-dominated prompts
```

Metrics:

- `correct_rate`: fraction of candidates passing correctness.
- `accepted_rate`: fraction of candidates accepted over baseline.
- `mean_speedup`: average speedup for correct candidates.
- `review_escape_rate`: fraction of incorrect candidates not caught by reviewer.
- `time_to_candidate`: wall-clock time for an implementation attempt.

The GEPA candidate must not know or optimize against a toy scoring shortcut. It only knows that Devin must produce candidate kernels and that an external scorer returns correctness, latency, optional Zeus energy, and accepted/rejected status. Hidden/held-out KernelBench cases should be used for validation once the local loop works.

The first remote smoke target is Prime Intellect:

```bash
ssh ubuntu@209.20.158.160 -p 22
```
