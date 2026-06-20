# GEPA Reference

This repo follows the official GEPA implementation and paper:

- Paper: `GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning`, arXiv `2507.19457`.
- Official implementation: `https://github.com/gepa-ai/gepa`.
- Official docs: `https://gepa-ai.github.io/gepa/`.

The official GEPA loop is:

1. Select a candidate from the Pareto frontier.
2. Execute it on a minibatch and capture full traces.
3. Reflect over traces and actionable side information.
4. Mutate the textual candidate.
5. Accept useful candidates and update the Pareto frontier.

For this repo, the textual candidate is not a CUDA kernel. The textual candidates are the Devin team prompts in `prompts/roles/` and rows in `prompts/candidates.jsonl`.

## Mapping To This Repo

- Executor: Devin CLI team running `.devin/tasks/*` to implement, review, and benchmark KernelBench kernels.
- Traces: files under `runs/`, including implementation notes, review notes, benchmark logs, and KernelBench JSONL.
- ASI: compiler errors, correctness failures, profiler output, latency, Zeus energy summaries, shape/dtype issues, and reviewer notes.
- Reflector: Codex or a GEPA reflection model generating prompt mutations from traces.
- Curator/Pareto tracker: `inferencetimegepa gepa-frontier` and, later, the official `gepa` package.

## Official API Hook

The official implementation supports `gepa.optimize()` for structured prompt optimization and `optimize_anything` for arbitrary text artifacts. This repo starts with `optimize_anything` because the evaluator shells out to Devin and KernelBench.

```python
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

seed_candidate = {
    "kernel_author": open("prompts/roles/kernel_author.md").read(),
    "correctness_reviewer": open("prompts/roles/correctness_reviewer.md").read(),
    "benchmark_engineer": open("prompts/roles/benchmark_engineer.md").read(),
    "gepa_reflector": open("prompts/roles/gepa_reflector.md").read(),
}

# Evaluator launches Devin on a small KernelBench batch,
# logs runs/* traces as ASI, and returns task metrics.
result = optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=run_devin_team_kernelbench_eval,
    dataset=[],
    valset=[],
    objective="Improve Devin team prompts for KernelBench correctness, accepted-rate, and speedup.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=100),
        reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
    ),
)
```

The local `src/inferencetimegepa/gepa.py` utilities are intentionally lightweight scaffolding until the full official adapter is wired in.

Do not expose the scoring implementation or hidden tests to the candidate prompt. GEPA should see score summaries and actionable traces, not a hackable scoring rule.
