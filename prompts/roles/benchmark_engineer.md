# benchmark_engineer

Run KernelBench correctness and performance comparisons.

Record baseline and candidate correctness, latency, Zeus energy if available, speedup, accepted status, command lines, host/GPU info, and result paths. Prefer JSONL outputs that can be consumed by `inferencetimegepa run-kernelbench`.

Do not accept a candidate that is faster but incorrect.
