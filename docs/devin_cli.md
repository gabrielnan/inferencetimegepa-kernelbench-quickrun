# Devin CLI Setup

Official docs:

- Quickstart: `https://docs.devin.ai/cli`
- Commands: `https://docs.devin.ai/cli/reference/commands`
- Enterprise auth: `https://docs.devin.ai/cli/enterprise/devin-auth`

Install options from the official docs:

```bash
curl -fsSL https://cli.devin.ai/install.sh | bash
```

or install from Devin Desktop with `Cmd+Shift+P` and `Install Devin CLI`.

Authenticate:

```bash
devin setup
devin auth login
```

For remote or SSH sessions:

```bash
devin setup --force-manual-token-flow
```

Run Devin from the repo root so it reads `AGENTS.md`, `.devin/tasks`, and `prompts/roles`.

Example local prompts to paste into Devin CLI:

```text
Use .devin/tasks/kernel_author.md and prompts/roles/kernel_author.md. Implement one KernelBench smoke candidate under runs/candidates and leave notes under runs/notes.
```

```text
Use .devin/tasks/benchmark_engineer.md. Benchmark the candidate against the baseline, save JSONL under runs/kernelbench, and record exact commands.
```
