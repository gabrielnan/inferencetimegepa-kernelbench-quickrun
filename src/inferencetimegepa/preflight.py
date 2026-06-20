from __future__ import annotations

import json
import shutil
from pathlib import Path

from inferencetimegepa.config import ExperimentConfig
from inferencetimegepa.runner import merge_task_results, run_agent, summarize, summarize_dict_rows, write_dict_jsonl, write_jsonl
from inferencetimegepa.samplers import MockSampler
from inferencetimegepa.tasks import load_jsonl


def run_pre_gpu_smoke(config: ExperimentConfig, out_dir: Path, k: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(config.tasks)
    sampler = MockSampler()
    rollouts_per_agent = k or min(config.rollouts_per_agent, 2)
    agent_outputs: list[Path] = []
    agent_summaries: dict[str, dict] = {}

    for idx, agent in enumerate(config.agents):
        path = out_dir / f"{agent.agent_id}.jsonl"
        results = run_agent(
            tasks=tasks,
            sampler=sampler,
            agent_id=agent.agent_id,
            k=rollouts_per_agent,
            seed=idx * 10_000,
            timeout_s=2.0,
            system_prompt=agent.system_prompt,
        )
        write_jsonl(path, results)
        agent_outputs.append(path)
        agent_summaries[agent.agent_id] = summarize(results)

    merged = merge_task_results(agent_outputs)
    merged_path = out_dir / "merged.jsonl"
    write_dict_jsonl(merged_path, merged)

    report = {
        "config": config.name,
        "model": config.model,
        "tasks": str(config.tasks),
        "agents": len(config.agents),
        "rollouts_per_agent": rollouts_per_agent,
        "agent_outputs": [str(path) for path in agent_outputs],
        "merged_output": str(merged_path),
        "agent_summaries": agent_summaries,
        "merged_summary": summarize_dict_rows(merged),
        "tools": {
            "devin": bool(shutil.which("devin")),
            "inferencetimegepa": bool(shutil.which("inferencetimegepa")),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
