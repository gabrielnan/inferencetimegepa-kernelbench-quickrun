from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferencetimegepa.config import load_experiment_config
from inferencetimegepa.communication import (
    build_mock_agents,
    build_openai_agents,
    run_communicating_agents,
    summarize_communicated,
    write_communicated_jsonl,
)
from inferencetimegepa.contribution import load_dict_jsonl, score_contributions
from inferencetimegepa.contribution import write_dict_jsonl as write_contribution_jsonl
from inferencetimegepa.gepa import (
    build_reflection_prompt,
    load_prompt_candidates,
    pareto_frontier,
    write_prompt_candidates,
)
from inferencetimegepa.kernelbench import load_kernel_tasks, run_kernelbench_eval, write_kernel_results
from inferencetimegepa.optimize_anything_runner import run_codex_reflection_packet, run_optimize_anything
from inferencetimegepa.preflight import run_pre_gpu_smoke
from inferencetimegepa.router import route_task
from inferencetimegepa.runner import (
    merge_task_results,
    run_agent,
    run_routed_agents,
    run_single_agent,
    summarize,
    summarize_dict_rows,
    write_dict_jsonl,
    write_jsonl,
)
from inferencetimegepa.samplers import build_sampler
from inferencetimegepa.tasks import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="inferencetimegepa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_single = subparsers.add_parser("run-single", help="Run setup 1: single-agent K rollouts")
    run_single.add_argument("--tasks", type=Path, required=True)
    run_single.add_argument("--k", type=int, default=8)
    run_single.add_argument("--sampler", choices=["mock", "openai"], default="mock")
    run_single.add_argument("--model")
    run_single.add_argument("--temperature", type=float, default=0.8)
    run_single.add_argument("--max-tokens", type=int, default=512)
    run_single.add_argument("--seed", type=int, default=0)
    run_single.add_argument("--timeout-s", type=float, default=5.0)
    run_single.add_argument("--out", type=Path)

    run_one = subparsers.add_parser("run-agent", help="Run one hardware-backed agent worker")
    run_one.add_argument("--tasks", type=Path, required=True)
    run_one.add_argument("--agent-id", required=True)
    run_one.add_argument("--k", type=int, default=8)
    run_one.add_argument("--system-prompt")
    run_one.add_argument("--sampler", choices=["mock", "openai"], default="openai")
    run_one.add_argument("--model")
    run_one.add_argument("--temperature", type=float, default=0.8)
    run_one.add_argument("--max-tokens", type=int, default=512)
    run_one.add_argument("--seed", type=int, default=0)
    run_one.add_argument("--timeout-s", type=float, default=5.0)
    run_one.add_argument("--out", type=Path, required=True)

    routed = subparsers.add_parser("run-routed", help="Run router-selected specialist agents")
    routed.add_argument("--config", type=Path, default=Path("configs/agents_2xh200.json"))
    routed.add_argument("--sampler", choices=["mock", "openai"], default="mock")
    routed.add_argument("--model")
    routed.add_argument("--rollouts-per-agent", type=int)
    routed.add_argument("--temperature", type=float, default=0.8)
    routed.add_argument("--max-tokens", type=int, default=512)
    routed.add_argument("--seed", type=int, default=0)
    routed.add_argument("--timeout-s", type=float, default=5.0)
    routed.add_argument("--out", type=Path, required=True)

    merge = subparsers.add_parser("merge-agent-runs", help="Merge per-agent JSONL outputs")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--out", type=Path, required=True)

    pre_gpu = subparsers.add_parser("pre-gpu-smoke", help="Validate config and run mock four-agent smoke")
    pre_gpu.add_argument("--config", type=Path, default=Path("configs/agents_2xh200.json"))
    pre_gpu.add_argument("--out-dir", type=Path, default=Path("runs/pre_gpu_smoke"))
    pre_gpu.add_argument("--k", type=int, help="Override rollouts per agent for the smoke run")

    contribution = subparsers.add_parser("score-contribution", help="Add contribution scores to rollout JSONL")
    contribution.add_argument("--input", type=Path, required=True)
    contribution.add_argument("--out", type=Path, required=True)
    contribution.add_argument("--method", choices=["heuristic", "openai", "anthropic"], default="heuristic")
    contribution.add_argument("--judge-model")

    communicate = subparsers.add_parser("run-communicate", help="Run communicating agents and evaluate one final committed solution")
    communicate.add_argument("--tasks", type=Path, required=True)
    communicate.add_argument("--sampler", choices=["mock", "openai"], default="mock")
    communicate.add_argument("--model")
    communicate.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    communicate.add_argument("--committer-index", type=int, default=0)
    communicate.add_argument("--committer-port", type=int)
    communicate.add_argument("--temperature", type=float, default=0.8)
    communicate.add_argument("--max-tokens", type=int, default=1024)
    communicate.add_argument("--comm-steps", type=int, default=4)
    communicate.add_argument("--seed", type=int, default=0)
    communicate.add_argument("--timeout-s", type=float, default=5.0)
    communicate.add_argument("--out", type=Path, required=True)

    kernelbench = subparsers.add_parser("run-kernelbench", help="Evaluate KernelBench kernels with optional Zeus energy metrics")
    kernelbench.add_argument("--tasks", type=Path, required=True)
    kernelbench.add_argument("--benchmark-cmd", help="Command template; use {kernel} for the kernel path")
    kernelbench.add_argument(
        "--compare-cmd",
        help="Pairwise command template; use {baseline} and {candidate}; emits baseline_* and candidate_* JSON fields",
    )
    kernelbench.add_argument("--zeus", action="store_true")
    kernelbench.add_argument("--mock", action="store_true", help="Use the toy local evaluator. Never use for GEPA/Devin optimization.")
    kernelbench.add_argument("--timeout-s", type=float, default=60.0)
    kernelbench.add_argument("--out", type=Path, required=True)

    gepa_frontier = subparsers.add_parser("gepa-frontier", help="Compute the prompt-candidate Pareto frontier")
    gepa_frontier.add_argument("--candidates", type=Path, required=True)
    gepa_frontier.add_argument("--metrics", nargs="+", default=["correct_rate", "mean_speedup", "accepted_rate"])
    gepa_frontier.add_argument("--out", type=Path, required=True)

    gepa_reflect = subparsers.add_parser("gepa-reflect-prompt", help="Build a reflection prompt from rollout traces")
    gepa_reflect.add_argument("--role", required=True)
    gepa_reflect.add_argument("--rollouts", type=Path, required=True)
    gepa_reflect.add_argument("--candidates", type=Path, default=Path("prompts/candidates.jsonl"))
    gepa_reflect.add_argument("--metrics", nargs="+", default=["correct_rate", "mean_speedup", "accepted_rate"])
    gepa_reflect.add_argument("--out", type=Path, required=True)

    optimize_anything = subparsers.add_parser("optimize-anything", help="Run GEPA optimize_anything over Devin team prompts")
    optimize_anything.add_argument("--dataset", type=Path, required=True)
    optimize_anything.add_argument("--out-dir", type=Path, default=Path("runs/gepa_optimize_anything"))
    optimize_anything.add_argument("--max-metric-calls", type=int, default=4)
    optimize_anything.add_argument("--max-candidate-proposals", type=int, default=1)
    optimize_anything.add_argument("--reflection-lm", default="openai/gpt-5.1")
    optimize_anything.add_argument("--kernelbench-timeout-s", type=float, default=900.0)
    optimize_anything.add_argument(
        "--devin-command",
        required=True,
        help="Command template. Available fields: {task_id}, {prompt_dir}, {eval_dir}, {baseline}, {entry_point}. Must write {eval_dir}/candidate.py.",
    )
    optimize_anything.add_argument(
        "--compare-cmd",
        required=True,
        help="KernelBench pairwise command template. Available fields passed by run-kernelbench: {baseline}, {candidate}, {task_id}, {entry_point}.",
    )

    codex_packet = subparsers.add_parser("codex-reflection-packet", help="Run Devin/KernelBench once and write a Codex GEPA reflection packet")
    codex_packet.add_argument("--dataset", type=Path, required=True)
    codex_packet.add_argument("--out-dir", type=Path, default=Path("runs/codex_reflection"))
    codex_packet.add_argument(
        "--devin-command",
        required=True,
        help="Command template. Available fields: {task_id}, {prompt_dir}, {eval_dir}, {baseline}, {entry_point}, {attempt_index}, {team_size}, {team_role}, {router_plan}. Must write {eval_dir}/candidate.py.",
    )
    codex_packet.add_argument(
        "--router-command",
        default=None,
        help="Optional router command template. Available fields: {task_id}, {prompt_dir}, {eval_dir}, {router_plan_path}, {baseline}, {entry_point}. Must write {router_plan_path}.",
    )
    codex_packet.add_argument("--team-size", type=int, default=1, help="Number of Devin implementation subagents per task.")
    codex_packet.add_argument("--compare-cmd", required=True)
    codex_packet.add_argument("--kernelbench-timeout-s", type=float, default=900.0)

    args = parser.parse_args()
    if args.command == "run-single":
        tasks = load_jsonl(args.tasks)
        sampler = build_sampler(
            args.sampler,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        results = run_single_agent(
            tasks=tasks,
            sampler=sampler,
            k=args.k,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        if args.out:
            write_jsonl(args.out, results)
        print(json.dumps(summarize(results), indent=2, sort_keys=True))
    elif args.command == "run-agent":
        tasks = load_jsonl(args.tasks)
        sampler = build_sampler(
            args.sampler,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        results = run_agent(
            tasks=tasks,
            sampler=sampler,
            agent_id=args.agent_id,
            k=args.k,
            seed=args.seed,
            timeout_s=args.timeout_s,
            system_prompt=args.system_prompt,
        )
        write_jsonl(args.out, results)
        print(json.dumps(summarize(results), indent=2, sort_keys=True))
    elif args.command == "run-routed":
        config = load_experiment_config(args.config)
        tasks = load_jsonl(config.tasks)
        samplers_by_agent = {
            agent.agent_id: build_sampler(
                args.sampler,
                model=args.model or config.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            for agent in config.agents
        }
        prompts_by_agent = {agent.agent_id: agent.system_prompt for agent in config.agents}
        results = run_routed_agents(
            tasks=tasks,
            samplers_by_agent=samplers_by_agent,
            prompts_by_agent=prompts_by_agent,
            router=lambda task: route_task(task, config.agents),
            rollouts_per_agent=args.rollouts_per_agent or config.rollouts_per_agent,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        write_jsonl(args.out, results)
        print(json.dumps(summarize(results), indent=2, sort_keys=True))
    elif args.command == "merge-agent-runs":
        rows = merge_task_results(args.inputs)
        write_dict_jsonl(args.out, rows)
        print(json.dumps(summarize_dict_rows(rows), indent=2, sort_keys=True))
    elif args.command == "pre-gpu-smoke":
        config = load_experiment_config(args.config)
        report = run_pre_gpu_smoke(config=config, out_dir=args.out_dir, k=args.k)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "score-contribution":
        rows = load_dict_jsonl(args.input)
        scored = score_contributions(rows, method=args.method, judge_model=args.judge_model)
        write_contribution_jsonl(args.out, scored)
        print(json.dumps(summarize_dict_rows(scored), indent=2, sort_keys=True))
    elif args.command == "run-communicate":
        tasks = load_jsonl(args.tasks)
        if args.sampler == "mock":
            agents = build_mock_agents(len(args.ports))
        else:
            if not args.model:
                raise ValueError("--model is required for --sampler openai")
            agents = build_openai_agents(
                model=args.model,
                ports=args.ports,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        if args.committer_port is not None:
            committer = build_openai_agents(
                model=args.model,
                ports=[args.committer_port],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )[0]
            committer = type(committer)(agent_id="router", sampler=committer.sampler, system_prompt=committer.system_prompt)
        else:
            committer = agents[args.committer_index]
        results = run_communicating_agents(
            tasks=tasks,
            agents=agents,
            committer=committer,
            seed=args.seed,
            timeout_s=args.timeout_s,
            comm_steps=args.comm_steps,
        )
        write_communicated_jsonl(args.out, results)
        print(json.dumps(summarize_communicated(results), indent=2, sort_keys=True))
    elif args.command == "run-kernelbench":
        tasks = load_kernel_tasks(args.tasks)
        results = run_kernelbench_eval(
            tasks=tasks,
            benchmark_cmd=args.benchmark_cmd,
            zeus_enabled=args.zeus,
            timeout_s=args.timeout_s,
            compare_cmd=args.compare_cmd,
            allow_mock=args.mock,
        )
        write_kernel_results(args.out, results)
        print(
            json.dumps(
                {
                    "tasks": len(results),
                    "correct": sum(r.correct for r in results),
                    "improved": sum(r.improved for r in results),
                    "accepted": sum(r.accepted for r in results),
                    "mean_score": sum(r.score for r in results) / len(results) if results else 0.0,
                    "mean_reward": sum(r.reward for r in results) / len(results) if results else 0.0,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gepa-frontier":
        candidates = load_prompt_candidates(args.candidates)
        frontier = pareto_frontier(candidates, args.metrics)
        write_prompt_candidates(args.out, frontier)
        print(json.dumps({"candidates": len(candidates), "frontier": len(frontier)}, indent=2, sort_keys=True))
    elif args.command == "gepa-reflect-prompt":
        candidates = load_prompt_candidates(args.candidates)
        frontier = pareto_frontier(candidates, args.metrics)
        prompt = build_reflection_prompt(args.role, args.rollouts, frontier)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt, encoding="utf-8")
        print(json.dumps({"frontier": len(frontier), "out": str(args.out)}, indent=2, sort_keys=True))
    elif args.command == "optimize-anything":
        result = run_optimize_anything(
            repo_root=Path.cwd(),
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            max_metric_calls=args.max_metric_calls,
            max_candidate_proposals=args.max_candidate_proposals,
            reflection_lm=args.reflection_lm,
            devin_command=args.devin_command,
            router_command=args.router_command,
            compare_cmd=args.compare_cmd,
            kernelbench_timeout_s=args.kernelbench_timeout_s,
            team_size=args.team_size,
        )
        print(
            json.dumps(
                {
                    "best_idx": getattr(result, "best_idx", None),
                    "num_candidates": getattr(result, "num_candidates", None),
                    "total_metric_calls": getattr(result, "total_metric_calls", None),
                    "out_dir": str(args.out_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "codex-reflection-packet":
        packet_path = run_codex_reflection_packet(
            repo_root=Path.cwd(),
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            devin_command=args.devin_command,
            router_command=args.router_command,
            compare_cmd=args.compare_cmd,
            kernelbench_timeout_s=args.kernelbench_timeout_s,
            team_size=args.team_size,
        )
        print(json.dumps({"packet": str(packet_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
