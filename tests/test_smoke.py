from inferencetimegepa.router import route_task
from inferencetimegepa.runner import merge_task_results, run_agent, run_routed_agents, run_single_agent, summarize, write_jsonl
from inferencetimegepa.samplers import MockSampler
from inferencetimegepa.tasks import CodeTask
from inferencetimegepa.config import load_experiment_config
from inferencetimegepa.contribution import score_contributions
from inferencetimegepa.preflight import run_pre_gpu_smoke
from inferencetimegepa.kernelbench import KernelTask, run_kernelbench_eval


def test_single_agent_smoke_pipeline_runs() -> None:
    tasks = [
        CodeTask(
            task_id="t/add",
            prompt="Write add(a, b)",
            entry_point="add",
            tests=("assert add(2, 3) == 5",),
        )
    ]

    results = run_single_agent(tasks, MockSampler(), k=3, seed=0, timeout_s=2.0)
    summary = summarize(results)

    assert summary["tasks"] == 1
    assert 0.0 <= summary["mean_score"] <= 1.0
    assert results[0].best_score == 1.0


def test_routed_agent_smoke_pipeline_runs(tmp_path) -> None:
    tasks = [
        CodeTask(
            task_id="t/add",
            prompt="Write add(a, b)",
            entry_point="add",
            tests=("assert add(2, 3) == 5",),
        )
    ]
    config = load_experiment_config(tmp_path.cwd() / "configs/agents_2xh200.json")
    samplers = {agent.agent_id: MockSampler() for agent in config.agents}
    prompts = {agent.agent_id: agent.system_prompt for agent in config.agents}

    results = run_routed_agents(
        tasks,
        samplers_by_agent=samplers,
        prompts_by_agent=prompts,
        router=lambda task: route_task(task, config.agents),
        rollouts_per_agent=2,
        seed=0,
        timeout_s=2.0,
    )

    assert results[0].k == 2
    assert {rollout.agent_id for rollout in results[0].rollouts} == {"kernel_author"}
    assert "router_reason" in results[0].rollouts[0].completion.meta
    assert results[0].best_score == 1.0


def test_agent_worker_outputs_merge(tmp_path) -> None:
    tasks = [
        CodeTask(
            task_id="t/add",
            prompt="Write add(a, b)",
            entry_point="add",
            tests=("assert add(2, 3) == 5",),
        )
    ]
    path0 = tmp_path / "agent_0.jsonl"
    path1 = tmp_path / "agent_1.jsonl"

    write_jsonl(path0, run_agent(tasks, MockSampler(), "agent_0", k=2, seed=0, timeout_s=2.0))
    write_jsonl(path1, run_agent(tasks, MockSampler(), "agent_1", k=2, seed=1, timeout_s=2.0))
    merged = merge_task_results([path0, path1])

    assert merged[0]["task_id"] == "t/add"
    assert merged[0]["k"] == 4
    assert {r["agent_id"] for r in merged[0]["rollouts"]} == {"agent_0", "agent_1"}


def test_pre_gpu_smoke_runs(tmp_path) -> None:
    config = load_experiment_config(tmp_path.cwd() / "configs/agents_2xh200.json")
    report = run_pre_gpu_smoke(config, out_dir=tmp_path / "preflight", k=1)

    assert report["agents"] == 4
    assert report["merged_summary"]["tasks"] == 3
    assert (tmp_path / "preflight" / "report.json").exists()


def test_contribution_scoring_marks_unique_valid_rollouts() -> None:
    rows = [
        {
            "task_id": "t/add",
            "rollouts": [
                {
                    "agent_id": "agent_0",
                    "completion": {"text": "def add(a, b):\n    return a + b\n", "meta": {}, "seed": 0},
                    "eval": {"valid": True, "score": 1.0},
                },
                {
                    "agent_id": "agent_1",
                    "completion": {"text": "def add(a, b):\n    return a + b\n", "meta": {}, "seed": 1},
                    "eval": {"valid": True, "score": 1.0},
                },
            ],
        }
    ]

    scored = score_contributions(rows)
    scores = [r["completion"]["meta"]["contribution_score"] for r in scored[0]["rollouts"]]

    assert scores == [1.0, 0.2]


def test_kernelbench_mock_improvement(tmp_path) -> None:
    baseline = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    baseline.write_text("# baseline kernel\n", encoding="utf-8")
    candidate.write_text("# fast efficient kernel\n", encoding="utf-8")

    results = run_kernelbench_eval(
        [KernelTask(task_id="kb/mock", baseline_path=str(baseline), candidate_path=str(candidate))],
        benchmark_cmd=None,
        zeus_enabled=True,
        timeout_s=2.0,
        allow_mock=True,
    )

    assert results[0].correct is True
    assert results[0].improved is True
    assert results[0].accepted is True
    assert results[0].score > 1.0


def test_kernelbench_pairwise_compare_command(tmp_path) -> None:
    baseline = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    script = tmp_path / "bench.py"
    baseline.write_text("# baseline kernel\n", encoding="utf-8")
    candidate.write_text("# candidate kernel\n", encoding="utf-8")
    script.write_text(
        "import json\n"
        "print(json.dumps({\n"
        "  'baseline_correct': True,\n"
        "  'candidate_correct': True,\n"
        "  'baseline_latency_ms': 2.0,\n"
        "  'candidate_latency_ms': 1.0,\n"
        "  'baseline_energy_j': 4.0,\n"
        "  'candidate_energy_j': 2.0,\n"
        "}))\n",
        encoding="utf-8",
    )

    results = run_kernelbench_eval(
        [KernelTask(task_id="kb/pair", baseline_path=str(baseline), candidate_path=str(candidate))],
        benchmark_cmd=None,
        zeus_enabled=True,
        timeout_s=2.0,
        compare_cmd=f"python {script} --baseline {{baseline}} --candidate {{candidate}}",
    )

    assert results[0].correct is True
    assert results[0].accepted is True
    assert results[0].energy_delta_j == -2.0
    assert results[0].latency_delta_ms == -1.0
    assert results[0].reward == 1.5
