from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from inferencetimegepa.code_extract import extract_python_code
from inferencetimegepa.evaluator import EvalResult, evaluate_completion
from inferencetimegepa.samplers import Completion, MockSampler, OpenAICompatibleSampler, Sampler
from inferencetimegepa.tasks import CodeTask


COLLABORATION_SYSTEM_PROMPT = (
    "You are one peer in a small group solving a Python programming task. Communicate concise, "
    "useful reasoning to the other peers. Do not assume a fixed role unless the group context makes "
    "one useful. Do not speak as a standalone assistant to a user; speak as your assigned agent id "
    "to the other agents."
)

FINAL_CODE_SYSTEM_PROMPT = (
    "You are a code-generation policy. Return only Python code that satisfies the requested "
    "function contract. Do not include explanation or markdown."
)


@dataclass(frozen=True)
class AgentEndpoint:
    agent_id: str
    sampler: Sampler
    system_prompt: str = FINAL_CODE_SYSTEM_PROMPT


@dataclass(frozen=True)
class CommunicatedTaskResult:
    task_id: str
    call_budget: int
    final_agent_id: str | None
    final_completion: Completion | None
    eval: EvalResult
    messages: tuple[dict, ...]
    submission_count: int


def run_communicating_agents(
    tasks: list[CodeTask],
    agents: list[AgentEndpoint],
    committer: AgentEndpoint,
    seed: int,
    timeout_s: float,
    comm_steps: int = 4,
) -> list[CommunicatedTaskResult]:
    results: list[CommunicatedTaskResult] = []
    for task_idx, task in enumerate(tasks):
        base_seed = seed + task_idx * 100_000
        router_agent_id = committer.agent_id if committer.agent_id == "router" else None
        messages = _collaborate(task, agents, comm_steps, base_seed, router_agent_id=router_agent_id)

        if committer.agent_id == "router":
            final = _router_commit(committer, task, messages, base_seed + 50_000)
            eval_result = evaluate_completion(task, final.text, timeout_s=timeout_s)
            final_agent_id: str | None = committer.agent_id
            submission_count = 1
        else:
            submissions = [message for message in messages if _looks_like_final_code(task, message["text"])]
            final_agent_id, final, eval_result = _evaluate_autonomous_submission(
                task=task,
                submissions=submissions,
                timeout_s=timeout_s,
            )
            submission_count = len(submissions)

        results.append(
            CommunicatedTaskResult(
                task_id=task.task_id,
                call_budget=len(messages) + (1 if committer.agent_id == "router" else 0),
                final_agent_id=final_agent_id,
                final_completion=final,
                eval=eval_result,
                messages=tuple(messages),
                submission_count=submission_count,
            )
        )
    return results


def write_communicated_jsonl(path: Path, results: list[CommunicatedTaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_result_to_dict(result), sort_keys=True) + "\n")


def summarize_communicated(results: list[CommunicatedTaskResult]) -> dict[str, float | int]:
    if not results:
        return {"tasks": 0, "pass_rate": 0.0, "mean_score": 0.0, "call_budget_per_task": 0}
    return {
        "tasks": len(results),
        "pass_rate": sum(r.eval.valid for r in results) / len(results),
        "mean_score": sum(r.eval.score for r in results) / len(results),
        "call_budget_per_task": results[0].call_budget,
    }


def build_mock_agents(n: int) -> list[AgentEndpoint]:
    return [AgentEndpoint(agent_id=f"agent_{idx}", sampler=MockSampler()) for idx in range(n)]


def build_openai_agents(model: str, ports: list[int], temperature: float, max_tokens: int) -> list[AgentEndpoint]:
    return [
        AgentEndpoint(
            agent_id=f"agent_{idx}",
            sampler=OpenAICompatibleSampler(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=f"http://127.0.0.1:{port}/v1",
            ),
        )
        for idx, port in enumerate(ports)
    ]


def _collaborate(
    task: CodeTask,
    agents: list[AgentEndpoint],
    comm_steps: int,
    seed: int,
    router_agent_id: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    for step_idx in range(comm_steps):
        for agent_idx, agent in enumerate(agents):
            completion = _contribute(
                agent,
                agents,
                task,
                messages,
                step_idx,
                comm_steps,
                seed + step_idx * 1_000 + agent_idx,
                router_agent_id=router_agent_id,
            )
            messages.append(_completion_row(agent.agent_id, completion, step_idx))
    return messages


def _contribute(
    agent: AgentEndpoint,
    agents: list[AgentEndpoint],
    task: CodeTask,
    transcript: list[dict],
    step_idx: int,
    comm_steps: int,
    seed: int,
    router_agent_id: str | None = None,
) -> Completion:
    transcript_text = _format_transcript(transcript, max_chars=3500) if transcript else "No peer messages yet."
    peer_ids = ", ".join(peer.agent_id for peer in agents)
    messages = [
        {"role": "system", "content": COLLABORATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"You are `{agent.agent_id}`. Known peer ids: {peer_ids}.\n\n"
                + f"{task.prompt}\n\n"
                + f"Entry point: `{task.entry_point}`\n"
                + f"Communication step: {step_idx + 1} of {comm_steps}\n\n"
                + "Peer transcript so far:\n"
                + f"{transcript_text}\n\n"
                + _collaboration_instruction(router_agent_id)
            ),
        },
    ]
    return _complete(agent, messages, seed, {"phase": "collaboration", "agent_id": agent.agent_id, "step": step_idx})


def _collaboration_instruction(router_agent_id: str | None) -> str:
    if router_agent_id:
        return (
            f"`{router_agent_id}` is the master router/finalizer. Coordinate with the other subagents, "
            "but keep your work in service of the router's synthesis: report edge cases, implementation "
            "ideas, risks, checks, and compact state when the transcript gets long. Do not submit final "
            "code yourself. Return a concise peer message, preferably starting with `REPORT:`."
        )
    return (
        "Add your next useful peer message. Coordinate however seems useful: strategy, edge cases, "
        "assumptions, checks, division of work, readiness, or final submission. If the transcript "
        "is getting long, compact the useful state instead of repeating prior messages: agreed "
        "approach, edge cases, unresolved issues, and any emergent submitter/readiness signal.\n\n"
        "The group is rewarded when one peer submits a correct final implementation for everyone. "
        "If another peer already appears to have submitted final code, avoid duplicate submissions. "
        "If you decide the group is ready and you should submit, return only Python code. Otherwise, "
        "return a concise peer message."
    )


def _evaluate_autonomous_submission(
    task: CodeTask,
    submissions: list[dict],
    timeout_s: float,
) -> tuple[str | None, Completion | None, EvalResult]:
    if len(submissions) != 1:
        error = "no_final_submission" if not submissions else "multiple_final_submissions"
        return None, None, EvalResult(
            score=0.0,
            valid=False,
            tests_passed=0,
            total_tests=len(task.tests),
            elapsed_s=0.0,
            error=error,
        )
    submission = submissions[0]
    completion = Completion(
        text=submission["text"],
        seed=submission["seed"],
        meta={**submission["meta"], "phase": "autonomous_submit"},
    )
    return submission["agent_id"], completion, evaluate_completion(task, completion.text, timeout_s=timeout_s)


def _router_commit(agent: AgentEndpoint, task: CodeTask, transcript: list[dict], seed: int) -> Completion:
    messages = [
        {"role": "system", "content": agent.system_prompt},
        {
            "role": "user",
            "content": (
                f"{task.prompt}\n\n"
                "The following agents communicated while working together on the task. Use their shared "
                f"work to commit one final implementation for `{task.entry_point}`.\n\n"
                f"{_format_transcript(transcript, max_chars=5000)}\n\nReturn only the final Python code."
            ),
        },
    ]
    return _complete(agent, messages, seed, {"phase": "router_commit", "agent_id": agent.agent_id})


def _looks_like_final_code(task: CodeTask, text: str) -> bool:
    code = extract_python_code(text)
    return bool(re.search(rf"^\s*def\s+{re.escape(task.entry_point)}\s*\(", code, re.MULTILINE))


def _completion_row(agent_id: str, completion: Completion, step_idx: int) -> dict:
    return {
        "agent_id": agent_id,
        "step": step_idx,
        "seed": completion.seed,
        "text": completion.text,
        "meta": completion.meta,
    }


def _format_transcript(messages: list[dict], max_chars: int | None = None) -> str:
    text = "\n\n".join(f"{message['agent_id']}:\n{message['text']}" for message in messages)
    if max_chars is None or len(text) <= max_chars:
        return text
    return "[earlier transcript omitted]\n\n" + text[-max_chars:]


def _complete(agent: AgentEndpoint, messages: list[dict[str, str]], seed: int, meta: dict) -> Completion:
    if isinstance(agent.sampler, OpenAICompatibleSampler):
        return agent.sampler.complete(messages=messages, seed=seed, meta=meta)
    text = agent.sampler.sample(
        CodeTask(task_id="mock", prompt="", entry_point="add", tests=()),
        k=1,
        seed=seed,
        system_prompt=agent.system_prompt,
    )[0].text
    return Completion(text=text, seed=seed, meta={"sampler": "mock", **meta})


def _result_to_dict(result: CommunicatedTaskResult) -> dict:
    return {
        "task_id": result.task_id,
        "call_budget": result.call_budget,
        "final_agent_id": result.final_agent_id,
        "final_completion": asdict(result.final_completion) if result.final_completion else None,
        "eval": asdict(result.eval),
        "messages": list(result.messages),
        "submission_count": result.submission_count,
    }
