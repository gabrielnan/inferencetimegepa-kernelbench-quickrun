from __future__ import annotations

from dataclasses import dataclass

from inferencetimegepa.config import AgentConfig
from inferencetimegepa.tasks import CodeTask


@dataclass(frozen=True)
class RouteDecision:
    agent_id: str
    reason: str


def route_task(task: CodeTask, agents: tuple[AgentConfig, ...]) -> RouteDecision:
    if not agents:
        raise ValueError("cannot route without agents")

    text = f"{task.task_id} {task.prompt} {task.entry_point}".lower()
    if any(word in text for word in ("edge", "empty", "negative", "correct", "test")):
        return _find_agent(agents, "correctness_reviewer", "correctness and edge-case task")
    if any(word in text for word in ("bench", "latency", "energy", "speedup", "profile")):
        return _find_agent(agents, "benchmark_engineer", "benchmark and measurement task")
    if any(word in text for word in ("prompt", "trace", "reflect", "gepa", "frontier")):
        return _find_agent(agents, "gepa_reflector", "prompt reflection task")
    return _find_agent(agents, "kernel_author", "default kernel implementation route")


def _find_agent(agents: tuple[AgentConfig, ...], preferred_id: str, reason: str) -> RouteDecision:
    for agent in agents:
        if agent.agent_id == preferred_id:
            return RouteDecision(agent_id=agent.agent_id, reason=reason)
    return RouteDecision(agent_id=agents[0].agent_id, reason=f"{reason}; preferred agent unavailable")
