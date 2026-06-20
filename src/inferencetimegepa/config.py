from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    role: str
    prompt_path: str
    system_prompt: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    tasks: Path
    rollouts_per_agent: int
    agents: tuple[AgentConfig, ...]


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    agents = tuple(
        AgentConfig(
            agent_id=str(agent["agent_id"]),
            role=str(agent.get("role", agent["agent_id"])),
            prompt_path=str(agent.get("prompt_path", "")),
            system_prompt=str(agent["system_prompt"]),
        )
        for agent in raw["agents"]
    )
    config = ExperimentConfig(
        name=str(raw["name"]),
        model=str(raw["model"]),
        tasks=_resolve_path(path.parent, Path(raw["tasks"])),
        rollouts_per_agent=int(raw["rollouts_per_agent"]),
        agents=agents,
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    if not config.agents:
        raise ValueError("config must contain at least one agent")
    if config.rollouts_per_agent < 1:
        raise ValueError("rollouts_per_agent must be >= 1")
    if len({a.agent_id for a in config.agents}) != len(config.agents):
        raise ValueError("agent_id values must be unique")
    if not config.tasks.exists():
        raise ValueError(f"task file does not exist: {config.tasks}")


def _resolve_path(config_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_relative = Path.cwd() / path
    if repo_relative.exists():
        return repo_relative
    return config_dir / path
