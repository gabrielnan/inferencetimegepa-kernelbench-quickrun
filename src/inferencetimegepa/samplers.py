from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from inferencetimegepa.tasks import CodeTask


@dataclass(frozen=True)
class Completion:
    text: str
    seed: int
    meta: dict[str, str | int | float]


class Sampler(ABC):
    @abstractmethod
    def sample(self, task: CodeTask, k: int, seed: int, system_prompt: str | None = None) -> list[Completion]:
        raise NotImplementedError


class MockSampler(Sampler):
    def sample(self, task: CodeTask, k: int, seed: int, system_prompt: str | None = None) -> list[Completion]:
        rng = random.Random(seed)
        candidates = _known_solutions(task.entry_point)
        out: list[Completion] = []
        for i in range(k):
            text = candidates[0] if i == 0 else rng.choice(candidates)
            meta: dict[str, str | int | float] = {"sampler": "mock"}
            if system_prompt:
                meta["system_prompt"] = system_prompt
            out.append(Completion(text=text, seed=seed + i, meta=meta))
        return out


class OpenAICompatibleSampler(Sampler):
    def __init__(
        self,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install with `python -m pip install -e '.[openai]'`") from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        seed: int,
        meta: dict[str, str | int | float] | None = None,
    ) -> Completion:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        return Completion(
            text=response.choices[0].message.content or "",
            seed=seed,
            meta={"sampler": "openai", "model": self.model, **(meta or {})},
        )

    def sample(self, task: CodeTask, k: int, seed: int, system_prompt: str | None = None) -> list[Completion]:
        prompt = (
            "Return only Python code. Do not include explanation.\n\n"
            f"{task.prompt}\n\n"
            f"The function entry point must be `{task.entry_point}`."
        )
        out: list[Completion] = []
        for i in range(k):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            meta: dict[str, str | int | float] = {"sampler": "openai", "model": self.model}
            if system_prompt:
                meta["system_prompt"] = system_prompt
            out.append(self.complete(messages=messages, seed=seed + i, meta=meta))
        return out


def build_sampler(
    name: str,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 512,
    base_url: str | None = None,
) -> Sampler:
    if name == "mock":
        return MockSampler()
    if name == "openai":
        if not model:
            raise ValueError("--model is required for --sampler openai")
        return OpenAICompatibleSampler(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
    raise ValueError(f"unknown sampler {name!r}")


def _known_solutions(entry_point: str) -> list[str]:
    correct = {
        "add": "def add(a, b):\n    return a + b\n",
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]\n",
        "fib": (
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
    }
    fallback = f"def {entry_point}(*args, **kwargs):\n    return None\n"
    wrong = f"def {entry_point}(*args, **kwargs):\n    return 0\n"
    return [correct.get(entry_point, fallback), wrong, fallback]
