"""LLM client interface and usage accounting.

``LLMClient`` is the minimal synchronous protocol every client implements.
``LLMUsage`` and ``LLMResponse`` carry the token and call counts of a
completion; ``estimate_tokens`` is the fallback count when a provider reports
no usage.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Token/call accounting for one model invocation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def combine_usage(usages: list[LLMUsage]) -> LLMUsage:
    """Combine usage from an initial call and any retry calls (field-wise sums)."""
    return LLMUsage(
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        model_calls=sum(usage.model_calls for usage in usages),
    )


class LLMResponse(BaseModel):
    """Raw text plus usage from an LLM client.

    ``raw_responses`` / ``raw_prompts`` optionally keep the raw provider
    replies and the prompts that were sent.
    """

    text: str
    usage: LLMUsage = LLMUsage()
    raw_responses: list[str] = Field(default_factory=list)
    raw_prompts: list[str] = Field(default_factory=list)


class LLMClient(Protocol):
    """Minimal synchronous LLM interface."""

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> LLMResponse:
        """Return a completion for a prompt.

        With ``json_mode`` (the default) the provider is constrained to emit a
        single JSON object. Pass ``json_mode=False`` for free-form text, e.g.
        planner calls that return Python source and the worker calls of a
        team program.
        """
        ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (whitespace-separated words, at least 1) for
    accounting when the provider reports no usage."""
    return max(1, len(text.split()))
