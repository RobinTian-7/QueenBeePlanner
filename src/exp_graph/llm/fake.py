"""Deterministic fake LLM client for tests and offline smoke runs."""

from __future__ import annotations

from exp_graph.llm.base import LLMResponse, LLMUsage, estimate_tokens


class FakeLLMClient:
    """Deterministic, zero-cost client that never calls a model.

    It does not solve any task: a JSON-mode call returns an empty JSON object
    and a free-text call returns an empty string, mirroring what
    ``OpenAIChatClient`` returns for an empty completion. Usage is estimated
    from the prompt and the returned text.
    """

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> LLMResponse:
        del model_name, temperature
        text = "{}" if json_mode else ""
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(text),
            ),
        )
