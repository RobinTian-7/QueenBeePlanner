"""Leakage audit for model-visible prompts.

Every worker prompt built from a Silo-Bench instance (the communication and
submit prompts of :class:`queenbee.bench.silo_protocol.SiloProtocolAdapter`)
passes this audit before it can be sent. It forbids (case-insensitively) the
tokens that would leak the answer key, the benchmark's annotated optimal
topology, or a copyable hand-designed topology into model context:

``expected_output``/``expected_outputs``/``answer_key``/``ground_truth``,
``optimal_topology``/``optimal_message_count``, the benchmark's
``Communication Protocol`` section, and the vocabulary of hand-designed
topology constructions ``one_peer``/``exponential``/``distance-doubling``/``pow2``.

The task statements, once the loader has removed their Communication
Protocol section (:func:`queenbee.bench.silo_bench.sanitize_task_description`),
contain none of these tokens, so an audit failure always means the prompt
pipeline re-introduced a leak: the audit raises and the prompt is never sent.
"""

from __future__ import annotations

# One shared, case-insensitive forbidden-token list for both tests and the
# runtime audit, so "tests pass" and "sent prompts are clean" are the same claim.
FORBIDDEN_PROMPT_TOKENS: tuple[str, ...] = (
    "expected_output",
    "expected_outputs",
    "answer_key",
    "ground_truth",
    "optimal_topology",
    "optimal_message_count",
    "communication protocol",
    "one_peer",
    "distance-doubling",
    "pow2",
    "exponential",
)


class PromptLeakageError(ValueError):
    """A model-visible prompt contains a forbidden leakage token."""


def find_leakage_tokens(
    prompt: str,
    *,
    allowed_tokens: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """The forbidden tokens found in ``prompt`` (an empty list = clean);
    tokens listed in ``allowed_tokens`` are not reported."""
    lowered = prompt.lower()
    allowed = {str(token).lower() for token in allowed_tokens}
    return [
        token
        for token in FORBIDDEN_PROMPT_TOKENS
        if token in lowered and token not in allowed
    ]


def assert_prompt_clean(
    prompt: str,
    *,
    context: str = "prompt",
    allowed_tokens: tuple[str, ...] | list[str] = (),
) -> str:
    """Raise :class:`PromptLeakageError` if the prompt contains a leak.

    Returns the prompt unchanged so call sites can wrap prompt construction:
    ``client.complete(assert_prompt_clean(build_prompt(...)))``.
    """
    hits = find_leakage_tokens(prompt, allowed_tokens=allowed_tokens)
    if hits:
        raise PromptLeakageError(
            f"leakage audit failed for {context}: forbidden tokens {hits}"
        )
    return prompt
