"""Per-case runtime budgets of a team program (rounds, worker calls, tokens, messages)."""

from __future__ import annotations


class PythonRunBudgets:
    """Per-case runtime budgets, stated to the planner on the API card and
    enforced by the sandbox ledger: exceeding any of them raises BudgetError,
    which fails the whole run (S = 0, failure class ``budget``)."""

    def __init__(
        self,
        *,
        max_rounds: int = 4,
        max_model_calls: int = 32,
        max_completion_tokens: int = 20_000,
        max_messages: int = 64,
    ) -> None:
        self.max_rounds = int(max_rounds)
        self.max_model_calls = int(max_model_calls)
        self.max_completion_tokens = int(max_completion_tokens)
        self.max_messages = int(max_messages)

    @classmethod
    def for_rounds(
        cls, max_rounds: int, *, n_agents: int
    ) -> "PythonRunBudgets":
        """Scale the call/token/message ceilings so that the round budget is
        actually usable: a dense mesh (every agent messaging every other
        agent) in every round but the last must still fit."""

        rounds = max(1, int(max_rounds))
        dense_messages = n_agents * (n_agents - 1) * max(1, rounds - 1)
        return cls(
            max_rounds=rounds,
            max_model_calls=max(32, n_agents * (rounds + 1) + 16),
            # 5k tokens per round plus 24k per agent: a reasoning worker can
            # spend about 24k completion tokens on its submit call alone, and
            # a run over the total fails as a whole (BudgetError), so the
            # headroom grows with the team size.
            max_completion_tokens=max(
                20_000, 5_000 * rounds + 24_000 * n_agents
            ),
            max_messages=max(64, dense_messages + 32),
        )
