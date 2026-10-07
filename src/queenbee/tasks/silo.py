"""The Silo-Bench task.

Every hook is ``None``: each dispatch point runs its built-in Silo-Bench
code (the 18 development / 12 sealed TEST templates of
:mod:`queenbee.evo.common`, the rungs ``o5`` / ``x5`` / ``o10``, the
``all_agents`` goal, the seed program (v1), the Silo-Bench scorer and
diagnosis cards, the planner texts of :mod:`queenbee.evo.prompt` and the API
card of :mod:`queenbee.evo.api_card`).
"""

from __future__ import annotations

from queenbee.tasks.base import TaskSpec

SILO_TASK = TaskSpec(name="silo")

__all__ = ["SILO_TASK"]
