"""Suite-wide isolation: the process-wide diagnosis-card options are
restored after every test, and a test that leaves a task other than
Silo-Bench active fails (the default task is restored first)."""

from __future__ import annotations

import os

import pytest

import queenbee.tasks as tasks
from queenbee.evo.credit import MULTISET_ENV
from queenbee.evo.diagnosis import ATTEMPT_ENV, LOCAL_ONLY_ENV

#: Diagnosis-card options (environment variables) that an EvoRun sets for
#: the whole process.
_CARD_OPTIONS = (LOCAL_ONLY_ENV, ATTEMPT_ENV, MULTISET_ENV)


@pytest.fixture(autouse=True)
def _task_and_card_options_restored():
    saved = {name: os.environ.get(name) for name in _CARD_OPTIONS}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    leaked = tasks.get_task()
    if leaked is not tasks.SILO_TASK:
        tasks.set_task(tasks.DEFAULT_TASK)
        pytest.fail(f"test left task {leaked.name!r} active")
