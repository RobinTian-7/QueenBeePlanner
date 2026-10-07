"""Task registry of QueenBee-Evo.

The active task decides every task-specific step of the loop, the racer,
the screen, the planner prompt and the evaluator (see
:class:`~queenbee.tasks.base.TaskSpec`).  The default is the built-in
Silo-Bench task ``"silo"``; the loop and the evaluator command lines
(:mod:`queenbee.evo.loop`, :mod:`queenbee.evaluate`) activate the task named
by ``--task`` once, before anything else runs::

    register_task(task_spec)                         # a TaskSpec
    register_task("name", "package.module:ATTR")     # imported on first use
    set_task("name")                                 # activate for the process
    get_task()                                       # the active TaskSpec
    with use_task("name"): ...                       # temporarily (tests)

Registering a task changes nothing until it is activated.  Shipped tasks:
``silo`` (Silo-Bench, the default) and ``cf`` (Count-Frequency,
:mod:`queenbee.tasks.cf`, imported when it is first resolved).
"""

from __future__ import annotations

import argparse
import importlib
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from queenbee.tasks.base import GOALS, TaskSpec
from queenbee.tasks.silo import SILO_TASK

DEFAULT_TASK = "silo"

_LOCK = threading.RLock()
_REGISTRY: dict[str, "TaskSpec | str | Callable[[], TaskSpec]"] = {
    DEFAULT_TASK: SILO_TASK,
    "cf": "queenbee.tasks.cf:CF_TASK",
}
_ACTIVE: TaskSpec = SILO_TASK


def register_task(task: TaskSpec | str, target: str | Callable[[], TaskSpec] | None = None, *,
                  replace: bool = False) -> None:
    """Register a :class:`TaskSpec`, or a name with a lazy ``target``
    (``"package.module:ATTR"`` or a zero-argument callable returning the
    TaskSpec), resolved on first use.  A different entry under a taken name
    raises unless ``replace`` (a lazy entry may always be replaced by a
    TaskSpec; registering the same entry again is a no-op); the built-in
    ``"silo"`` is never replaced."""

    if isinstance(task, TaskSpec):
        if target is not None:
            raise TypeError("register_task(spec) takes no target")
        name, entry = task.name, task
    else:
        name, entry = str(task), target
        if entry is None or not (isinstance(entry, str) or callable(entry)):
            raise TypeError("register_task(name, target): target is 'module:attr' or a callable")
        if isinstance(entry, str) and ":" not in entry:
            raise ValueError(f"task {name!r}: lazy target {entry!r} is not 'module:attr'")
    with _LOCK:
        if name == DEFAULT_TASK:
            raise ValueError(f"the built-in task {DEFAULT_TASK!r} cannot be replaced")
        old = _REGISTRY.get(name)
        lazy_to_spec = old is not None and not isinstance(old, TaskSpec) and isinstance(entry, TaskSpec)
        if old is not None and old != entry and not (replace or lazy_to_spec):
            raise ValueError(f"task {name!r} is already registered")
        _REGISTRY[name] = entry


def available_tasks() -> tuple[str, ...]:
    """Registered task names, the default first."""

    with _LOCK:
        return (DEFAULT_TASK, *sorted(n for n in _REGISTRY if n != DEFAULT_TASK))


def resolve_task(task: TaskSpec | str) -> TaskSpec:
    """The :class:`TaskSpec` of a registered name (a TaskSpec is returned as
    is); a lazy entry is resolved (imported or called) and cached on first
    use."""

    if isinstance(task, TaskSpec):
        return task
    name = str(task)
    with _LOCK:
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"unknown task {name!r} (registered: {', '.join(available_tasks())})")
        if isinstance(entry, TaskSpec):
            return entry
        if isinstance(entry, str):
            module, _, attr = entry.partition(":")
            spec = getattr(importlib.import_module(module), attr)
        else:
            spec = entry()
        if not isinstance(spec, TaskSpec) or spec.name != name:
            raise TypeError(f"task {name!r}: {entry!r} did not give a TaskSpec named {name!r}")
        _REGISTRY[name] = spec
        return spec


def get_task() -> TaskSpec:
    """The active task (``"silo"`` unless :func:`set_task` chose another)."""

    return _ACTIVE


def set_task(task: TaskSpec | str) -> TaskSpec:
    """Activate a task for the process (:func:`parse_task_args` calls it
    once, before anything runs); returns its TaskSpec.  A TaskSpec whose
    name is registered for a different TaskSpec is refused."""

    global _ACTIVE
    spec = resolve_task(task)
    if isinstance(task, TaskSpec):
        with _LOCK:
            known = spec.name in _REGISTRY
        if known and resolve_task(spec.name) is not spec:
            raise ValueError(f"another task is registered as {spec.name!r}")
    with _LOCK:
        _ACTIVE = spec
    return spec


@contextmanager
def use_task(task: TaskSpec | str) -> Iterator[TaskSpec]:
    """Activate a task inside a ``with`` block and restore the previous one
    afterwards (tests; the active task is process-wide)."""

    global _ACTIVE
    with _LOCK:
        previous = _ACTIVE
    spec = set_task(task)
    try:
        yield spec
    finally:
        with _LOCK:
            _ACTIVE = previous


# --------------------------------------------------------------------------- #
# Command-line helpers
# --------------------------------------------------------------------------- #


def add_task_argument(parser: argparse.ArgumentParser) -> None:
    """``--task NAME`` (default ``silo``; the registered names)."""

    parser.add_argument(
        "--task", default=DEFAULT_TASK, choices=available_tasks(),
        help="task family (default %(default)s); its defaults apply to the options "
             "not given on the command line",
    )


def task_from_argv(argv: Sequence[str] | None) -> str:
    """The ``--task`` value of ``argv`` (``sys.argv[1:]`` when None), else
    the default task."""

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", default=DEFAULT_TASK)
    known, _rest = pre.parse_known_args(None if argv is None else list(argv))
    return str(known.task)


def apply_cli_defaults(parser: argparse.ArgumentParser, task: TaskSpec,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make the task's ``cli_defaults`` (and ``extra``; the task's value wins
    on a shared dest) the defaults of the parser's options with those dests
    (unknown dests are ignored; an option given on the command line still
    wins; a required option with a default becomes optional).  Returns the
    defaults applied."""

    wanted = dict(extra or {}) | dict(task.cli_defaults or {})
    applied: dict[str, Any] = {}
    for action in parser._actions:
        if action.dest in wanted:
            action.default = wanted[action.dest]
            action.required = False
            applied[action.dest] = action.default
    return applied


def parse_task_args(parser: argparse.ArgumentParser, argv: Sequence[str] | None, *,
                    extra: Callable[[TaskSpec], dict[str, Any]] | None = None
                    ) -> tuple[argparse.Namespace, TaskSpec]:
    """Activate the task named by ``--task``, apply its command-line defaults
    (plus ``extra(task)``), then parse ``argv``."""

    name = task_from_argv(argv)
    try:
        task = set_task(name)
    except ValueError as exc:
        parser.error(str(exc))
    apply_cli_defaults(parser, task, extra(task) if extra is not None else None)
    args = parser.parse_args(None if argv is None else list(argv))
    return args, task


__all__ = [
    "DEFAULT_TASK",
    "GOALS",
    "SILO_TASK",
    "TaskSpec",
    "add_task_argument",
    "apply_cli_defaults",
    "available_tasks",
    "get_task",
    "parse_task_args",
    "register_task",
    "resolve_task",
    "set_task",
    "task_from_argv",
    "use_task",
]
