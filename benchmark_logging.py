#!/usr/bin/env python3
"""
benchmark_logging.py — multiprocess-safe logging for the benchmark harness.

Single reason to exist: make one ``logging`` configuration work correctly across
a ``ProcessPoolExecutor``.  This module knows nothing about the simulation; it is
pure plumbing, importable and testable in isolation.

Architecture (the one genuinely hard part)
------------------------------------------

``run_benchmark()`` fans out one worker process per government.  Eight processes
cannot share a ``logging.FileHandler``: each would hold its own buffered file
object over the same path, records would interleave mid-line, and the file would
be corrupted in exactly the situation you most need it (a long unattended run).

    worker process (xN)                       main process
    +--------------------------+              +---------------------------------+
    | logging.getLogger("sim") |              | QueueListener (background thread)|
    |        |                 |   mp.Queue   |        |                        |
    |        v                 |------------->|        +--> FileHandler         |
    |  QueueHandler            |              |        +--> stdout StreamHandler|
    |   + ContextFilter        |              |  parent records go straight to  |
    +--------------------------+              |  the same two handler objects   |
                                              +---------------------------------+

Why ``spawn``, not ``fork``
---------------------------

Every child process this harness creates is started with the **spawn** start
method (:data:`WORKER_START_METHOD` / :func:`worker_mp_context`).  This is a
correctness requirement, not a preference.

``fork`` copies the parent's *memory* but only the *calling thread*.  Locks that
were held by some other thread at the instant of the fork are inherited in a
locked state, by a child that contains no thread able to release them.  The
``QueueListener`` runs on exactly such a thread, and it spends its life inside
``FileHandler.emit() -> flush()``, holding the C-level ``_io.BufferedWriter``
lock of the log file.  A worker forked during that window inherits a permanently
locked buffer and hangs the first time anything touches it.

``logging`` is fork-aware, but only for its own locks:
``os.register_at_fork(after_in_child=...)`` reinitialises ``Handler.lock`` and
the module ``_lock``.  It cannot reach inside the C stream object, so no amount
of care on the ``logging`` side fixes this.

The window is not limited to the log file.  Under the documented production
invocation::

    nohup python3 run_full_simulation.py > full_run_console.log 2>&1 &

``sys.stdout`` is a *block-buffered* file rather than a line-buffered tty, and
the listener writes the console stream too.  A forked child that inherits that
buffer mid-write deadlocks at interpreter teardown, in
``BaseProcess._bootstrap``'s ``_flush_std_streams()`` — before any of our code
gets a say.  There is therefore no handler-level patch that closes the whole
hole; only not forking does.

``spawn`` starts a fresh interpreter, so a child inherits neither the parent's
threads nor its buffers, and the entire failure class disappears.  The cost is a
re-import of this package per worker (measured: a few hundred ms, against a
sweep measured in hours).

Invariants that must not be broken (each one is a real failure mode):

* **Children are started with ``spawn``.**  See above.  Reverting to ``fork``
  reintroduces a ~40%-per-sweep hang at ``--log-level DEBUG``.

* **Workers must not inherit the parent's handlers, and must never flush one.**
  Under ``spawn`` there is nothing to inherit, which is the point.
  :func:`init_worker_logging` still *strips* whatever handlers it finds and
  installs a ``QueueHandler`` instead, but it does so without calling
  ``close()`` or ``flush()`` on them: those are precisely the blocking calls
  that hang a forked child, and a defensive path is worthless if it is itself
  the landmine.

* **The queue must be picklable.**  Under ``spawn`` the queue is pickled into
  the child, which rules out a raw ``multiprocessing.Queue``.  A
  ``Manager().Queue()`` proxy pickles cleanly; at ~13k records for a full sweep
  the proxy's per-record cost is irrelevant next to being correct.

* **A logging fault must never hang the sweep.**  ``QueueHandler.emit`` reports
  through ``handleError`` rather than raising, so a dead manager drops records
  instead of blocking a ten-hour run.

* **The listener outlives every pool.**  It is started before the first pool and
  stopped only after the last one has exited, so no record is lost at shutdown.

* **The context filter lives on handlers, not loggers.**  ``Logger.handle()``
  runs only the *originating* logger's filters, so a filter on ``sim`` would
  never see a record from ``sim.ADS``.  ``Handler.handle()`` runs its filters on
  every record it receives, which is what stamps ``ctx`` onto everything.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Root of the logger hierarchy the whole simulation already uses.
ROOT_LOGGER = "sim"

#: Start method for **every** child process the harness creates — pool workers
#: and the ``Manager`` alike.  Read the "Why ``spawn``" section of the module
#: docstring before changing this; ``"fork"`` is a known, reproducible deadlock.
WORKER_START_METHOD = "spawn"

#: ``PYTHONHASHSEED`` given to every worker.  See :func:`worker_mp_context` —
#: this is a reproducibility requirement, not a micro-optimisation.
WORKER_HASH_SEED = "0"

#: Harness progress logger.  Pinned at INFO so the engine level cannot mute it.
BENCH_LOGGER = "sim.bench"

#: Hard ceiling on how many runs may request per-agent state dumps: ~1.6 MB
#: per run means an accidental ``--detailed-agents`` on the full 16,800-run
#: sweep would write ~27 GB.
DETAILED_AGENTS_MAX_RUNS = 20

#: Default level for the *engine* loggers (sim.<Gov>, sim.events).  WARNING, not
#: CRITICAL+1: engine warnings and errors now reach the file, while the
#: per-cycle / per-agent DEBUG flood stays off.
DEFAULT_ENGINE_LEVEL = "WARNING"

LOG_LEVEL_NAMES: Tuple[str, ...] = (
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
)

#: Levels at which a multi-run sweep produces an alarming number of lines.
_VERBOSE_LEVELS = frozenset({"DEBUG", "INFO"})

#: Below this estimate the warning is noise.  A guard that fires on every quick
#: test teaches the reader to ignore it, which costs more than it saves on the
#: one occasion it matters.
_VOLUME_WARNING_LINES = 100_000

_FILE_FORMAT = "%(asctime)s.%(msecs)03dZ  %(levelname)-7s [%(ctx)-18s] %(message)s"
_FILE_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_CONSOLE_FORMAT = "%(message)s"

#: Filename timestamp: ISO-8601 basic, explicitly UTC, lexicographically sortable.
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


# ---------------------------------------------------------------------------
# Multiprocessing start method
# ---------------------------------------------------------------------------

def pin_worker_hash_seed() -> None:
    """
    Fix ``PYTHONHASHSEED`` in the environment inherited by future child processes.

    A ``spawn``\\ ed child is a brand-new interpreter, so it draws its own random
    string-hash seed unless the environment pins one.  Under ``fork`` every
    worker shared the parent's single seed; under ``spawn``, N workers would draw
    N independent random seeds.  Pinning closes that semantic gap and makes the
    sweep's hash order reproducible by construction rather than by luck.

    Honest scope: no hash-order dependence has actually been *demonstrated* in
    this simulation — a known ADS reproducibility defect still reproduces with
    the seed pinned, so it is a different bug and this does not fix it.  This
    is prophylactic, and cheap enough (one environment variable) to be worth
    it in a package whose reason to exist is reproducible published results.

    An explicit ``PYTHONHASHSEED`` already in the environment is left alone: a
    caller who set it did so deliberately, quite possibly to probe exactly this.

    Only the parent's *children* are affected — a running interpreter's hash seed
    is fixed at startup and cannot be changed.  That is sufficient, because the
    parent runs no simulation: ``build_plan`` is verified hash-stable (identical
    pickle digest under four different seeds) and the workers do the rest.
    """
    os.environ.setdefault("PYTHONHASHSEED", WORKER_HASH_SEED)


def worker_mp_context() -> multiprocessing.context.BaseContext:
    """
    Return the multiprocessing context every child process must be started from.

    Passed explicitly to ``ProcessPoolExecutor(mp_context=...)`` and used for the
    session's ``Manager``, rather than relying on the process-global default.
    Correctness must not depend on a caller having remembered to call
    :func:`~benchmark_core.configure_multiprocessing` first — ``run_benchmark()``
    is a public entry point and is called directly by the test suite.

    **Side effect:** calls :func:`pin_worker_hash_seed`.  It lives here rather
    than at the call sites because this function is the single choke point
    through which every child process is created, and a reproducibility
    guarantee that depends on remembering a second call is not a guarantee.

    Raises:
        RuntimeError: if the interpreter has no ``spawn`` context.  Every
            supported CPython platform does, so this means something is deeply
            wrong; falling back to ``fork`` would silently restore the deadlock,
            which is strictly worse than refusing to start.
    """
    pin_worker_hash_seed()
    try:
        return multiprocessing.get_context(WORKER_START_METHOD)
    except ValueError as exc:                      # pragma: no cover - defensive
        raise RuntimeError(
            f"the {WORKER_START_METHOD!r} multiprocessing start method is "
            f"unavailable on this interpreter. The benchmark harness cannot run "
            f"safely without it: under 'fork', a worker created while the log "
            f"listener thread holds a stream buffer lock inherits that lock and "
            f"hangs forever."
        ) from exc


# ---------------------------------------------------------------------------
# Process-global context ("which cell am I?")
# ---------------------------------------------------------------------------
#
# One string per process, narrowed and restored around each run.  This is what
# makes an eight-way-interleaved file greppable:
#     grep '\[ads/D=56' sim_full_*.log
# extracts one cell's complete history.

_context: str = "sweep"


def set_context(ctx: str) -> None:
    """Set this process's log context tag (e.g. ``"ads/D=56/k=4"``)."""
    global _context
    _context = str(ctx)


def get_context() -> str:
    """Return this process's current log context tag."""
    return _context


class ContextFilter(logging.Filter):
    """
    Stamp ``record.ctx`` with the emitting process's context.

    Attached to *handlers*, never to loggers — see the module docstring.  A
    record that already carries ``ctx`` (i.e. one that crossed the queue from a
    worker, where the tag was applied at source) is left alone, so the parent's
    handlers never overwrite a worker's context with ``"sweep"``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "ctx", None) is None:
            record.ctx = _context
        return True


# ---------------------------------------------------------------------------
# Handlers / formatters
# ---------------------------------------------------------------------------

class _StdoutHandler(logging.StreamHandler):
    """
    A ``StreamHandler`` that resolves ``sys.stdout`` at emit time.

    ``logging.StreamHandler`` captures the stream object at construction.  That
    breaks ``contextlib.redirect_stdout`` (used by the test suite) and anything
    else that swaps ``sys.stdout`` after setup.  Resolving lazily keeps the
    console behaviour identical to the ``print()`` calls this replaces.
    """

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    @property                      # type: ignore[override]
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # noqa: D401 - deliberately ignored
        """Ignore assignment; the stream is always the *current* sys.stdout."""


class _FileFormatter(logging.Formatter):
    """
    The file formatter, with one refinement: it repeats the prefix on every
    physical line of a multi-line record.

    Several harness messages embed newlines (banner separators, blank spacers,
    tracebacks).  Without this, those continuation lines land in the file naked
    — no timestamp, no level, no context — and
    ``grep '\\[ads/D=56' sim_full_*.log`` silently misses them, which defeats
    the one property the ``[ctx]`` column exists to provide.  It also makes a
    continuation line indistinguishable from a torn write, so the regression
    test that watches for interleaving would be unable to tell the two apart.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if "\n" not in text:
            return text
        first, *rest = text.split("\n")
        # The context bracket always precedes the message, so the first "] " in
        # the formatted line terminates the prefix.
        marker = first.find("] ")
        prefix = first[: marker + 2] if marker != -1 else ""
        return "\n".join([first] + [prefix + line for line in rest])


def _file_formatter() -> logging.Formatter:
    fmt = _FileFormatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)
    fmt.converter = time.gmtime          # timestamps are UTC, matching the 'Z'
    return fmt


def _console_formatter() -> logging.Formatter:
    return logging.Formatter(_CONSOLE_FORMAT)


def level_value(name: str) -> int:
    """Translate a level name to its numeric value, rejecting unknown names."""
    upper = str(name).upper()
    if upper not in LOG_LEVEL_NAMES:
        raise ValueError(
            f"unknown log level {name!r}; expected one of {', '.join(LOG_LEVEL_NAMES)}"
        )
    return getattr(logging, upper)


# ---------------------------------------------------------------------------
# Log file naming
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def default_log_path(prefix: str, when: datetime, directory: str = ".") -> str:
    """
    Build the default log path: ``<directory>/<prefix>_<UTC timestamp>.log``.

    UTC with an explicit ``Z``, because these files are produced on a remote box,
    read in another timezone and archived next to a paper — an implicit zone is
    a timestamp you cannot correlate with anything later.
    """
    name = f"{prefix}_{when.strftime(_TIMESTAMP_FORMAT)}.log"
    return os.path.abspath(os.path.join(os.path.expanduser(directory), name))


def resolve_log_path(
    prefix: str,
    override: Optional[str],
    when: Optional[datetime] = None,
    directory: str = ".",
) -> str:
    """
    Resolve the log path from the ``--log-file`` override, if any.

    * ``None``            -> generated name in *directory* (the CWD by default).
    * an existing dir     -> generated name inside it.
    * anything else       -> used verbatim as the file path.
    """
    when = when or utc_now()
    if override is None:
        return default_log_path(prefix, when, directory)
    path = os.path.abspath(os.path.expanduser(str(override)))
    if os.path.isdir(path):
        return default_log_path(prefix, when, path)
    return path


# ---------------------------------------------------------------------------
# Parent-side session
# ---------------------------------------------------------------------------

def _start_manager() -> Any:
    """
    Start the sweep's ``Manager``, translating a missing ``__main__`` guard.

    ``spawn`` re-imports the program's ``__main__`` module in every child.  If
    that module runs the sweep at import time — i.e. it lacks an
    ``if __name__ == "__main__":`` guard — the child re-enters this function,
    ``multiprocessing`` refuses the recursive launch, and the child dies before
    it can report anything useful.  What the parent sees is a bare ``EOFError``
    from ``managers.py``: a genuinely baffling message whose cause is nowhere in
    the traceback.

    This costs nothing (it wraps the child we were starting anyway) and turns
    the sweep's very first child process into a launch check.  The shipped entry
    points are guarded; ad-hoc driver scripts are exactly where this bites.
    """
    try:
        return worker_mp_context().Manager()
    except EOFError as exc:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        where = f" ({main_file})" if main_file else ""
        raise RuntimeError(
            f"failed to start the logging manager process. The usual cause is "
            f"that the script which started this sweep{where} calls it at import "
            f"time. Worker processes are started with the "
            f"{WORKER_START_METHOD!r} method, which re-imports that script in "
            f"every child, so the call must be behind a guard:\n"
            f"\n"
            f"    if __name__ == \"__main__\":\n"
            f"        raise SystemExit(main())\n"
            f"\n"
            f"(run_full_simulation.py and run_quick_test.py already are.)"
        ) from exc

class LoggingSession:
    """
    Owns the parent process's logging configuration for one sweep.

    Use as a context manager wrapping the *entire* sweep, including every
    process pool::

        with setup_parent_logging(config) as session:
            ...
            ProcessPoolExecutor(initializer=init_worker_logging,
                                initargs=session.worker_initargs())

    On exit the listener is stopped (draining anything still queued), the
    manager process is shut down and the file handle is closed.
    """

    def __init__(
        self,
        log_path: str,
        engine_level: str = DEFAULT_ENGINE_LEVEL,
        console: bool = True,
    ) -> None:
        self.log_path = os.path.abspath(os.path.expanduser(log_path))
        self.engine_level = str(engine_level).upper()
        self._console_enabled = console
        self._queue: Any = None
        self._manager: Any = None
        self._listener: Optional[logging.handlers.QueueListener] = None
        self._file_handler: Optional[logging.Handler] = None
        self._console_handler: Optional[logging.Handler] = None
        self._saved_handlers: list = []
        self._saved_level: Optional[int] = None
        self._saved_propagate: Optional[bool] = None
        self._active = False

    # -- lifecycle ------------------------------------------------------

    def __enter__(self) -> "LoggingSession":
        level = level_value(self.engine_level)

        parent_dir = os.path.dirname(self.log_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setFormatter(_file_formatter())
        self._file_handler.addFilter(ContextFilter())

        handlers: list = [self._file_handler]
        if self._console_enabled:
            self._console_handler = _StdoutHandler()
            self._console_handler.setFormatter(_console_formatter())
            self._console_handler.addFilter(ContextFilter())
            handlers.append(self._console_handler)

        # The queue is created ONCE for the whole sweep (not per difficulty
        # level) and handed to every pool, so a worker's records survive its
        # pool being torn down at the end of a difficulty level.
        #
        # Started from the spawn context like everything else.  The manager is
        # created before the listener thread exists, so forking it would happen
        # to be safe today — but "safe because of the current statement order"
        # is exactly the reasoning that produced the deadlock this module now
        # documents.  One start method, no exceptions, nothing to re-derive.
        #
        # This is also the first child process of the sweep, which makes it the
        # natural place to diagnose a bad launch: see _start_manager.
        self._manager = _start_manager()
        self._queue = self._manager.Queue()

        # respect_handler_level=False: a worker only enqueues records that
        # already passed its logger's level check, so re-filtering here would
        # only risk dropping something the worker deliberately emitted.
        self._listener = logging.handlers.QueueListener(
            self._queue, *handlers, respect_handler_level=False
        )
        self._listener.start()

        root = logging.getLogger(ROOT_LOGGER)
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        self._saved_propagate = root.propagate
        for handler in self._saved_handlers:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        # No propagation to the root logger: nothing should be double-printed by
        # a basicConfig() some other caller may have installed.
        root.propagate = False

        # Pinned explicitly.  Python checks the *originating* logger's effective
        # level and then walks to ancestor HANDLERS without re-checking ancestor
        # levels, so `sim` sitting at WARNING does not suppress `sim.bench` at
        # INFO.  This is load-bearing and non-obvious; it has a dedicated test.
        logging.getLogger(BENCH_LOGGER).setLevel(logging.INFO)

        self._active = True
        _mark_configured(True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._active = False
        _mark_configured(False)

        # Stop the listener FIRST so everything already queued is drained to the
        # file before the handlers go away.
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:                      # pragma: no cover - defensive
                pass
            self._listener = None

        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:                      # pragma: no cover - defensive
                pass
            self._manager = None
        self._queue = None

        root = logging.getLogger(ROOT_LOGGER)
        for handler in (self._file_handler, self._console_handler):
            if handler is None:
                continue
            try:
                root.removeHandler(handler)
                handler.close()
            except Exception:                      # pragma: no cover - defensive
                pass
        self._file_handler = None
        self._console_handler = None

        for handler in self._saved_handlers:
            root.addHandler(handler)
        if self._saved_level is not None:
            root.setLevel(self._saved_level)
        if self._saved_propagate is not None:
            root.propagate = self._saved_propagate
        self._saved_handlers = []

        return False        # never suppress an exception

    # -- worker plumbing ------------------------------------------------

    def worker_initargs(self) -> Tuple[Any, str]:
        """Arguments for ``ProcessPoolExecutor(initializer=init_worker_logging)``."""
        if not self._active:
            raise RuntimeError(
                "worker_initargs() requires an active LoggingSession; "
                "call it inside the `with setup_parent_logging(...)` block"
            )
        return (self._queue, self.engine_level)


def setup_parent_logging(
    log_path: str,
    engine_level: str = DEFAULT_ENGINE_LEVEL,
    console: bool = True,
) -> LoggingSession:
    """Build (but do not start) the parent's logging session."""
    return LoggingSession(log_path, engine_level=engine_level, console=console)


# ---------------------------------------------------------------------------
# Worker-side initialisation
# ---------------------------------------------------------------------------

def init_worker_logging(queue: Any, engine_level: str) -> None:
    """
    ``ProcessPoolExecutor`` initializer — runs once, inside each worker.

    Points the worker's ``sim`` hierarchy at a single ``QueueHandler`` feeding
    the parent's listener, after detaching any handler that is already attached.

    Under :data:`WORKER_START_METHOD` (``spawn``) a worker starts from a fresh
    interpreter, so ordinarily there is nothing attached and the detach loop is
    a no-op.  It is kept for the case where an import installs a handler of its
    own, and because a worker inheriting a second sink would tear the log file.

    **Nothing here may flush or close a handler.**  Both ``Handler.close()`` and
    ``Handler.flush()`` acquire the underlying ``_io.BufferedWriter`` lock; if
    this ever runs in a *forked* child that lock can be inherited already held,
    by a thread that does not exist here, and the worker blocks forever before
    it ever reaches its task loop: a ``handler.close()`` call intended to
    *prevent* log corruption would itself become the hang.  Detaching is
    sufficient — the file descriptor is released when the worker exits, and
    the parent owns the real handler regardless.
    """
    root = logging.getLogger(ROOT_LOGGER)
    for handler in list(root.handlers):
        root.removeHandler(handler)      # detach only — never close(), see above

    queue_handler = logging.handlers.QueueHandler(queue)
    # Stamp ctx before the record crosses the process boundary — the parent has
    # no idea which cell a worker record came from.
    queue_handler.addFilter(ContextFilter())
    root.addHandler(queue_handler)

    try:
        root.setLevel(level_value(engine_level))
    except ValueError:                             # pragma: no cover - defensive
        root.setLevel(logging.WARNING)
    root.propagate = False
    logging.getLogger(BENCH_LOGGER).setLevel(logging.INFO)
    _mark_configured(True)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

_configured = False


def _mark_configured(value: bool) -> None:
    global _configured
    _configured = value


def _install_fallback_console() -> None:
    """
    Attach a console-only handler when no session has been set up.

    Keeps direct API users working — the test suite calls ``run_gov_difficulty``
    outside any sweep, and a library that swallows its own output when used
    slightly off the beaten path is a bad library.
    """
    root = logging.getLogger(ROOT_LOGGER)
    handler = _StdoutHandler()
    handler.setFormatter(_console_formatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(level_value(DEFAULT_ENGINE_LEVEL))
    root.propagate = False
    logging.getLogger(BENCH_LOGGER).setLevel(logging.INFO)


def bench_log(level: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Emit a harness progress record on ``sim.bench``.

    ``%``-style lazy arguments are supported; with no ``args`` the message is
    passed through verbatim, so progress bars containing ``%`` are safe.
    """
    if not _configured and not logging.getLogger(ROOT_LOGGER).handlers:
        _install_fallback_console()
    logging.getLogger(BENCH_LOGGER).log(level, msg, *args, **kwargs)


def bench_logger() -> logging.Logger:
    """Return the harness progress logger."""
    return logging.getLogger(BENCH_LOGGER)


# ---------------------------------------------------------------------------
# Volume guard
# ---------------------------------------------------------------------------

def verbose_level_warning(
    engine_level: str, total_runs: int, max_cycles: int, n_agents: int
) -> Optional[str]:
    """
    Return a warning string when the requested engine level will flood the log.

    DEBUG on a full sweep is tens of gigabytes.  The user has to be told before
    the sweep starts, not after.  Returns ``None`` when the level is safe — and
    a small verbose run *is* safe, so the warning stays credible by not firing
    on every quick test.
    """
    level = str(engine_level).upper()
    if level not in _VERBOSE_LEVELS or total_runs <= 1:
        return None

    if level == "DEBUG":
        # Dominated by the per-agent health/collect debug lines.
        est_lines = total_runs * max_cycles * max(1, n_agents)
    else:
        # One INFO cycle summary per cycle per run.
        est_lines = total_runs * max_cycles

    if est_lines < _VOLUME_WARNING_LINES:
        return None

    est_bytes = est_lines * 120
    return (
        f"--log-level {level} across {total_runs} runs is estimated to emit "
        f"~{est_lines:,} log lines (~{est_bytes / 1e9:.2f} GB). "
        f"Narrow the sweep (--governments / --difficulties) before debugging at "
        f"this level."
    )
