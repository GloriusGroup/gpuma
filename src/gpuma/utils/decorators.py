"""Decorators and timing utilities used across the GPUMA package.

Includes :class:`capture_timings`, an opt-in context manager intended for
benchmarking and debugging that records the named phases produced by
:class:`timed_block` (e.g. ``"Model loading"``, ``"Memory estimation"``,
``"Optimization"``, ``"Total optimization"``) without affecting the
public API of any function.
"""

import contextvars
import logging
from functools import wraps
from time import perf_counter

logger = logging.getLogger(__name__)

# Stack of currently-active TimingCapture objects. Stored in a
# ``ContextVar`` so concurrent benchmarks (threads or async tasks)
# do not leak phase events into each other.
_active_captures: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "_gpuma_active_captures", default=()
)


def time_it(func):
    """Measure the execution time of a function and log the result.

    Parameters
    ----------
    func:
        Callable to be wrapped.

    Returns
    -------
    callable
        Wrapped function that logs its runtime at :mod:`logging.INFO` level.

    """

    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start_time
        logger.info("Function: %r took: %.2f sec", func.__name__, elapsed)
        return result

    return wrap


class timed_block:
    """Context manager that measures and logs a named code block.

    The elapsed time (in seconds) is available via the :attr:`elapsed`
    attribute after the block exits. While inside any
    :class:`capture_timings` context manager, the (name, elapsed) pair
    is also published to that capture.

    Example
    -------
    >>> with timed_block("model loading") as tb:
    ...     model = load_model()
    >>> print(tb.elapsed)
    """

    def __init__(self, name: str, *, level: int = logging.INFO):
        self.name = name
        self.elapsed: float = 0.0
        self._level = level

    def __enter__(self):
        """Start the timer and return ``self`` for attribute access."""
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the timer, store elapsed time, log, and notify captures."""
        self.elapsed = perf_counter() - self._start
        logger.log(self._level, "%s took %.2f sec", self.name, self.elapsed)
        for capture in _active_captures.get():
            capture._record(self.name, self.elapsed)


class TimingCapture:
    """Recording target for :class:`capture_timings`.

    Canonical batch-optimization phase names (produced by ``timed_block``
    inside the GPUMA optimizer) are exposed as attributes:

    - ``model_loading``       — ``"Model loading"``
    - ``memory_estimation``   — ``"Memory estimation"`` (autobatcher probe)
    - ``optimization``        — ``"Optimization"``
    - ``total``               — ``"Total optimization"``

    Other ``timed_block`` events are still recorded; access them via
    :meth:`get` or :attr:`raw`. Missing phases default to ``0.0``.
    """

    _PHASE_BY_LABEL = {
        "Model loading":      "model_loading",
        "Memory estimation":  "memory_estimation",
        "Optimization":       "optimization",
        "Total optimization": "total",
    }

    def __init__(self) -> None:
        self.raw: dict[str, float] = {}

    def _record(self, label: str, elapsed: float) -> None:
        self.raw[label] = elapsed

    def get(self, label: str, default: float = 0.0) -> float:
        return self.raw.get(label, default)

    @property
    def model_loading(self) -> float:
        return self.raw.get("Model loading", 0.0)

    @property
    def memory_estimation(self) -> float:
        return self.raw.get("Memory estimation", 0.0)

    @property
    def optimization(self) -> float:
        return self.raw.get("Optimization", 0.0)

    @property
    def total(self) -> float:
        return self.raw.get("Total optimization", 0.0)

    @property
    def overhead(self) -> float:
        """``model_loading + memory_estimation``."""
        return self.model_loading + self.memory_estimation

    def as_dict(self) -> dict[str, float]:
        """Return canonical phases (and ``overhead``) as a flat dict."""
        return {
            "model_loading":     self.model_loading,
            "memory_estimation": self.memory_estimation,
            "overhead":          self.overhead,
            "optimization":      self.optimization,
            "total":             self.total,
        }


class capture_timings:
    """Opt-in context manager that captures :class:`timed_block` phases.

    Intended for benchmarking and debugging only — the public API of
    GPUMA functions is unaffected. Nesting is supported; each capture
    only sees events that fire while it is active.

    Example
    -------
    >>> from gpuma import capture_timings, optimize_structure_batch
    >>> with capture_timings() as t:
    ...     results = optimize_structure_batch(structures, config)
    >>> t.model_loading, t.memory_estimation, t.optimization, t.total
    """

    def __enter__(self) -> TimingCapture:
        self._capture = TimingCapture()
        self._token = _active_captures.set(
            (*_active_captures.get(), self._capture)
        )
        return self._capture

    def __exit__(self, *exc_info) -> None:
        _active_captures.reset(self._token)
