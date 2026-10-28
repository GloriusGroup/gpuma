"""Opt-in diagnostic facts without changing scientific return values.

Use ``with diagnostic_report(input_ids=ids) as report: ...`` around existing
generation/optimization calls, then ``report.to_dict()`` for JSON transport.
IDs are caller labels aligned to each top-level operation's input order (use
one context per chunk when that mapping changes). Duplicate labels are valid;
``operation_id`` / ``input_index`` identify instances. Conformer IDs are local
to an attempt. Embedding output indices are local to each input ensemble;
optimization output indices address the returned list (zero for a single input).

``complete`` means reporting integrity, never convergence or scientific success.
Force-field convergence is unknown through the current backends. TorchSim
criterion attribution is supported for 0.6.2 only; steps and stop reasons remain
unknown. Observations retain only the latest criterion per input, not trajectories.

Workers must open their own context with no observer and return ``to_dict()`` in
their existing result envelope. Parent ``merge(payload)`` replays each report
once, keyed by report ID; equal facts from distinct reports are preserved. No
context or callbacks are transported. Fork-inherited contexts are inactive.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import wraps
from importlib.metadata import version
from typing import get_args, get_type_hints
from uuid import uuid4

_active = ContextVar("gpuma_report", default=None)
_scope = ContextVar("gpuma_report_scope", default=(None, None, None, None))
_notifying = ContextVar("gpuma_report_notifying", default=False)


@dataclass(frozen=True)
class DiagnosticEvent:
    """Immutable primitive observation; absent evidence is represented by None."""

    operation_id: str | None
    attempt_id: str | None
    input_index: int | None
    stage: str
    reason: str
    input_id: str | int | None = None
    conformer_id: int | None = None
    output_index: int | None = None
    backend: str | None = None
    force_field: str | None = None
    optimizer: str | None = None
    requested: int | None = None
    generated: int | None = None
    retained: int | None = None
    criterion: str | None = None
    threshold: float | None = None
    threshold_unit: str | None = None
    converged: bool | None = None
    optimized: bool | None = None
    steps: int | None = None
    max_steps: int | None = None
    termination_reason: str | None = None
    evidence_source: str | None = None
    exception_type: str | None = None


def _validate_versions(metadata):
    if type(metadata) is not dict or set(metadata) != {"engine_version", "backend_versions"}:
        raise ValueError("invalid version metadata")
    if type(metadata["engine_version"]) not in (str, type(None)):
        raise ValueError("invalid engine version")
    versions = metadata["backend_versions"]
    if type(versions) is not dict or any(
        type(name) is not str or type(value) not in (str, type(None))
        for name, value in versions.items()
    ):
        raise ValueError("invalid backend versions")


def _validate_snapshot(payload):
    """Validate the entire schema before merge bookkeeping or observer replay."""
    required = {
        "schema_version",
        "report_id",
        "engine_version",
        "backend_versions",
        "events",
        "complete",
        "observer_errors",
        "reporting_errors",
        "merged_reports",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("invalid report fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported schema")
    if type(payload["report_id"]) is not str or type(payload["complete"]) is not bool:
        raise ValueError("invalid report identity or completeness")
    for name in ("observer_errors", "reporting_errors"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError("invalid error counter")
    metadata = {name: payload[name] for name in ("engine_version", "backend_versions")}
    _validate_versions(metadata)
    merged = payload["merged_reports"]
    if type(merged) is not dict:
        raise ValueError("invalid merged provenance")
    for report_id, versions in merged.items():
        if type(report_id) is not str:
            raise ValueError("invalid merged report identity")
        _validate_versions(versions)
    if type(payload["events"]) is not list:
        raise ValueError("invalid event list")
    field_types = get_type_hints(DiagnosticEvent)
    events = []
    for event in payload["events"]:
        if type(event) is not dict or set(event) != set(field_types):
            raise ValueError("invalid event fields")
        for name, annotation in field_types.items():
            value = event[name]
            if type(value) not in (get_args(annotation) or (annotation,)):
                raise ValueError("invalid event value")
            if type(value) is int and name != "input_id" and value < 0:
                raise ValueError("negative event count or index")
        events.append(DiagnosticEvent(**event))
    return events, deepcopy(metadata)


def _version(distribution):
    try:
        return version(distribution)
    except BaseException:
        return None


class DiagnosticReport:
    """An in-process collector. Export primitive data with :meth:`to_dict`."""

    def __init__(self, on_event=None, input_ids=None):
        self.report_id = uuid4().hex
        self.engine_version = _version("gpuma")
        self.backend_versions = {
            name: _version(name)
            for name in ("ase", "torch-sim-atomistic", "morfeus-ml", "nvmolkit", "rdkit")
        }
        self.complete = False
        self.observer_errors = 0
        self.reporting_errors = 0
        self._events = []
        self._observer = on_event
        self._input_ids = tuple(input_ids) if input_ids is not None else None
        if self._input_ids is not None and any(
            type(value) not in (str, int, type(None)) for value in self._input_ids
        ):
            raise TypeError("input_ids must contain strings, integers or None")
        self._pid = os.getpid()
        self._merged = {self.report_id}
        self._merged_reports = {}

    @property
    def events(self):
        """Immutable snapshot of the observations collected so far."""
        return tuple(self._events)

    def _append(self, event):
        self._events.append(event)
        if self._observer is not None and not _notifying.get() and self._pid == os.getpid():
            token = _notifying.set(True)
            try:
                self._observer(event)
            except BaseException:
                self.observer_errors += 1
                self.complete = False
            finally:
                _notifying.reset(token)

    def to_dict(self):
        """Return a fresh JSON-serializable snapshot, with no callbacks or tensors."""
        return {
            "schema_version": 1,
            "report_id": self.report_id,
            "engine_version": self.engine_version,
            "backend_versions": dict(self.backend_versions),
            "events": [asdict(event) for event in self._events],
            "complete": self.complete,
            "observer_errors": self.observer_errors,
            "reporting_errors": self.reporting_errors,
            "merged_reports": deepcopy(self._merged_reports),
        }

    def merge(self, payload):
        """Replay a completed worker snapshot once; malformed transport marks loss.

        Merge direct worker reports once in the parent, rather than routing the
        same worker through several aggregate reports. ``merged_reports`` keeps
        worker provenance; top-level metadata describes only the parent process.
        """
        try:
            events, metadata = _validate_snapshot(payload)
            report_id = payload["report_id"]
            if report_id in self._merged:
                return
            self._merged.add(report_id)
            self._merged_reports[report_id] = metadata
            self.observer_errors += payload["observer_errors"]
            self.reporting_errors += payload["reporting_errors"]
            if not payload["complete"]:
                self.reporting_errors += 1
            for event in events:
                self._append(event)
            if self.observer_errors or self.reporting_errors:
                self.complete = False
        except BaseException:
            self.reporting_errors += 1
            self.complete = False


@contextmanager
def diagnostic_report(on_event=None, *, input_ids=None):
    """Collect diagnostics for existing engine calls; nested contexts are isolated.

    Observer failures, including BaseException, never escape into the engine.
    Callbacks must be read-only: they receive immutable events, and engine work
    recursively invoked by a callback is deliberately not reported.
    """
    report = DiagnosticReport(on_event, input_ids)
    token = _active.set(report)
    scope_token = _scope.set((None, None, None, None))
    try:
        yield report
    finally:
        report.complete = not (report.observer_errors or report.reporting_errors)
        _scope.reset(scope_token)
        _active.reset(token)


def _current():
    report = _active.get()
    return (
        report
        if (report is not None and report._pid == os.getpid() and not _notifying.get())
        else None
    )


def _safe(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        report = _current()
        if report is None:
            return None
        try:
            return function(*args, **kwargs)
        except BaseException:
            report.reporting_errors += 1
            report.complete = False
            return None

    return wrapped


@_safe
def _emit(stage, reason, **facts):
    report = _current()
    operation_id, attempt_id, index, backend = _scope.get()
    index = facts.pop("input_index", index)
    input_id = None
    if report._input_ids is not None and index is not None:
        input_id = report._input_ids[index]
    report._append(
        DiagnosticEvent(
            operation_id,
            attempt_id,
            index,
            stage,
            reason,
            input_id=input_id,
            backend=facts.pop("backend", backend),
            **facts,
        )
    )


def _instrument(stage, backend=None, *, share=False):
    """Internal passive call boundary; original exceptions always propagate."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _current() is None:
                return function(*args, **kwargs)
            previous = _scope.get()
            token = None
            try:
                operation = previous[0] if (backend or share) else None
                scope = (
                    operation or uuid4().hex,
                    uuid4().hex if backend else previous[1],
                    (previous[2] if previous[2] is not None else 0 if backend == "ase" else None),
                    backend or previous[3],
                )
                token = _scope.set(scope)
            except BaseException:
                _current().reporting_errors += 1
            try:
                _emit(stage, "operation_started")
                result = function(*args, **kwargs)
                _emit(stage, "operation_returned")
                return result
            except BaseException as exc:
                _emit(
                    stage,
                    "operation_exception",
                    exception_type=type(exc).__name__,
                    termination_reason="exception",
                )
                raise
            finally:
                if token is not None:
                    _scope.reset(token)

        return wrapped

    return decorate


@contextmanager
def _input(index):
    operation, attempt, _, backend = _scope.get()
    token = _scope.set((operation, attempt, index, backend))
    try:
        yield
    finally:
        _scope.reset(token)


@_safe
def _generated(index, count, requested):
    _emit("embed", "embedding_returned", input_index=index, generated=count, requested=requested)
    if count < requested:
        _emit(
            "embed",
            "embedding_shortfall",
            input_index=index,
            generated=count,
            requested=requested,
            evidence_source="cause_unknown",
        )


@_safe
def _cpu_conformers(index, ensemble, requested):
    conformers = list(ensemble)
    _generated(index, len(conformers), requested)
    for ordinal in range(len(conformers)):
        _emit("embed", "conformer_generated", input_index=index, conformer_id=ordinal)
    return {id(conformer): ordinal for ordinal, conformer in enumerate(conformers)}


@_safe
def _conformer_id(identities, conformer):
    return identities[id(conformer)]


@_safe
def _selection(index, identities, survivors, n_keep):
    survivors_set = set(survivors)
    for i, identity in enumerate(identities):
        if i not in survivors_set:
            _emit("embed", "rmsd_pruned", input_index=index, conformer_id=identity)
    for i in survivors[n_keep:] if n_keep is not None else ():
        _emit("embed", "return_cap", input_index=index, conformer_id=identities[i])
    for output_index, i in enumerate(survivors[:n_keep]):
        _emit(
            "embed",
            "conformer_returned",
            input_index=index,
            conformer_id=identities[i],
            output_index=output_index,
        )
    _emit(
        "embed",
        "selection_returned",
        input_index=index,
        requested=n_keep,
        retained=len(survivors[:n_keep]),
    )


@_safe
def _field(index, field, source):
    _emit(
        "ff_optimize",
        "force_field_selected" if field else "force_field_unavailable",
        input_index=index,
        force_field=field,
        evidence_source=source,
    )


@_safe
def _ase_result(converged, steps, optimizer, criterion, threshold):
    _emit(
        "mlip_optimize",
        "criterion_result",
        input_index=_scope.get()[2] or 0,
        optimizer=optimizer,
        criterion=criterion,
        threshold=float(threshold),
        threshold_unit="eV/angstrom" if criterion == "force" else "eV",
        converged=bool(converged),
        steps=int(steps),
        termination_reason="criterion_satisfied" if converged else "unknown",
        evidence_source="ase_run",
    )


@_safe
def _ase_returned():
    _emit(
        "mlip_optimize", "optimization_returned", optimized=True, output_index=_scope.get()[2] or 0
    )


class _CriterionObserver:
    """Passive TorchSim 0.6.2 callback capture; never retain tensor references."""

    def __init__(self, batcher, backend_version, count, input_indices=None):
        self.batcher = batcher
        self.supported = backend_version == "0.6.2"
        self.count = count
        self.input_indices = input_indices
        self.latest = {}

    def wrap(self, criterion):
        def observed(*args, **kwargs):
            indices = self._indices()
            result = criterion(*args, **kwargs)  # exactly once; errors are scientific
            self._capture(indices, result)
            return result  # TorchSim may mutate this object for budget exhaustion

        return observed

    @_safe
    def _indices(self):
        if self.supported:
            return list(self.batcher.current_idx)
        return None

    @_safe
    def _capture(self, indices, result):
        if indices is None:
            self.latest.clear()
            return
        if any(type(i) is not int or not 0 <= i < self.count for i in indices):
            self.latest.clear()
            raise ValueError("unattributable criterion index")
        if len(set(indices)) != len(indices):
            self.latest.clear()
            raise ValueError("duplicate criterion index")
        for index in indices:
            self.latest.pop(index, None)
        values = result.detach().clone().cpu().tolist()
        if type(values) is not list or len(indices) != len(values):
            self.latest.clear()
            raise ValueError("unattributable criterion batch")
        if any(type(value) is not bool for value in values):
            raise ValueError("non-boolean criterion mask")
        self.latest.update(zip(indices, values, strict=True))

    @_safe
    def finish(self, returned, criterion, threshold, unit, optimizer):
        for index in range(self.count):
            converged = self.latest.get(index)
            _emit(
                "mlip_optimize",
                "criterion_result",
                input_index=self.input_indices[index] if self.input_indices else index,
                converged=converged,
                criterion=criterion,
                threshold=float(threshold),
                threshold_unit=unit,
                optimizer=optimizer,
                termination_reason=(
                    "exception"
                    if not returned
                    else "criterion_satisfied"
                    if converged
                    else "unknown"
                ),
                evidence_source="torchsim_criterion" if converged is not None else "unknown",
            )


@_safe
def _observe_criterion(batcher, count, criterion, input_indices=None):
    observer = _CriterionObserver(
        batcher, _current().backend_versions["torch-sim-atomistic"], count, input_indices
    )
    return observer, observer.wrap(criterion)


@_safe
def _gpu_generated(members, requested):
    for index, _, mol, _ in members:
        _generated(index, mol.GetNumConformers(), requested)
        for conformer in mol.GetConformers():
            _emit("embed", "conformer_generated", input_index=index, conformer_id=conformer.GetId())


@_safe
def _gpu_field(members, mol, field):
    index = next(index for index, _, candidate, _ in members if candidate is mol)
    _field(index, field, "nvmolkit_host_flags_unavailable")


@_safe
def _finish_criterion(observer, returned, config, optimizer, default_force):
    if observer is None:
        return
    force = config.optimization.force_convergence_criterion
    energy = config.optimization.energy_convergence_criterion
    if force is not None or energy is None:
        observer.finish(
            returned,
            "force",
            force if force is not None else default_force,
            "eV/angstrom",
            optimizer,
        )
    else:
        observer.finish(returned, "energy", energy, "eV", optimizer)
