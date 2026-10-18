import dataclasses
import importlib.util
import json
import multiprocessing
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest


@contextmanager
def _operation(reporting):
    """Provide only the identity scope; engine integration is tested separately."""
    token = reporting._scope.set((uuid4().hex, None, None, None))
    try:
        yield
    finally:
        reporting._scope.reset(token)


def test_reports_isolate_nested_observers_and_transport(modules):
    r = modules("reporting")
    seen = []

    def observer(event):
        seen.append(event)
        r._emit("prepare", "parse_rejected", input_index=99)
        raise KeyboardInterrupt

    with r.diagnostic_report(observer, input_ids=["same", "same"]) as outer:
        with _operation(r):
            r._emit("prepare", "parse_rejected", input_index=1)
        with r.diagnostic_report() as child:
            with _operation(r):
                r._emit("prepare", "parse_rejected", input_index=0)
        assert len(outer.events) == 1
        outer.merge(json.loads(json.dumps(child.to_dict())))
        outer.merge(child.to_dict())
    assert len(outer.events) == len(seen) == 2
    assert outer.events[0].input_id == "same"
    assert outer.events[0].input_index == 1
    assert outer.events[0].operation_id != outer.events[1].operation_id
    assert outer.observer_errors == 2
    assert not outer.complete
    with pytest.raises(dataclasses.FrozenInstanceError):
        outer.events[0].reason = "changed"
    assert child.complete


def test_criterion_observer_copies_mask_and_preserves_identity(modules):
    r = modules("reporting")

    class Tensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.values[:])

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Batcher:
        current_idx = [2, 0]

    batcher = Batcher()
    calls = []
    mask = Tensor([False, True])

    def criterion(*args, **kwargs):
        calls.append((args, kwargs))
        return mask

    with r.diagnostic_report() as report:
        with _operation(r):
            tracker = r._CriterionObserver(batcher, "0.6.2", 3)
            wrapped = tracker.wrap(criterion)
            assert wrapped("state", previous_energy="old") is mask
            mask.values[:] = [True, True]
            batcher.current_idx[:] = [1]
            mask.values[:] = [False]
            assert wrapped("new") is mask
            tracker.finish(True, "energy", 0.01, "eV", "fire")
    assert calls == [(("state",), {"previous_energy": "old"}), (("new",), {})]
    events = [e for e in report.events if e.reason == "criterion_result"]
    assert [(e.input_index, e.converged) for e in events] == [(0, True), (1, False), (2, False)]
    assert all(e.steps is None and e.max_steps is None for e in events)
    assert events[1].termination_reason == "unknown"


def test_criterion_exception_is_original_and_unknown_version_is_honest(modules):
    r = modules("reporting")
    error = ValueError("original")

    def fail(*args):
        raise error

    with r.diagnostic_report() as report:
        with _operation(r):
            tracker = r._CriterionObserver(object(), None, 1)
            with pytest.raises(ValueError) as caught:
                tracker.wrap(fail)(None)
            assert caught.value is error
            tracker.finish(False, "force", 0.1, "eV/angstrom", "fire")
    assert report.events[-1].converged is None
    assert report.events[-1].evidence_source == "unknown"


def _worker_snapshot(connection, path):
    spec = importlib.util.spec_from_file_location("report_worker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with module.diagnostic_report(input_ids=["worker"]) as report:
        with _operation(module):
            module._emit("prepare", "parse_rejected", input_index=0)
    connection.send(report.to_dict())
    connection.close()


def test_spawn_transport_merges_once_and_keeps_worker_provenance(modules):
    r = modules("reporting")
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker_snapshot, args=(sender, str(Path(r.__file__))))
    process.start()
    sender.close()
    try:
        assert receiver.poll(15)
        payload = receiver.recv()
    finally:
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
    assert process.exitcode == 0
    seen = []
    with r.diagnostic_report(seen.append) as parent:
        parent.merge(payload)
        parent.merge(payload)
    assert len(seen) == 1
    assert seen[0].input_id == "worker"
    assert parent.complete
    assert (
        parent.to_dict()["merged_reports"][payload["report_id"]]["backend_versions"]
        == payload["backend_versions"]
    )


def test_inherited_report_never_notifies_parent_and_copy_failure_is_isolated(modules, monkeypatch):
    r = modules("reporting")
    seen = []
    with r.diagnostic_report(seen.append) as report:
        monkeypatch.setattr(r.os, "getpid", lambda: report._pid + 1)
        r._emit("prepare", "parse_rejected")
        assert seen == []
        monkeypatch.undo()
        with _operation(r):
            tracker = r._CriterionObserver(types.SimpleNamespace(current_idx=[0]), "0.6.2", 1)
            original = object()
            assert tracker.wrap(lambda: original)() is original
            tracker.finish(True, "force", 0.1, "eV/angstrom", "fire")
    assert not report.complete
    assert report.reporting_errors == 1
    assert report.events[-1].converged is None


def test_import_sandbox_restores_cached_modules_and_parent_attributes(monkeypatch):
    from .conftest import isolated_modules

    parent = types.ModuleType("gpuma")
    original = types.ModuleType("gpuma.reporting")
    parent.reporting = original
    morfeus = types.ModuleType("morfeus")
    morfeus.conformer = object()
    saved = morfeus.conformer
    monkeypatch.setitem(sys.modules, "gpuma", parent)
    monkeypatch.setitem(sys.modules, "gpuma.reporting", original)
    monkeypatch.setitem(sys.modules, "morfeus", morfeus)
    with isolated_modules() as load:
        assert load("reporting") is not original
        morfeus.conformer = object()
        morfeus.new_child = object()
    assert sys.modules["gpuma.reporting"] is parent.reporting is original
    assert morfeus.conformer is saved
    assert not hasattr(morfeus, "new_child")


def test_later_capture_failure_invalidates_stale_convergence(modules):
    from .test_optimizer import Tensor

    r = modules("reporting")
    batcher = types.SimpleNamespace(current_idx=[0])
    with r.diagnostic_report() as report:
        with _operation(r):
            tracker = r._CriterionObserver(batcher, "0.6.2", 1)
            tracker.wrap(lambda: Tensor([True]))()
            result = object()
            assert tracker.wrap(lambda: result)() is result
            tracker.finish(True, "force", 0.1, "eV/angstrom", "fire")
    assert report.events[-1].converged is None
    assert not report.complete


@pytest.mark.parametrize(
    "path,bad",
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("complete",), "false"),
        (("observer_errors",), -1),
        (("reporting_errors",), True),
        (("reporting_errors",), "0"),
        (("report_id",), 123),
        (("engine_version",), 81),
        (("backend_versions",), []),
        (("backend_versions", "ase"), False),
        (("backend_versions",), {1: "3.27.0"}),
        (("merged_reports",), {1: {"engine_version": None, "backend_versions": {}}}),
        (("merged_reports",), {"child": {"engine_version": [], "backend_versions": {}}}),
        (("merged_reports",), {"child": {"engine_version": None, "backend_versions": {"ase": 3}}}),
        (("events", 0, "converged"), "false"),
        (("events", 0, "optimized"), 1),
        (("events", 0, "input_index"), "0"),
        (("events", 0, "generated"), True),
        (("events", 0, "steps"), -1),
        (("events", 0, "threshold"), "0.05"),
        (("events", 0, "stage"), None),
        (("events", 0, "input_id"), False),
    ],
)
def test_merge_rejects_malformed_schema_atomically_then_accepts_valid_retry(modules, path, bad):
    r = modules("reporting")
    with r.diagnostic_report() as child:
        with _operation(r):
            r._emit("mlip_optimize", "criterion_result", input_index=0, converged=False)
    valid = child.to_dict()
    malformed = deepcopy(valid)
    target = malformed
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = bad
    seen = []
    with r.diagnostic_report(seen.append) as parent:
        parent.merge(malformed)
        assert seen == []
        assert parent.events == ()
        assert parent.to_dict()["merged_reports"] == {}
        assert parent.reporting_errors == 1
        parent.merge(valid)
        parent.merge(valid)
    assert len(seen) == 1
    assert seen[0].converged is False
    assert parent.observer_errors == 0
    assert not parent.complete


@pytest.mark.parametrize(
    "indices,values",
    [
        (["0"], [True]),
        ([False], [True]),
        ([-1], [True]),
        ([2], [True]),
        ([0, 0], [True, True]),
        ([0], [True, False]),
        ([0, 1], [True]),
        ([[0]], [True]),
    ],
)
def test_untrusted_attribution_clears_all_prior_evidence(modules, indices, values):
    from .test_optimizer import Tensor

    r = modules("reporting")
    batcher = types.SimpleNamespace(current_idx=[0, 1])
    with r.diagnostic_report() as report:
        with _operation(r):
            tracker = r._CriterionObserver(batcher, "0.6.2", 2)
            tracker.wrap(lambda: Tensor([False, True]))()
            batcher.current_idx = indices
            mask = Tensor(values)
            calls = []

            def criterion():
                calls.append(1)
                return mask

            assert tracker.wrap(criterion)() is mask
            tracker.finish(True, "force", 0.1, "eV/angstrom", "fire")
    assert calls == [1]
    assert [(event.converged, event.evidence_source) for event in report.events] == [
        (None, "unknown"),
        (None, "unknown"),
    ]
    assert not report.complete


def test_valid_index_copy_failure_preserves_other_inputs(modules):
    from .test_optimizer import Tensor

    r = modules("reporting")
    batcher = types.SimpleNamespace(current_idx=[0, 1])
    with r.diagnostic_report() as report:
        with _operation(r):
            tracker = r._CriterionObserver(batcher, "0.6.2", 2)
            tracker.wrap(lambda: Tensor([False, True]))()
            batcher.current_idx = [0]
            result = object()
            assert tracker.wrap(lambda: result)() is result
            tracker.finish(True, "force", 0.1, "eV/angstrom", "fire")
    assert [event.converged for event in report.events] == [None, True]
    assert not report.complete
