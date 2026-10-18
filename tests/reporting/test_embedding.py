import random
import sys
import types
from contextlib import nullcontext

import pytest


def install(name, **attrs):
    module = types.ModuleType(name)
    vars(module).update(attrs)
    sys.modules[name] = module
    return module


def setup_embed(modules, monkeypatch, *, zero=False, sort_error=False, field="MMFF94"):
    embed = modules("conformer_generation.embed")
    calls = []

    class Ensemble:
        def __init__(self):
            self.conformers = (
                []
                if zero
                else [
                    types.SimpleNamespace(elements=["C"], coordinates=[]),
                    *[
                        types.SimpleNamespace(elements=["C"], coordinates=[(i, 0, 0)])
                        for i in (1, 2, 3)
                    ],
                ]
            )

        @classmethod
        def from_rdkit(cls, mol, **kwargs):
            calls.append(kwargs)
            return cls()

        def __iter__(self):
            return iter(self.conformers)

        def sort(self):
            if sort_error:
                raise ValueError("sort failed")
            self.conformers.reverse()

    install("morfeus.conformer", ConformerEnsemble=Ensemble)
    install("gpuma.conformer_generation.mol_utils", _to_coord_list=list, _to_symbol_list=list)
    monkeypatch.setattr(embed, "_prepare", lambda value: (object(), 0) if value else None)
    monkeypatch.setattr(embed, "_force_field_for", lambda mol: field)
    monkeypatch.setattr(embed, "_prune_by_rmsd", lambda *a, **k: [0, 2])
    config = types.SimpleNamespace(
        technical=types.SimpleNamespace(device="cpu"),
        optimization=types.SimpleNamespace(multiplicity=1),
    )
    return embed, config, calls


def test_embedding_observes_before_sort_and_preserves_slots_and_calls(
    modules, monkeypatch, capsys, caplog
):
    r = modules("utils.reporting")
    embed, config, calls = setup_embed(modules, monkeypatch)
    outputs = []
    logs = []
    for reporting in (False, True):
        calls.clear()
        caplog.clear()
        random_before = random.getstate()
        with (
            r.diagnostic_report(input_ids=["dup", "bad", "dup"])
            if reporting
            else nullcontext() as report
        ):
            outputs.append(
                embed.generate_ensembles(
                    ["C", "", "C"], config=config, n_confs=6, max_num_confs=1, seed=17
                )
            )
        assert random.getstate() == random_before
        logs.append((list(calls), capsys.readouterr(), [rec.message for rec in caplog.records]))
    assert outputs[0] == outputs[1]
    assert outputs[1][1] is None
    assert outputs[1][0][0].coordinates == [(3, 0, 0)]
    assert logs[0] == logs[1]
    events = report.events
    assert [(e.input_index, e.generated) for e in events if e.reason == "embedding_returned"] == [
        (0, 4),
        (2, 4),
    ]
    assert [(e.input_index, e.conformer_id) for e in events if e.reason == "invalid_conformer"] == [
        (0, 0),
        (2, 0),
    ]
    assert [(e.input_index, e.conformer_id) for e in events if e.reason == "rmsd_pruned"] == [
        (0, 2),
        (2, 2),
    ]
    assert [(e.input_index, e.conformer_id) for e in events if e.reason == "return_cap"] == [
        (0, 1),
        (2, 1),
    ]
    assert [
        (e.input_index, e.conformer_id, e.output_index)
        for e in events
        if e.reason == "conformer_returned"
    ] == [(0, 3, 0), (2, 3, 0)]
    assert all(e.converged is None for e in events)
    assert [(e.requested, e.retained) for e in events if e.reason == "selection_returned"] == [
        (1, 1),
        (1, 1),
    ]
    assert [e.conformer_id for e in events if e.reason == "conformer_generated"] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]
    assert {e.operation_id for e in events}.__len__() == 1


@pytest.mark.parametrize("zero,sort_error,expected", [(True, False, 0), (False, True, 4)])
def test_generation_count_survives_zero_or_sort_exception(
    modules, monkeypatch, zero, sort_error, expected
):
    r = modules("utils.reporting")
    embed, config, _ = setup_embed(
        modules, monkeypatch, zero=zero, sort_error=sort_error, field=None
    )
    with r.diagnostic_report() as report:
        assert embed.generate_structures(["C"], config=config, n_confs=6) == [None]
    generated = [e for e in report.events if e.reason == "embedding_returned"]
    assert generated[0].generated == expected
    assert any(e.reason == "force_field_unavailable" for e in report.events)
    if sort_error:
        assert any(
            e.reason == "operation_exception" and e.exception_type == "ValueError"
            for e in report.events
        )


def test_gpu_retry_has_distinct_attempt_and_preserves_exception(modules, monkeypatch):
    r = modules("utils.reporting")
    embed, config, calls = setup_embed(modules, monkeypatch)
    config.technical.device = "cuda"
    error = RuntimeError("original CUDA failure")

    @r._instrument("embed", "nvmolkit")
    def failed(*args):
        r._generated(0, 2, 6)
        raise error

    monkeypatch.setattr(embed, "_embed_gpu", failed)
    with r.diagnostic_report() as report:
        result = embed.generate_structures(["C"], config=config, n_confs=6)
    assert result[0].coordinates == [(3, 0, 0)]
    returned = [e for e in report.events if e.reason == "embedding_returned"]
    assert [e.generated for e in returned] == [2, 4]
    assert returned[0].attempt_id != returned[1].attempt_id
    assert any(e.reason == "backend_fallback" for e in report.events)
    with r.diagnostic_report(), pytest.raises(RuntimeError) as caught:
        embed.generate_structures(["C"], config=config, allow_cpu_fallback=False)
    assert caught.value is error
    assert len(calls) == 1


def test_gpu_host_mode_counts_ids_fields_and_order(modules, monkeypatch):
    r = modules("utils.reporting")
    embed = modules("conformer_generation.embed")
    calls = []

    class Mol:
        def __init__(self, field):
            self.field = field
            self.conformers = []

        def GetNumConformers(self):
            return len(self.conformers)

        def GetConformers(self):
            return self.conformers

        def GetConformer(self, cid):
            return next(c for c in self.conformers if c.GetId() == cid)

        def GetAtoms(self):
            return [types.SimpleNamespace(GetSymbol=lambda: "C")]

    def embed_molecules(mols, params, **kwargs):
        calls.append(("embed", kwargs["confsPerMolecule"]))
        assert set(kwargs) == {"confsPerMolecule", "hardwareOptions"}
        for mol in mols:
            mol.conformers = [
                types.SimpleNamespace(GetId=lambda i=i: i, GetPositions=lambda i=i: [(i, 0, 0)])
                for i in (8, 3, 11)
            ]

    def minimize(mols, **kwargs):
        calls.append(("minimize", len(mols), kwargs["maxIters"]))
        assert set(kwargs) == {"maxIters", "hardwareOptions"}
        return [[2, 1, 3] for _ in mols]

    install("nvmolkit.embedMolecules", EmbedMolecules=embed_molecules)
    install("nvmolkit.mmffOptimization", MMFFOptimizeMoleculesConfs=minimize)
    install("nvmolkit.uffOptimization", UFFOptimizeMoleculesConfs=minimize)
    install("nvmolkit.types", HardwareOptions=lambda **kwargs: kwargs)
    monkeypatch.setattr(embed, "_prepare", lambda value: (Mol(value or None), 0))
    monkeypatch.setattr(embed, "_force_field_for", lambda mol: mol.field)
    monkeypatch.setattr(embed, "_etkdg_params", lambda *args, **kwargs: "params")
    monkeypatch.setattr(embed, "_prune_by_rmsd", lambda *args, **kwargs: [0, 2])
    config = types.SimpleNamespace(
        technical=types.SimpleNamespace(device="cuda"),
        optimization=types.SimpleNamespace(multiplicity=1),
    )
    outputs, counts = [], []
    for enabled in (False, True):
        calls.clear()
        with r.diagnostic_report() if enabled else nullcontext() as report:
            outputs.append(
                embed.generate_structures(["MMFF94", "UFF", ""], config=config, n_confs=5)
            )
        counts.append(list(calls))
    assert outputs[0] == outputs[1]
    assert counts[0] == counts[1] == [("embed", 5), ("minimize", 1, 200), ("minimize", 1, 200)]
    assert [s.coordinates for s in outputs[1]] == [
        [(3.0, 0.0, 0.0)],
        [(3.0, 0.0, 0.0)],
        [(8.0, 0.0, 0.0)],
    ]
    assert [
        (e.input_index, e.generated) for e in report.events if e.reason == "embedding_returned"
    ] == [(0, 3), (1, 3), (2, 3)]
    assert [e.conformer_id for e in report.events if e.reason == "conformer_generated"] == [
        8,
        3,
        11,
    ] * 3
    assert [
        (e.input_index, e.force_field) for e in report.events if e.reason == "force_field_selected"
    ] == [(0, "MMFF94"), (1, "UFF")]
    assert any(e.reason == "force_field_unavailable" and e.input_index == 2 for e in report.events)
    assert all(e.converged is None for e in report.events)


def test_observer_baseexception_never_changes_fallback_or_return(modules, monkeypatch):
    r = modules("utils.reporting")
    embed, config, calls = setup_embed(modules, monkeypatch)

    def observer(event):
        raise SystemExit("observer only")

    with r.diagnostic_report(observer) as report:
        output = embed.generate_structures(["C"], config=config, n_confs=6)
    assert output[0].coordinates == [(3, 0, 0)]
    assert len(calls) == 1
    assert not report.complete
    assert report.observer_errors == len(report.events)


def test_prepare_exception_keeps_original_error_and_input_identity(modules, monkeypatch):
    r = modules("utils.reporting")
    embed, config, _ = setup_embed(modules, monkeypatch)
    error = ValueError("private molecule text")

    def prepare(value):
        if value == "bad":
            raise error
        return object(), 0

    monkeypatch.setattr(embed, "_prepare", prepare)
    with r.diagnostic_report() as report, pytest.raises(ValueError) as caught:
        embed.generate_structures(["C", "bad"], config=config)
    assert caught.value is error
    events = [
        e for e in report.events if e.stage == "prepare" and e.reason == "operation_exception"
    ]
    assert len(events) == 1
    assert events[0].input_index == 1
    assert "private molecule text" not in str(report.to_dict())
