import types
from contextlib import contextmanager, nullcontext

import pytest

from .test_embedding import install


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

    def __getitem__(self, index):
        return types.SimpleNamespace(item=lambda: self.values[index])


def setup_optimizer(modules, monkeypatch, converged=False):
    calls = []

    class Atoms:
        def __init__(self, symbols, positions, info=None):
            self.symbols, self.positions, self.info = symbols, positions, info

        def get_positions(self):
            return Tensor(self.positions)

        def get_potential_energy(self):
            calls.append("energy")
            return -1.0

        def get_chemical_symbols(self):
            return self.symbols

    class Optimizer:
        def __init__(self, atoms, logfile):
            assert logfile is None
            self.atoms = atoms

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            if self.atoms.positions[0][0] == 99:
                raise ValueError("bad coordinates")
            return converged

        def get_number_of_steps(self):
            calls.append("steps")
            return 7

    @contextmanager
    def timed(*args):
        yield types.SimpleNamespace(elapsed=0)

    install("ase", Atoms=Atoms)
    install("ase.optimize", BFGS=Optimizer, FIRE=Optimizer, LBFGS=Optimizer)
    install(
        "gpuma.config",
        DEFAULT_CONFIG={"optimization": {"force_convergence_criterion": 0.05}},
        VALID_MEMORY_SCALES_WITH={"n_atoms"},
        Config=object,
        load_config_from_file=lambda: None,
        resolve_model_type=lambda config: "fairchem",
    )
    install(
        "gpuma.models",
        _parse_device_string=lambda device: device,
        _device_for_torch=lambda device: device,
        load_calculator=lambda config: None,
        load_torchsim_model=lambda config: None,
    )
    install("gpuma.utils.decorators", timed_block=timed)
    install("gpuma.utils.logging_utils", log_optimization_summary=lambda *args: None)
    optimizer = modules("optimizer")
    structure = modules("structure").Structure
    monkeypatch.setattr(optimizer, "_get_cached_calculator", lambda config: object())
    config = types.SimpleNamespace(
        technical=types.SimpleNamespace(
            device="cpu",
            max_memory_padding=0.9,
            memory_scaling_factor=1.2,
            max_atoms_to_try=50,
            steps_between_swaps=10,
        ),
        optimization=types.SimpleNamespace(
            batch_optimizer="fire",
            batch_optimization_mode="batch",
            force_convergence_criterion=0.05,
            energy_convergence_criterion=None,
        ),
        model=types.SimpleNamespace(model_name="example"),
    )
    return optimizer, structure, config, calls, Atoms


@pytest.mark.parametrize("converged", [True, False])
def test_ase_bool_none_slots_and_science_calls_unchanged(modules, monkeypatch, capsys, converged):
    r = modules("reporting")
    optimizer, Structure, config, calls, _ = setup_optimizer(modules, monkeypatch, converged)
    outputs, observations = [], []
    for enabled in (False, True):
        calls.clear()
        inputs = [
            Structure(["C"], [(x, 0, 0)], 0, 1, comment="keep", metadata={"x": 1})
            for x in (0, 99, 2)
        ]
        with r.diagnostic_report() if enabled else nullcontext() as report:
            outputs.append(optimizer.optimize_structure_batch(inputs, config))
        observations.append((list(calls), capsys.readouterr()))
    assert outputs[0] == outputs[1]
    assert outputs[1][1] is None
    assert observations[0] == observations[1]
    assert len([call for call in calls if call == "energy"]) == 2
    events = [e for e in report.events if e.reason == "criterion_result"]
    assert [(e.input_index, e.converged, e.steps) for e in events] == [
        (0, converged, 7),
        (2, converged, 7),
    ]
    assert all(e.max_steps is None for e in events)
    assert all(e.evidence_source == "ase_run" for e in events)
    assert any(e.input_index == 1 and e.reason == "operation_exception" for e in report.events)
    assert [
        (e.input_index, e.output_index, e.optimized)
        for e in report.events
        if e.reason == "optimization_returned"
    ] == [(0, 0, True), (2, 2, True)]


@pytest.mark.parametrize("energy,failure", [(False, False), (True, False), (False, True)])
def test_torchsim_actual_callback_attribution_and_budget_mutation(
    modules, monkeypatch, energy, failure
):
    r = modules("reporting")
    optimizer, Structure, config, calls, Atoms = setup_optimizer(modules, monkeypatch)
    config.technical.device = "cuda"
    if energy:
        config.optimization.force_convergence_criterion = None
        config.optimization.energy_convergence_criterion = 0.01

    class Batcher:
        def __init__(self, *args, **kwargs):
            self.current_idx = [2, 0]

    mask = Tensor([False, True])

    def criterion(state, previous):
        calls.append(("criterion", state, previous))
        return mask

    error = ValueError("original runner failure")

    def optimize(**kwargs):
        calls.append(("optimize", sorted(kwargs)))
        batcher = kwargs["autobatcher"]
        mask.values[:] = [False, True]
        assert kwargs["convergence_fn"]("state", "previous") is mask
        mask.values[:] = [True, True]  # budget completion must not rewrite evidence
        if failure:
            raise error
        batcher.current_idx[:] = [1]
        mask.values[:] = [False]
        assert kwargs["convergence_fn"]("state2", "previous2") is mask
        mask.values[:] = [True]
        atoms = [Atoms(["C"], [(i, 0, 0)]) for i in range(3)]
        return types.SimpleNamespace(
            to_atoms=lambda: atoms,
            energy=Tensor([-1, -2, -3]),
            charge=Tensor([0, 0, 0]),
            spin=Tensor([1, 1, 1]),
        )

    install("torch", device=lambda value: value, float64="float64")
    install(
        "torch_sim",
        Optimizer=types.SimpleNamespace(
            fire="fire", gradient_descent="gd", lbfgs="lbfgs", bfgs="bfgs"
        ),
        generate_force_convergence_fn=lambda **kwargs: criterion,
        generate_energy_convergence_fn=lambda **kwargs: criterion,
        io=types.SimpleNamespace(atoms_to_state=lambda *args, **kwargs: object()),
        optimize=optimize,
    )
    install("torch_sim.autobatching", InFlightAutoBatcher=Batcher)
    install("torch_sim.typing", SystemExtras=types.SimpleNamespace(CHARGE="charge", SPIN="spin"))
    install("gpuma.autobatching", install_chunked_edge_scalers=lambda: None)
    monkeypatch.setattr(optimizer, "_get_cached_torchsim_model", lambda config: object())
    monkeypatch.setattr(optimizer, "_whole_job_memory_scaler", lambda *args: 1)
    monkeypatch.setattr(optimizer, "_memory_scales_with", lambda config: "n_atoms")
    outputs, observations = [], []
    for enabled in (False, True):
        calls.clear()
        with r.diagnostic_report() if enabled else nullcontext() as report:
            if enabled:
                report.backend_versions["torch-sim-atomistic"] = "0.6.2"
            inputs = [Structure(["C"], [(i, 0, 0)], 0, 1) for i in range(3)]
            if failure:
                with pytest.raises(ValueError) as caught:
                    optimizer.optimize_structure_batch(inputs, config)
                assert caught.value is error
            else:
                outputs.append(optimizer.optimize_structure_batch(inputs, config))
        observations.append(list(calls))
    assert observations[0] == observations[1]
    if not failure:
        assert outputs[0] == outputs[1]
    events = [e for e in report.events if e.reason == "criterion_result"]
    assert [(e.input_index, e.converged) for e in events] == [
        (0, True),
        (1, None if failure else False),
        (2, False),
    ]
    assert all(e.criterion == ("energy" if energy else "force") for e in events)
    assert all(e.steps is None and e.max_steps is None for e in events)
    assert events[2].termination_reason == ("exception" if failure else "unknown")
