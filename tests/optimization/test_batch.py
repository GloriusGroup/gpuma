"""Tests for batch optimization — real ORB models."""

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_structure_batch
from gpuma.structure import Structure

from conftest import DEVICE, MULTI_XYZ, SMALL_BATCH_XYZ, requires_gpu


class TestOptimizeSequential:
    """Sequential (one-by-one) batch optimization."""

    def test_sequential_batch(self, methane, ethanol, orb_sequential_config):
        """Multiple structures are optimized sequentially with energies."""
        results = optimize_structure_batch([methane, ethanol], orb_sequential_config)
        assert len(results) == 2
        for r in results:
            assert r.energy is not None

    def test_sequential_on_cpu(self, methane):
        """Sequential optimization works on CPU."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "sequential",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": "cpu"},
        })
        results = optimize_structure_batch([methane], config)
        assert len(results) == 1
        assert results[0].energy is not None


class TestOptimizeBatch:
    """GPU-accelerated batch optimization via torch-sim."""

    @requires_gpu
    def test_batch_ethanol(self, ethanol, orb_config):
        """Batch optimization of identical structures produces valid energies."""
        results = optimize_structure_batch([ethanol] * 3, orb_config)
        assert len(results) == 3
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_from_file(self, orb_config):
        """Batch optimization of real structures from test data file."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(SMALL_BATCH_XYZ))[:10]
        results = optimize_structure_batch(structures, orb_config)
        assert len(results) >= 1
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_fallback_on_cpu(self, methane):
        """Batch mode with CPU device falls back to sequential."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "batch",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": "cpu"},
        })
        results = optimize_structure_batch([methane], config)
        assert len(results) == 1

    @requires_gpu
    def test_batch_unknown_mode(self, methane, orb_config):
        """Unknown optimization mode raises ValueError."""
        orb_config.optimization.batch_optimization_mode = "unknown"
        with pytest.raises(ValueError, match="Unknown optimization mode"):
            optimize_structure_batch([methane], orb_config)

    @requires_gpu
    def test_batch_mixed_size_structures(self, ethanol):
        """Autobatcher handles structures with different atom counts."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(MULTI_XYZ))
        config = Config({
            "optimization": {
                "batch_optimization_mode": "batch",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE, "max_atoms_to_try": 10000},
        })
        results = optimize_structure_batch(structures, config)
        assert len(results) == len(structures)
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_preserves_structure_count(self, orb_config):
        """Batch optimization returns exactly as many structures as input."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(MULTI_XYZ))
        n = len(structures)
        results = optimize_structure_batch(structures, orb_config)
        assert len(results) == n


class TestSequentialVsBatchConsistency:
    """Sequential and batch modes should produce comparable results."""

    @requires_gpu
    def test_energies_are_comparable(self):
        """Same structures give similar energies in sequential and batch modes."""
        import gpuma

        all_structures = gpuma.read_multi_xyz(str(MULTI_XYZ))

        def _copy(structs):
            return [
                Structure(
                    symbols=list(s.symbols),
                    coordinates=[list(c) for c in s.coordinates],
                    charge=s.charge,
                    multiplicity=s.multiplicity,
                )
                for s in structs
            ]

        common = {
            "optimization": {"force_convergence_criterion": 5e-2},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE, "max_atoms_to_try": 10000},
        }

        seq_config = Config({
            **common,
            "optimization": {
                **common["optimization"],
                "batch_optimization_mode": "sequential",
            },
        })
        batch_config = Config({
            **common,
            "optimization": {
                **common["optimization"],
                "batch_optimization_mode": "batch",
            },
        })

        seq_results = optimize_structure_batch(_copy(all_structures), seq_config)
        batch_results = optimize_structure_batch(_copy(all_structures), batch_config)

        assert len(seq_results) == len(batch_results)

        for s, b in zip(seq_results, batch_results):
            assert abs(s.energy - b.energy) < 1.0, (
                f"Sequential ({s.energy:.4f}) and batch ({b.energy:.4f}) "
                f"energies differ by more than 1 eV"
            )


class TestFailureAlignment:
    """A structure that fails to optimize must not shift the others.

    A stub calculator stands in for the MLIP: the property under test is how
    _optimize_sequential handles an exception, and a real model would have to
    be coaxed into failing on cue to show it.
    """

    @staticmethod
    def _flaky_calculator(fail_on_n_atoms):
        import numpy as np

        class FlakyCalc:
            def _check(self, atoms):
                if len(atoms) == fail_on_n_atoms:
                    raise RuntimeError("simulated failure")

            def get_potential_energy(self, atoms=None, force_consistent=False):
                self._check(atoms)
                return 0.0

            def get_forces(self, atoms=None):
                self._check(atoms)
                return np.zeros((len(atoms), 3))

            def get_property(self, name, atoms=None, allow_calculation=True):
                self._check(atoms)
                return {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}[name]

            def calculation_required(self, atoms, properties):
                return True

        return FlakyCalc()

    @staticmethod
    def _chain(n, tag):
        from gpuma.structure import Structure

        return Structure(
            symbols=["H"] * n,
            coordinates=[(i * 1.0, 0.0, 0.0) for i in range(n)],
            charge=0,
            multiplicity=1,
            comment=tag,
        )

    def test_failure_becomes_none_in_place(self, monkeypatch):
        """Results stay aligned to inputs, with None marking the failure.

        The old code appended only successes, so [A, B, C_fails, D] came back
        as [A, B, D] -- and D was then written out labelled as structure 3.
        """
        from gpuma import optimizer as opt
        from gpuma.config import Config

        monkeypatch.setattr(
            opt, "_get_cached_calculator", lambda config: self._flaky_calculator(10)
        )
        config = Config({
            "optimization": {"batch_optimization_mode": "sequential"},
            "technical": {"device": "cpu"},
        })
        inputs = [
            self._chain(3, "A"),
            self._chain(4, "B"),
            self._chain(10, "C_fails"),
            self._chain(5, "D"),
        ]

        results = opt.optimize_structure_batch(inputs, config)

        assert len(results) == len(inputs)
        assert results[2] is None
        assert [r.comment for r in results if r is not None] == ["A", "B", "D"]
        assert results[3].comment == "D"

    def test_all_succeed_has_no_none(self, monkeypatch):
        """The common path is unchanged."""
        from gpuma import optimizer as opt
        from gpuma.config import Config

        monkeypatch.setattr(
            opt, "_get_cached_calculator", lambda config: self._flaky_calculator(-1)
        )
        config = Config({
            "optimization": {"batch_optimization_mode": "sequential"},
            "technical": {"device": "cpu"},
        })
        inputs = [self._chain(3, "A"), self._chain(4, "B")]

        results = opt.optimize_structure_batch(inputs, config)

        assert [r.comment for r in results] == ["A", "B"]

    def test_summary_counts_only_successes(self, monkeypatch, caplog):
        """None entries must not be counted as optimized structures."""
        import logging as _logging

        from gpuma import optimizer as opt
        from gpuma.config import Config

        monkeypatch.setattr(
            opt, "_get_cached_calculator", lambda config: self._flaky_calculator(10)
        )
        config = Config({
            "optimization": {"batch_optimization_mode": "sequential"},
            "technical": {"device": "cpu"},
        })
        inputs = [self._chain(3, "A"), self._chain(10, "B_fails")]

        with caplog.at_level(_logging.INFO):
            opt.optimize_structure_batch(inputs, config)

        assert "Structures output:   1" in caplog.text
        assert "Success rate:        1/2" in caplog.text

    def test_api_labels_survivors_by_input_position(self, monkeypatch, tmp_path):
        """The output file must not renumber structures around a failure."""
        from gpuma import api, optimizer as opt
        from gpuma.config import Config

        monkeypatch.setattr(
            opt, "_get_cached_calculator", lambda config: self._flaky_calculator(10)
        )
        inputs = [
            self._chain(3, "A"),
            self._chain(10, "B_fails"),
            self._chain(5, "C"),
        ]
        monkeypatch.setattr(api, "read_multi_xyz", lambda *a, **kw: inputs)
        monkeypatch.setattr(api.os.path, "isfile", lambda p: True)

        out = tmp_path / "out.xyz"
        config = Config({
            "optimization": {"batch_optimization_mode": "sequential"},
            "technical": {"device": "cpu"},
        })
        api.optimize_batch_multi_xyz_file("in.xyz", str(out), config)

        text = out.read_text()
        # The third input keeps label 3; it must not inherit the failure's 2.
        assert "Optimized structure 1 from" in text
        assert "Optimized structure 3 from" in text
        assert "Optimized structure 2 from" not in text
