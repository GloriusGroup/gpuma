"""Tests for convergence criteria, warnings, validation, and summary logging."""

import logging

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_single_structure, optimize_structure_batch
from gpuma.structure import Structure

from conftest import DEVICE


class TestConvergenceWarnings:
    """Warnings when convergence criteria are ambiguous or unsupported."""

    def test_both_criteria_warns(self, methane, caplog):
        """Both force and energy criteria set logs a warning."""
        config = Config({
            "optimization": {
                "force_convergence_criterion": 0.5,
                "energy_convergence_criterion": 0.001,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.WARNING):
            optimize_single_structure(methane, config)
        assert "Both force and energy convergence criteria given" in caplog.text

    def test_energy_only_criterion_is_used(self, methane):
        """The ASE path used to swap an energy criterion for the default force one."""
        events = _ase_criterion(methane, energy_convergence_criterion=10.0)
        assert (events.criterion, events.threshold_unit) == ("energy", "eV")
        # A 10 eV tolerance is met by the first step; the force default needs more.
        assert events.converged and events.steps == 1


def _ase_criterion(structure, **optimization):
    """Optimize ``structure`` on the ASE path and return its criterion_result event."""
    from gpuma.utils.reporting import diagnostic_report

    config = Config({
        "optimization": {"force_convergence_criterion": None, **optimization},
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE},
    })
    with diagnostic_report() as report:
        optimize_single_structure(structure, config)
    return next(e for e in report.events if e.reason == "criterion_result")


class TestStepLimit:
    """max_steps caps the ASE path, which used to run without limit."""

    def test_step_limit_marks_unconverged(self, ethanol):
        events = _ase_criterion(ethanol, force_convergence_criterion=1e-6, max_steps=2)
        assert not events.converged and events.steps == 2
        assert ethanol.converged is False

    def test_converged_flag(self, methane):
        _ase_criterion(methane, force_convergence_criterion=0.5)
        assert methane.converged is True

    def test_non_finite_energy_is_a_failure(self, methane):
        """NaN must not reach the output file or the JSON summary."""
        import numpy as np
        from ase.calculators.calculator import Calculator, all_changes

        class NaNCalc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.results = {"energy": float("nan"), "forces": np.zeros((len(atoms), 3))}

        config = Config({"technical": {"device": "cpu"}})
        with pytest.raises(RuntimeError, match="non-finite"):
            optimize_single_structure(methane, config, NaNCalc())

    def test_impossible_spin_state_is_rejected(self):
        h3 = Structure(
            symbols=["H"] * 3, coordinates=[(0.0, 0.0, float(i)) for i in range(3)],
            charge=0, multiplicity=1,
        )
        with pytest.raises(ValueError, match="multiplicity"):
            optimize_single_structure(h3, Config({"technical": {"device": "cpu"}}))


class TestConvergenceBehavior:
    """Verify that convergence criteria affect optimization results."""

    def test_tighter_criterion_lowers_energy(self, ethanol):
        """Tighter force convergence produces lower or equal energy."""
        loose = Config({
            "optimization": {"force_convergence_criterion": 0.5},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        tight = Config({
            "optimization": {"force_convergence_criterion": 0.01},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })

        loose_input = Structure(
            symbols=list(ethanol.symbols),
            coordinates=[list(c) for c in ethanol.coordinates],
            charge=ethanol.charge,
            multiplicity=ethanol.multiplicity,
        )
        tight_input = Structure(
            symbols=list(ethanol.symbols),
            coordinates=[list(c) for c in ethanol.coordinates],
            charge=ethanol.charge,
            multiplicity=ethanol.multiplicity,
        )

        loose_result = optimize_single_structure(loose_input, loose)
        tight_result = optimize_single_structure(tight_input, tight)

        assert tight_result.energy <= loose_result.energy + 0.01


class TestOptimizationSummary:
    """Optimization summary is logged after batch runs."""

    def test_summary_logged(self, methane, orb_sequential_config, caplog):
        """Summary includes structure counts and optimizer info."""
        with caplog.at_level(logging.INFO, logger="gpuma.utils.logging_utils"):
            optimize_structure_batch([methane], orb_sequential_config)
        assert "GPUMA Optimization Summary" in caplog.text
        assert "Structures input:    1" in caplog.text
        assert "Optimizer:" in caplog.text

    def test_summary_includes_optimizer_name(self, methane, caplog):
        """Summary shows the configured optimizer name."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "sequential",
                "batch_optimizer": "lbfgs",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.INFO, logger="gpuma.utils.logging_utils"):
            optimize_structure_batch([methane], config)
        assert "Optimizer:           lbfgs" in caplog.text


class TestValidation:
    """Input validation for optimize_structure_batch."""

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert optimize_structure_batch([]) == []

    def test_mismatched_coords(self):
        """Mismatched symbols/coordinates raises ValueError."""
        bad = Structure(
            symbols=["C", "H"],
            coordinates=[(0, 0, 0)],
            charge=0,
            multiplicity=1,
        )
        with pytest.raises(ValueError, match="symbols/coords length mismatch"):
            optimize_structure_batch([bad])

    def test_empty_structure(self):
        """Zero-atom structure raises ValueError."""
        empty = Structure(symbols=[], coordinates=[], charge=0, multiplicity=1)
        with pytest.raises(ValueError, match="empty structure"):
            optimize_structure_batch([empty])
