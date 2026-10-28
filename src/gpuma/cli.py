"""Command Line Interface for GPUMA.

This module provides a command-line interface for molecular geometry
optimization using Fairchem UMA, ORB-v3 or SevenNet models.  The CLI supports
three main optimization modes:

1. Single Structure Optimization: Optimize individual molecular structures.
2. Ensemble Optimization (SMILES): Optimize multiple conformers generated
   from a SMILES string using batch inference.
3. Batch Optimization (Files): Optimize multiple structures from multi-XYZ
   files or directories.

The model backend is selected via the ``model_type`` configuration key
(``"fairchem"``/``"uma"``, ``"orb"``/``"orb-v3"``, or ``"sevennet"``/``"7net"``).
The ``--model-type`` CLI flag can override it for any backend
(``fairchem``/``uma``, ``orb``/``orb-v3``, ``sevennet``/``7net``).
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict

from .agent_skills import install_skill
from .api import (
    _optimize_ensemble_smiles,
    optimize_batch_multi_xyz_file,
    optimize_batch_xyz_directory,
    optimize_single_smiles,
    optimize_single_xyz_file,
)
from .cli_report import (
    EXIT_FAILURE,
    EXIT_INVALID_INPUT,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_USAGE,
    CommandResult,
    build_summary,
    exit_code_for,
    redacted_config,
    stdout_to_stderr,
    structure_record,
)
from .config import (
    Config,
    check_d3_allowed,
    check_model_cache_dir,
    check_model_path,
    load_config_from_file,
    resolve_model_type,
    save_config_to_file,
)
from .config_schema import config_json_schema
from .doctor import recommended_device, run_checks
from .models.base import AVAILABLE_FAIRCHEM_MODELS, AVAILABLE_SEVENNET_MODELS
from .models.device import _parse_device_string
from .structure import Structure, check_spin_state
from .utils.io_handler import (
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from .utils.logging_utils import configure_logging
from .utils.reporting import diagnostic_report

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class _UsageError(SystemExit):
    """An argparse usage error; still a ``SystemExit(2)``, but it keeps the message."""

    def __init__(self, message: str) -> None:
        super().__init__(EXIT_USAGE)
        self.message = message


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose errors main() can report in the ``--json`` summary."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise _UsageError(message)


def _level_from_string(name: str) -> int:
    """Map a log level name to its numeric value.

    Parameters
    ----------
    name:
        Name of the logging level (e.g. ``"INFO"``).

    Returns
    -------
    int
        Corresponding :mod:`logging` module integer level. Defaults to
        :data:`logging.INFO` if the name is unknown.

    """
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return levels.get((name or "").upper(), logging.INFO)


def _global_options_parser(defaults: bool = False) -> argparse.ArgumentParser:
    """Return the options accepted before or after any subcommand.

    Subparsers get ``SUPPRESS`` defaults so they never clobber a value given
    before the subcommand; the top-level parser carries the real defaults.
    """
    parser = argparse.ArgumentParser(add_help=False)
    flag_default = False if defaults else argparse.SUPPRESS
    value_default = None if defaults else argparse.SUPPRESS
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=flag_default,
        help="Enable verbose output (overrides config setting)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=flag_default,
        help="Suppress output (overrides config setting)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=value_default,
        help=(
            "Override compute device from config. "
            "Accepted values: 'cpu', 'cuda', 'cuda:N' (e.g. 'cuda:0')."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=value_default,
        choices=["fairchem", "uma", "orb", "orb-v3", "sevennet", "7net"],
        help=(
            "Override model backend from config. "
            "'fairchem'/'uma' for Fairchem UMA models, 'orb'/'orb-v3' for ORB-v3 models, "
            "'sevennet'/'7net' for SevenNet models."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=flag_default,
        help="Print a single JSON summary to stdout; logs go to stderr",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=flag_default,
        help="Validate inputs, config and device without loading a model",
    )
    return parser


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser for GPUMA.

    Returns
    -------
    argparse.ArgumentParser
        Configured :class:`ArgumentParser` instance with all subcommands and
        options.

    """
    parser = _Parser(
        parents=[_global_options_parser(defaults=True)],
        description=(
            "GPUMA - Optimize molecular structures using Fairchem UMA, ORB-v3 or SevenNet models"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OPTIMIZATION MODES:

1. Single Structure Optimization:
   Optimize individual molecular structures from SMILES or XYZ files.

2. Ensemble Optimization (SMILES):
   Optimize multiple conformers of the same molecule generated from SMILES
   using batch inference.

3. Batch Optimization (Files):
   Optimize multiple structures from a multi-XYZ file or a directory of
   XYZ files.

UTILITY COMMANDS:

4. Structure Conversion:
   Convert SMILES to 3D coordinates without optimization.

5. Conformer Generation:
   Generate multiple conformers from SMILES without optimization.

6. Configuration Management:
   Create default configuration files or validate existing ones.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    common = [_global_options_parser()]

    # Single structure optimization
    optimize_parser = subparsers.add_parser(
        "optimize",
        parents=common,
        help="Optimize a single molecular structure",
        description=("Optimize a single molecular structure from a SMILES string or an XYZ file."),
    )
    optimize_input = optimize_parser.add_mutually_exclusive_group(required=True)
    optimize_input.add_argument(
        "--smiles",
        type=str,
        help="SMILES string of the molecule to optimize",
    )
    optimize_input.add_argument(
        "--xyz",
        type=str,
        help="Path to XYZ file containing the structure to optimize",
    )
    optimize_parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total charge for XYZ input (ignored for SMILES). Default: from config",
    )
    optimize_parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity for XYZ or SMILES input. Default: from config",
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path for optimized structure",
    )
    optimize_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # 'smiles' alias for single-structure optimization from SMILES only
    smiles_parser = subparsers.add_parser(
        "smiles",
        parents=common,
        help="Optimize a single structure from a SMILES string",
        description=("Shorthand for optimizing a single structure directly from a SMILES string."),
    )
    smiles_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string of the molecule to optimize",
    )
    smiles_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path for optimized structure",
    )
    smiles_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Ensemble optimization (conformers of same molecule from SMILES)
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        parents=common,
        help="Optimize conformer ensemble from SMILES using batch inference",
        description=(
            "Optimize multiple conformers of the same molecule generated from "
            "SMILES efficiently using batch inference."
        ),
    )
    ensemble_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to generate and optimize conformers",
    )
    ensemble_parser.add_argument(
        "--conformers",
        type=int,
        default=None,
        help="Number of conformers to generate from SMILES (default: from config)",
    )
    ensemble_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for optimized ensemble",
    )
    ensemble_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Batch optimization (structures from files)
    batch_parser = subparsers.add_parser(
        "batch",
        parents=common,
        help="Optimize multiple structures from multi-XYZ file or directory",
        description=(
            "Optimize multiple structures using batch inference from a "
            "multi-XYZ file or a directory of XYZ files."
        ),
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        "--multi-xyz",
        type=str,
        help="Multi-structure XYZ file containing structures to optimize",
    )
    batch_input.add_argument(
        "--xyz-dir",
        type=str,
        help="Directory containing XYZ files of structures to optimize",
    )
    batch_parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total charge for XYZ inputs. Default: from config",
    )
    batch_parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity for XYZ inputs. Default: from config",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for optimized structures",
    )
    batch_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # SMILES to XYZ conversion (utility)
    convert_parser = subparsers.add_parser(
        "convert",
        parents=common,
        help="Convert SMILES to 3D coordinates (no optimization)",
        description="Convert SMILES strings to XYZ format with 3D coordinates.",
    )
    convert_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to convert to XYZ format",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path",
    )
    convert_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Conformer generation (utility)
    generate_parser = subparsers.add_parser(
        "generate",
        parents=common,
        help="Generate conformer ensemble from SMILES (no optimization)",
        description="Generate multiple conformers from SMILES without optimization.",
    )
    generate_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to generate conformers from",
    )
    generate_parser.add_argument(
        "--conformers",
        type=int,
        default=None,
        help="Number of conformers to generate (default: from config)",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for conformer ensemble",
    )
    generate_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Configuration management
    config_parser = subparsers.add_parser(
        "config",
        parents=common,
        help="Configuration file management",
        description=(
            "Create, validate or show configuration files, or print the config JSON Schema."
        ),
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--create",
        type=str,
        help="Write the built-in defaults to a new file (refuses to overwrite)",
    )
    config_group.add_argument(
        "--validate",
        type=str,
        help="Validate existing configuration file",
    )
    config_group.add_argument(
        "--show",
        action="store_true",
        help="Print the resolved configuration (after --config and overrides) as JSON",
    )
    config_group.add_argument(
        "--schema",
        action="store_true",
        help="Print the JSON Schema of the configuration file",
    )
    config_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file for --show (default: config.json if exists)",
    )

    # Environment checks
    doctor_parser = subparsers.add_parser(
        "doctor",
        parents=common,
        help="Check GPUs, device, Hugging Face token, model cache and agent skill",
        description=(
            "Check the environment without loading a model: CUDA, free GPU memory, the "
            "configured device, the Hugging Face token (UMA), the model cache and the "
            "installed agent skill. Exits 1 if any check fails."
        ),
    )
    doctor_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Agent skill installation
    skills_parser = subparsers.add_parser(
        "skills",
        parents=common,
        help="Install the gpuma skill for AI coding agents",
        description=(
            "Install the gpuma Agent Skill into ~/.claude/skills (Claude Code) and "
            "~/.agents/skills (Codex, Gemini CLI, Copilot, Cursor)."
        ),
    )
    skills_actions = skills_parser.add_subparsers(dest="skills_action", required=True)
    install_parser = skills_actions.add_parser(
        "install",
        parents=common,
        help="Install or update the skill",
        description="Copy the skill shipped with this gpuma version into the agents' folders.",
    )
    install_parser.add_argument(
        "--target",
        action="append",
        help="Skills folder to install into (repeatable; default: the two user folders)",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a gpuma skill folder that was not installed by gpuma",
    )

    return parser


def _cli_override(args, attr: str, config_val: int) -> int:
    """Return CLI arg if provided, otherwise the config value."""
    cli_val = getattr(args, attr, None)
    return int(cli_val) if cli_val is not None else int(config_val)


def _resolve_conformer_count(args, config: Config) -> int:
    """Return the conformer count, preferring an explicit ``--conformers``.

    Tests ``is not None`` rather than truthiness: ``--conformers 0`` used to
    fall through to the config value, quietly generating conformers after the
    user asked for none. Zero is rejected instead, where the message names the
    flag that caused it.
    """
    requested = getattr(args, "conformers", None)
    if requested is None:
        return int(config.conformer_generation.max_num_conformers)
    if requested <= 0:
        raise ValueError(f"--conformers must be positive, got {requested}")
    return int(requested)


def cmd_optimize(args, config: Config) -> CommandResult:
    """Handle the single-structure optimization command.

    This command optimizes a single molecular structure from either a SMILES
    string or an XYZ file using the configured model backend.
    """
    if not args.smiles:
        config.optimization.charge = _cli_override(args, "charge", config.optimization.charge)
    config.optimization.multiplicity = _cli_override(
        args, "multiplicity", config.optimization.multiplicity
    )
    if args.dry_run:
        return _dry_run_result("optimize", args, config)
    _check_inputs(args, config)
    if args.smiles:
        logger.info(
            "Converting SMILES '%s' to 3D coordinates and optimizing (multiplicity=%d)...",
            args.smiles,
            config.optimization.multiplicity,
        )
        optimized = optimize_single_smiles(
            smiles=args.smiles,
            output_file=args.output,
            config=config,
        )
    else:
        logger.info(
            "Reading and optimizing structure from %s (charge=%d, multiplicity=%d)",
            args.xyz,
            config.optimization.charge,
            config.optimization.multiplicity,
        )
        optimized = optimize_single_xyz_file(
            input_file=args.xyz,
            output_file=args.output,
            config=config,
        )

    logger.info(
        "Optimization completed. Final energy: %.6f eV",
        optimized.energy if optimized.energy is not None else float("nan"),
    )
    logger.info("Optimized structure saved to %s", args.output)
    return _optimized_result("optimize", args, [optimized], n_input=1, n_expected=1)


def cmd_ensemble(args, config: Config) -> CommandResult:
    """Handle conformer ensemble optimization from SMILES.

    This command optimizes multiple conformers of the same molecule using
    batch inference.
    """
    if args.dry_run:
        return _dry_run_result("ensemble", args, config)
    _check_inputs(args, config)
    num_conf = _resolve_conformer_count(args, config)
    logger.info("Generating %d conformers for SMILES: %s", num_conf, args.smiles)
    config.conformer_generation.max_num_conformers = num_conf

    optimized_conformers, n_generated = _optimize_ensemble_smiles(
        smiles=args.smiles,
        output_file=args.output,
        config=config,
    )

    if not optimized_conformers:
        raise RuntimeError("No conformers were optimized")

    logger.info(
        "Ensemble optimization finished: %d optimized conformers saved to %s",
        len(optimized_conformers),
        args.output,
    )
    return _optimized_result(
        "ensemble",
        args,
        optimized_conformers,
        n_input=num_conf,
        n_expected=n_generated,
        extra={"n_generated": n_generated},
    )


def _optimized_result(
    command: str,
    args,
    structures: list[Structure],
    n_input: int,
    n_expected: int,
    extra: dict | None = None,
) -> CommandResult:
    """Return ``ok``, or ``partial`` when a structure failed or didn't converge."""
    n_failed = n_expected - len(structures)
    n_unconverged = sum(s.converged is False for s in structures)
    if n_failed:
        logger.warning(
            "Only %d of %d structures were optimized; see the log for the failures.",
            len(structures),
            n_expected,
        )
    if n_unconverged:
        logger.warning("%d structure(s) did not converge within max_steps.", n_unconverged)
    partial = n_failed or n_unconverged
    return CommandResult(
        command,
        "partial" if partial else "ok",
        EXIT_PARTIAL if partial else EXIT_OK,
        output_file=args.output,
        n_input=n_input,
        structures=[structure_record(i, s) for i, s in enumerate(structures)],
        extra=extra or {},
    )


_DRY_RUN_COMMANDS = ("optimize", "smiles", "ensemble", "batch")


def _check_smiles(smiles: str, multiplicity: int) -> None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    symbols = [atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms()]
    check_spin_state(symbols, Chem.GetFormalCharge(mol), multiplicity)


_DEFAULT_MODEL_NAMES = {
    "fairchem": "uma-s-1p2p1-ft-d4rev",
    "orb": "orb_v3_direct_omol",
    "sevennet": "7net-omni",
}


def _known_model_names(model_type: str) -> set[str]:
    """Return the model names the backend can load by name."""
    if model_type == "orb":
        from orb_models.forcefield import pretrained

        return {name for name in dir(pretrained) if not name.startswith("_")}
    if model_type == "fairchem":
        from fairchem.core import pretrained_mlip

        # gpuma runs the omol task only; r2scan, OC25 and ODAC25 models lack it.
        omol = {name for name in pretrained_mlip.available_models if name.endswith("-omol")}
        return set(AVAILABLE_FAIRCHEM_MODELS) | omol
    return set(AVAILABLE_SEVENNET_MODELS)


def _check_model(config: Config) -> None:
    """Reject a model name the configured backend cannot load (unless a checkpoint is given)."""
    model_type = resolve_model_type(config)
    if config.model.model_path:
        return
    name = config.model.model_name
    if name not in _known_model_names(model_type):
        raise ValueError(
            f"Unsupported {model_type} model_name {name!r}; for model_type {model_type!r} "
            f"use e.g. {_DEFAULT_MODEL_NAMES[model_type]!r}."
        )


def _check_inputs(args, config: Config) -> int:
    """Validate inputs, model and device without loading a model; return ``n_input``."""
    # CLI overrides apply after Config validation, so re-check what they can change.
    check_model_path(config)
    check_model_cache_dir(config)
    check_d3_allowed(config)
    _check_model(config)
    _parse_device_string(str(config.technical.device))
    charge = int(config.optimization.charge)
    multiplicity = int(config.optimization.multiplicity)
    if args.command in ("optimize", "smiles", "ensemble") and args.smiles:
        _check_smiles(args.smiles, multiplicity)
        return _resolve_conformer_count(args, config) if args.command == "ensemble" else 1
    if args.command in ("optimize", "smiles"):
        structures = [read_xyz(args.xyz, charge=charge, multiplicity=multiplicity)]
    elif args.multi_xyz:
        structures = read_multi_xyz(args.multi_xyz, charge=charge, multiplicity=multiplicity)
    else:
        structures = read_xyz_directory(args.xyz_dir, charge=charge, multiplicity=multiplicity)
    if not structures:
        raise ValueError("No input structures found")
    for i, s in enumerate(structures):
        try:
            check_spin_state(s.symbols, s.charge, s.multiplicity)
        except ValueError as exc:
            raise ValueError(f"structure {i + 1}: {exc}") from exc
    return len(structures)


def _dry_run_result(command: str, args, config: Config) -> CommandResult:
    n_input = _check_inputs(args, config)
    logger.info("Dry run: %d input structure(s) valid; no model loaded.", n_input)
    return CommandResult(command, "dry_run", EXIT_OK, output_file=args.output, n_input=n_input)


def cmd_batch(args, config: Config) -> CommandResult:
    """Handle batch optimization from files (multi-XYZ or directory)."""
    eff_charge = _cli_override(args, "charge", config.optimization.charge)
    config.optimization.charge = eff_charge
    eff_mult = _cli_override(args, "multiplicity", config.optimization.multiplicity)
    config.optimization.multiplicity = eff_mult
    if args.dry_run:
        return _dry_run_result("batch", args, config)
    n_input = _check_inputs(args, config)

    if args.multi_xyz:
        logger.info(
            "Reading structures from multi-XYZ file: %s (charge=%d, multiplicity=%d)",
            args.multi_xyz,
            eff_charge,
            eff_mult,
        )

        structures = optimize_batch_multi_xyz_file(
            input_file=args.multi_xyz,
            output_file=args.output,
            config=config,
        )
    else:
        logger.info(
            "Reading structures from directory: %s (charge=%d, multiplicity=%d)",
            args.xyz_dir,
            eff_charge,
            eff_mult,
        )
        structures = optimize_batch_xyz_directory(
            input_directory=args.xyz_dir,
            output_file=args.output,
            config=config,
        )

    if not structures:
        raise RuntimeError("No structures were optimized")

    logger.info("Optimized structures saved to %s", args.output)
    return _optimized_result("batch", args, structures, n_input=n_input, n_expected=n_input)


def cmd_convert(args, config: Config) -> CommandResult:
    """Handle the SMILES to XYZ conversion command.

    This command generates a single 3D structure from SMILES without running
    any optimization. ``config`` selects the embedding device.
    """
    _parse_device_string(str(config.technical.device))
    logger.info("Converting SMILES '%s' to XYZ without optimization", args.smiles)
    structure = smiles_to_xyz(args.smiles, config=config)
    save_xyz_file(structure, args.output)
    logger.info("Structure saved to %s", args.output)
    return CommandResult(
        "convert",
        "ok",
        EXIT_OK,
        output_file=args.output,
        n_input=1,
        structures=[structure_record(0, structure)],
    )


def cmd_generate(args, config: Config) -> CommandResult:
    """Handle conformer generation from SMILES.

    This command generates a conformer ensemble from a SMILES string without
    optimization.
    """
    _parse_device_string(str(config.technical.device))
    num_conf = _resolve_conformer_count(args, config)
    logger.info(
        "Generating %d conformers for SMILES (no optimization): %s",
        num_conf,
        args.smiles,
    )
    # multiplicity was omitted here, so smiles_to_ensemble fell back to its
    # own default of 1 and silently ignored the configured value -- unlike
    # the 'ensemble' command, which passes it.
    structures = smiles_to_ensemble(
        args.smiles,
        num_conf,
        multiplicity=int(config.optimization.multiplicity),
        config=config,
    )
    comments = [
        f"Generated conformer {i + 1} from SMILES: {args.smiles}" for i in range(len(structures))
    ]
    save_multi_xyz(structures, args.output, comments)
    logger.info("Generated conformers saved to %s", args.output)
    return CommandResult(
        "generate",
        "ok",
        EXIT_OK,
        output_file=args.output,
        n_input=num_conf,
        structures=[structure_record(i, s) for i, s in enumerate(structures)],
    )


def cmd_config(args, config: Config) -> CommandResult:
    """Handle configuration file management commands."""
    if args.show or args.schema:
        # With --json the summary carries the config; the schema goes in "schema".
        data = redacted_config(config) if args.show else config_json_schema()
        if not args.json:
            print(json.dumps(data, indent=2))
        extra = {"schema": data} if args.schema else {}
        return CommandResult("config", "ok", EXIT_OK, extra=extra)
    if args.create:
        if os.path.exists(args.create):
            raise FileExistsError(f"{args.create} already exists; remove it or pick another path")
        logger.info("Creating default configuration file at %s", args.create)
        save_config_to_file(Config(), args.create)
        logger.info("Configuration file created")
        return CommandResult("config", "ok", EXIT_OK, output_file=args.create)
    if not os.path.exists(args.validate):
        raise FileNotFoundError(f"Configuration file not found: {args.validate}")
    logger.info("Validating configuration file at %s", args.validate)
    _ = load_config_from_file(args.validate)
    logger.info("Configuration file is valid")
    return CommandResult("config", "ok", EXIT_OK)


def _apply_global_verbosity_flags(config: Config, verbose: bool, quiet: bool) -> None:
    """Apply global verbosity CLI flags to the configuration in-place."""
    if verbose:
        config.technical.logging_level = "DEBUG"
    elif quiet:
        config.technical.logging_level = "ERROR"


def _apply_device_override(config: Config, device) -> None:
    """Apply a global device override if provided via CLI."""
    if device:
        config.technical.device = device


def _apply_model_type_override(config: Config, model_type) -> None:
    """Apply a global model-type override if provided via CLI."""
    if model_type:
        config.model.model_type = model_type


def _resolve_config_path(args, parser: argparse.ArgumentParser) -> tuple[str, bool]:
    """Return the config path to load and whether built-in defaults stand in for it.

    An explicit ``--config`` that does not exist is a usage error, so a typo
    cannot silently swap in the default model.
    """
    explicit = getattr(args, "config", None)
    if explicit is not None:
        if not os.path.exists(explicit):
            parser.error(f"config file not found: {explicit}")
        return explicit, False
    return "config.json", not os.path.exists("config.json")


def cmd_doctor(args, config: Config) -> CommandResult:
    """Handle the environment checks."""
    checks = run_checks(config)
    for check in checks:
        log = logger.error if check.status == "fail" else logger.info
        log("%-12s %-4s %s", check.name, check.status.upper(), check.detail)
        if check.hint and check.status != "ok":
            log("%-12s      -> %s", "", check.hint)
    device = recommended_device(checks)
    if device:
        logger.info("Recommended device: %s", device)
    extra = {
        "checks": [asdict(c) for c in checks],
        "recommended_device": device,
    }
    failed = [c.name for c in checks if c.status == "fail"]
    if failed:
        message = f"failing checks: {', '.join(failed)}"
        return CommandResult(
            "doctor",
            "error",
            EXIT_FAILURE,
            error={"type": "DoctorFailure", "message": message},
            extra=extra,
        )
    return CommandResult("doctor", "ok", EXIT_OK, extra=extra)


def cmd_skills(args, config: Config | None) -> CommandResult:
    """Handle installation of the agent skill."""
    installed = install_skill(args.target, force=args.force)
    for path in installed:
        logger.info("Installed gpuma skill to %s", path)
    return CommandResult("skills", "ok", EXIT_OK, extra={"installed": [str(p) for p in installed]})


_HANDLERS = {
    "optimize": cmd_optimize,
    "smiles": cmd_optimize,
    "ensemble": cmd_ensemble,
    "batch": cmd_batch,
    "convert": cmd_convert,
    "generate": cmd_generate,
    "config": cmd_config,
    "doctor": cmd_doctor,
    "skills": cmd_skills,
}

# Commands that never read a config file, so a broken ./config.json cannot block them.
_NO_CONFIG_COMMANDS = ("skills",)


def _load_config(args, parser: argparse.ArgumentParser) -> Config:
    """Load the config and apply the global CLI overrides and logging level."""
    cfg_path, using_defaults = _resolve_config_path(args, parser)
    config = load_config_from_file(None if using_defaults else cfg_path)

    _apply_global_verbosity_flags(config, args.verbose, args.quiet)
    _apply_device_override(config, args.device)
    _apply_model_type_override(config, args.model_type)
    configure_logging(_level_from_string(config.technical.logging_level))

    if using_defaults and args.command != "config":
        logger.warning(
            "No --config given and no config.json in the working directory; "
            "using built-in defaults (model %s / %s).",
            config.model.model_type,
            config.model.model_name,
        )
    return config


def _error_result(command: str, exc: Exception) -> CommandResult:
    """Log ``exc`` and return the matching error result."""
    code = exit_code_for(exc)
    if code == EXIT_INVALID_INPUT:
        logger.error("gpuma %s: %s", command, exc)
    else:
        logger.exception("gpuma %s failed: %s", command, exc)
    return CommandResult(
        command, "error", code, error={"type": type(exc).__name__, "message": str(exc)}
    )


def _execute(args, parser: argparse.ArgumentParser) -> tuple[CommandResult, Config | None]:
    """Load the config and run the command, turning exceptions into results."""
    config = None
    try:
        if args.command in _NO_CONFIG_COMMANDS:
            configure_logging(logging.DEBUG if args.verbose else logging.INFO)
        else:
            config = _load_config(args, parser)
        return _HANDLERS[args.command](args, config), config
    except Exception as exc:
        return _error_result(args.command, exc), config


def main(argv=None) -> int:
    """Entry point for the ``gpuma`` CLI.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments. If not provided, uses
        :data:`sys.argv`.

    Returns
    -------
    int
        Exit status code (see :mod:`gpuma.cli_report`).

    """
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        return _main(argv)
    except _UsageError as exc:
        if "--json" in argv:
            command = next((a for a in argv if a in _HANDLERS), None)
            error = {"type": "UsageError", "message": exc.message}
            result = CommandResult(command, "error", EXIT_USAGE, error=error)
            print(json.dumps(build_summary(result, None, 0.0, None)), flush=True)
        raise


def _main(argv: list[str]) -> int:
    """Parse ``argv`` and run the command; return the exit code."""
    parser = setup_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1
    if args.dry_run and args.command not in _DRY_RUN_COMMANDS:
        parser.error(f"--dry-run applies to {', '.join(_DRY_RUN_COMMANDS)} only")

    if not args.json:
        result, _ = _execute(args, parser)
        return result.exit_code

    start = time.perf_counter()
    with diagnostic_report() as report, stdout_to_stderr():
        result, config = _execute(args, parser)
    summary = build_summary(result, config, time.perf_counter() - start, report.to_dict())
    print(json.dumps(summary), flush=True)
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
