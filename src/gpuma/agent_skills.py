"""Install the packaged gpuma Agent Skill into AI agents' skill folders."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from .cli_report import gpuma_version

SKILL_NAME = "gpuma"
VERSION_FILE = ".gpuma-version"
# Claude Code reads ~/.claude/skills; Codex, Gemini CLI, Copilot and Cursor read ~/.agents/skills.
DEFAULT_TARGETS = (Path.home() / ".claude" / "skills", Path.home() / ".agents" / "skills")


def packaged_skill_dir() -> Traversable:
    """Return the skill folder shipped inside the gpuma package."""
    return files("gpuma") / "skills" / SKILL_NAME


def install_skill(targets: Sequence[Path] | None = None, force: bool = False) -> list[Path]:
    """Copy the packaged skill to ``<target>/gpuma`` for each target; return the folders.

    A folder that gpuma did not install (no version file) is only replaced with
    ``force``; otherwise :class:`FileExistsError` is raised before anything changes.
    """
    dests = [Path(t).expanduser() / SKILL_NAME for t in (targets or DEFAULT_TARGETS)]
    for dest in dests:
        foreign = (dest.exists() or dest.is_symlink()) and not (dest / VERSION_FILE).is_file()
        if foreign and not force:
            raise FileExistsError(
                f"{dest} exists and was not installed by gpuma; pass --force to replace it"
            )
    with as_file(packaged_skill_dir()) as source:
        for dest in dests:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
            (dest / VERSION_FILE).write_text(f"{gpuma_version() or 'unknown'}\n")
    return dests


def installed_skill_versions(targets: Sequence[Path] | None = None) -> dict[Path, str | None]:
    """Return the gpuma version each target's installed skill came from, or ``None``."""
    versions = {}
    for target in targets or DEFAULT_TARGETS:
        stamp = Path(target).expanduser() / SKILL_NAME / VERSION_FILE
        versions[Path(target)] = stamp.read_text().strip() if stamp.is_file() else None
    return versions
