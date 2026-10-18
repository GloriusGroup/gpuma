"""The packaged skill must only name commands, flags and config keys that exist."""

import re
from pathlib import Path

import pytest
import yaml

from gpuma.cli import setup_parser
from gpuma.config import DEFAULT_CONFIG

SKILL_DIR = Path(__file__).resolve().parent.parent / "src" / "gpuma" / "skills" / "gpuma"
SKILL_MD = SKILL_DIR / "SKILL.md"
FILES = [SKILL_MD, *sorted((SKILL_DIR / "references").glob("*.md"))]
SPAN = re.compile(r"`([^`\n]+)`")
GPUMA_COMMAND = re.compile(r"^(?:\S+=\S+ |nohup )*gpuma\b")


def _frontmatter():
    _, header, _ = SKILL_MD.read_text().split("---\n", 2)
    return yaml.safe_load(header)


def _subparser_maps(parser):
    return [a.choices for a in parser._actions if isinstance(a.choices, dict)]


def _parsers(parser):
    yield parser
    for choices in _subparser_maps(parser):
        for sub in choices.values():
            yield from _parsers(sub)


def _subcommands():
    return {name for p in _parsers(setup_parser()) for c in _subparser_maps(p) for name in c}


def _all_flags():
    return {
        opt for p in _parsers(setup_parser()) for a in p._actions for opt in a.option_strings
    }


def _spans():
    """Yield inline code spans and command lines inside fenced code blocks."""
    for path in FILES:
        text = path.read_text()
        for span in SPAN.findall(text):
            yield path.name, span
        for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
            for line in block.splitlines():
                if GPUMA_COMMAND.match(line.strip()):
                    yield path.name, line.strip()


def test_frontmatter():
    meta = _frontmatter()
    assert meta["name"] == SKILL_DIR.name
    assert 0 < len(meta["description"]) <= 1024


def test_skill_md_is_short():
    assert len(SKILL_MD.read_text().splitlines()) <= 500


def test_references_are_linked_and_exist():
    linked = set(re.findall(r"\]\((references/[^)]+)\)", SKILL_MD.read_text()))
    assert linked, "SKILL.md links no reference files"
    for rel in linked:
        assert (SKILL_DIR / rel).is_file(), rel
    on_disk = {f"references/{p.name}" for p in FILES[1:]}
    assert on_disk == linked


@pytest.mark.parametrize(("source", "span"), list(_spans()))
def test_named_cli_elements_exist(source, span):
    subcommands, flags = _subcommands(), _all_flags()
    if GPUMA_COMMAND.match(span):
        for word in re.findall(r"\bgpuma ([a-z][a-z-]*)", span):
            if not word.startswith("-"):
                assert word in subcommands, f"{source}: unknown subcommand in `{span}`"
    if GPUMA_COMMAND.match(span) or span.startswith("--"):
        for flag in re.findall(r"(?<![\w-])(--?[a-zA-Z][a-z-]*)", span):
            assert flag in flags, f"{source}: unknown flag {flag} in `{span}`"
    key = re.fullmatch(r"(optimization|model|conformer_generation|technical)\.([a-z_]+)", span)
    if key:
        assert key.group(2) in DEFAULT_CONFIG[key.group(1)], f"{source}: unknown key `{span}`"


def test_no_job_scheduler_mentions():
    for path in FILES:
        assert not re.search(r"slurm|sbatch|srun", path.read_text(), re.IGNORECASE), path.name
