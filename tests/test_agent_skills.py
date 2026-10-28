"""Tests for installing the packaged agent skill."""

import pytest

from gpuma.agent_skills import (
    VERSION_FILE,
    install_skill,
    installed_skill_versions,
    packaged_skill_dir,
)
from gpuma.cli import main
from gpuma.cli_report import gpuma_version


def test_packaged_skill_present():
    assert (packaged_skill_dir() / "SKILL.md").is_file()


def test_install_writes_version(tmp_path):
    [d] = install_skill([tmp_path])
    assert d == tmp_path / "gpuma"
    assert (d / "SKILL.md").is_file() and (d / "references").is_dir()
    assert (d / VERSION_FILE).read_text().strip() == gpuma_version()


def test_install_refuses_foreign_skill(tmp_path):
    (tmp_path / "gpuma").mkdir()
    (tmp_path / "gpuma" / "SKILL.md").write_text("mine")
    with pytest.raises(FileExistsError):
        install_skill([tmp_path])
    assert (tmp_path / "gpuma" / "SKILL.md").read_text() == "mine"


def test_refusal_touches_no_target(tmp_path):
    ok, foreign = tmp_path / "a", tmp_path / "b"
    (foreign / "gpuma").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        install_skill([ok, foreign])
    assert not (ok / "gpuma").exists()


def test_force_replaces_foreign_skill(tmp_path):
    (tmp_path / "gpuma").mkdir()
    (tmp_path / "gpuma" / "SKILL.md").write_text("mine")
    install_skill([tmp_path], force=True)
    assert (tmp_path / "gpuma" / "SKILL.md").read_text() != "mine"


def test_reinstall_replaces_own_skill(tmp_path):
    [d] = install_skill([tmp_path])
    (d / "stale.md").write_text("old")
    install_skill([tmp_path])
    assert not (d / "stale.md").exists()


def test_installed_skill_versions(tmp_path):
    install_skill([tmp_path / "a"])
    versions = installed_skill_versions([tmp_path / "a", tmp_path / "b"])
    assert versions == {tmp_path / "a": gpuma_version(), tmp_path / "b": None}


def test_cli_skills_install(tmp_path):
    assert main(["skills", "install", "--target", str(tmp_path)]) == 0
    assert (tmp_path / "gpuma" / "SKILL.md").is_file()


def test_cli_skills_install_foreign_is_exit_3(tmp_path):
    (tmp_path / "gpuma").mkdir()
    assert main(["skills", "install", "--target", str(tmp_path)]) == 3


def test_cli_skills_ignores_broken_config_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text('{"optimization": {"batch_optimizer": "nope"}}')
    assert main(["skills", "install", "--target", str(tmp_path / "t")]) == 0


def test_install_without_package_metadata_writes_unknown(tmp_path, monkeypatch):
    import gpuma.agent_skills as agent_skills

    monkeypatch.setattr(agent_skills, "gpuma_version", lambda: None)
    [d] = install_skill([tmp_path])
    assert (d / VERSION_FILE).read_text().strip() == "unknown"
