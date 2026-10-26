import subprocess
import sys

import pytest

from cfos_mapper.cli import main, parser


def test_help_lists_every_step():
    text = parser().format_help()
    for step in ["init", "napari", "prepare", "prepare-all", "predict", "predict-all", "train", "run-brain",
                 "register", "count", "summary"]:
        assert step in text


def test_settings_accepted_before_and_after_the_step():
    assert parser().parse_args(["--settings", "a.json", "prepare", "B", "R"]).settings == "a.json"
    assert parser().parse_args(["prepare", "B", "R", "--settings", "a.json"]).settings == "a.json"
    assert parser().parse_args(["prepare", "B", "R"]).settings is None


def test_init_writes_templates(tmp_path, capsys):
    assert main(["init", str(tmp_path / "p")]) == 0
    assert (tmp_path / "p" / "settings.json").exists()


def test_version():
    assert main(["--version"]) == 0


def test_prepare_from_the_command_line(project):
    """The same command the napari plugin runs in the background."""
    result = subprocess.run([sys.executable, "-m", "cfos_mapper", "--settings", project["settings_file"],
                             "prepare", "B1", "regionA"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "done" in result.stdout


def test_unknown_brain_fails(project):
    with pytest.raises(KeyError):
        main(["--settings", project["settings_file"], "prepare", "NOPE", "regionA"])
