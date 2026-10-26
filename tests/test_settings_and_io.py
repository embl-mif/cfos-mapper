import json

import numpy as np
import pandas as pd
import pytest

import cfos_mapper as cf
from conftest import SHAPE, VOXEL


def test_load_settings_fills_in_folders(project):
    results = project["results_folder"]
    assert project["networks_folder"].endswith("networks")
    assert project["regions_file"].endswith("regions.json")
    assert project["atlas"]["name"] == "allen_mouse_25um"
    assert project["atlas"]["folder"].startswith(results.rsplit("results", 1)[0])


def test_settings_found_from_environment(project, monkeypatch, tmp_path):
    monkeypatch.setenv("CFOS_MAPPER_SETTINGS", project["settings_file"])
    monkeypatch.chdir(tmp_path.parent)
    assert cf.load_settings()["settings_file"] == project["settings_file"]


def test_missing_settings_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CFOS_MAPPER_SETTINGS", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="cfos-mapper init"):
        cf.load_settings()


def test_regions_saved_with_comments_kept(project):
    regions = cf.load_regions(project)
    assert list(regions) == ["B1"]
    regions["B1"]["regionB"] = {"z": [0, 10], "y": [0, 10], "x": [0, 10]}
    cf.save_regions(regions, project)
    raw = json.load(open(project["regions_file"]))
    assert raw["_comment"] == "test" and "regionB" in raw["B1"]


def test_write_templates(tmp_path):
    written = cf.write_templates(tmp_path / "new")
    assert sorted(p.name for p in written) == ["regions.json", "settings.json"]
    assert cf.write_templates(tmp_path / "new") == []  # existing files are kept


def test_open_brain_and_channels(project):
    assert cf.brain_names(project) == ["B1"]
    brain = cf.open_brain("B1", project)
    assert cf.voxel_size(brain) == VOXEL
    cfos = cf.load_into_memory(cf.get_channel(brain, project["signal_channel"]))
    assert cfos.shape == SHAPE and cfos.dtype == np.uint16
    with pytest.raises(ValueError):
        cf.get_channel(brain, "missing-channel")


def test_save_region(project, tmp_path):
    brain = cf.open_brain("B1", project)
    region = cf.save_region(brain, (5, 15), (10, 30), (20, 60), tmp_path / "r.ome.zarr")
    cfos = cf.load_into_memory(cf.get_channel(region, "cFos"))
    full = cf.load_into_memory(cf.get_channel(brain, "cFos"))
    np.testing.assert_array_equal(cfos, full[5:15, 10:30, 20:60])


def test_density_map(project, tmp_path):
    brain = cf.open_brain("B1", project)
    cells = pd.DataFrame({"z": [1, 2, 30], "y": [1, 2, 100], "x": [1, 2, 100]})
    density = cf.save_density_map(cells, brain, tmp_path / "d.ome.zarr", bin_um=30)
    assert density.sum() == 3 and density.max() == 2
