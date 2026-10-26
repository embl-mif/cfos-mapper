"""
A tiny synthetic project for the tests: one brain (two channels, a few bright
blobs in the cFos channel), settings.json and regions.json in a temporary folder.
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SHAPE = (40, 160, 160)       # z, y, x
VOXEL = (3.0, 1.5, 1.5)      # um
BLOBS = [(12, 40, 40), (20, 80, 100), (28, 120, 60), (20, 50, 120), (25, 110, 120)]


def synthetic_brain(seed=0):
    """(cfos, autofluorescence) uint16 images: noisy tissue with bright cell-sized blobs in cFos."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(SHAPE)
    cfos = rng.normal(200, 10, SHAPE)
    for bz, by, bx in BLOBS:  # ~10 um spheres
        distance2 = ((z - bz) * VOXEL[0]) ** 2 + ((y - by) * VOXEL[1]) ** 2 + ((x - bx) * VOXEL[2]) ** 2
        cfos += 3000 * np.exp(-distance2 / (2 * 4.0 ** 2))
    autofluorescence = rng.normal(500, 10, SHAPE)
    return np.clip(cfos, 0, 65535).astype(np.uint16), np.clip(autofluorescence, 0, 65535).astype(np.uint16)


@pytest.fixture
def project(tmp_path):
    """Settings of a synthetic project with brain 'B1' and one test region."""
    import cfos_mapper as cf

    brains = tmp_path / "brains"
    brains.mkdir()
    cfos, autofluorescence = synthetic_brain()
    cf.save_image(np.stack([autofluorescence, cfos]), VOXEL, brains / "B1.ome.zarr",
                  channel_names=["autofluo-488", "cFos-647"])

    settings = json.loads((cf.settings.TEMPLATES / "settings.json").read_text())
    settings.update(brains_folder=str(brains), results_folder=str(tmp_path / "project" / "results"))
    settings["find_candidates"]["n_free_cpus"] = 1
    settings["classify"]["cube_size"] = [20, 50, 50]
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "regions.json").write_text(json.dumps({
        "_comment": "test",
        "B1": {"regionA": {"z": [0, 40], "y": [0, 160], "x": [0, 160]}},
    }))
    return cf.load_settings(tmp_path / "settings.json")
