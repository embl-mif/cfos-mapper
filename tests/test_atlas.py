"""
Cells -> atlas regions, with a tiny fake atlas and a fake registration
(deformation fields that map every voxel to itself), so no download is needed.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

import cfos_mapper as cf

# region tree: root (997) > brain (8) > A (1), B (2)
STRUCTURES = {
    997: {"acronym": "root", "name": "root", "structure_id_path": [997]},
    8: {"acronym": "grey", "name": "Basic cell groups", "structure_id_path": [997, 8]},
    1: {"acronym": "A", "name": "Region A", "structure_id_path": [997, 8, 1]},
    2: {"acronym": "B", "name": "Region B", "structure_id_path": [997, 8, 2]},
}
ATLAS_SHAPE = (10, 12, 14)   # atlas voxels, orientation "asr"
ATLAS_UM = 100.0


def fake_atlas():
    annotation = np.zeros(ATLAS_SHAPE, np.uint32)
    annotation[2:8, 2:10, 2:7] = 1      # region A on the right half
    annotation[2:8, 2:10, 7:12] = 2     # region B on the left half
    hemispheres = np.ones(ATLAS_SHAPE, np.uint8)  # 1 = left
    hemispheres[:, :, :7] = 2                      # x < 7 = right ("asr": x starts on the right)
    return SimpleNamespace(annotation=annotation, hemispheres=hemispheres, structures=STRUCTURES,
                           resolution=(ATLAS_UM,) * 3, metadata={"orientation": "asr"})


@pytest.fixture
def registered(project):
    """A fake brainreg output for brain B1: the brain IS the atlas (same orientation and voxel size)."""
    folder = cf.registration_folder("B1", project)
    (folder / "input").mkdir(parents=True)
    atlas = fake_atlas()
    tifffile.imwrite(folder / "downsampled.tiff", np.zeros(ATLAS_SHAPE, np.uint16))
    tifffile.imwrite(folder / "registered_atlas.tiff", atlas.annotation)
    tifffile.imwrite(folder / "registered_hemispheres.tiff", atlas.hemispheres)
    grid = np.indices(ATLAS_SHAPE).astype(np.float32) * ATLAS_UM / 1000  # atlas position in mm
    for axis in range(3):
        tifffile.imwrite(folder / f"deformation_field_{axis}.tiff", grid[axis])
    json.dump(dict(full_resolution_shape=list(ATLAS_SHAPE), full_resolution_voxel_size=[ATLAS_UM] * 3,
                   atlas="fake_atlas_100um", orientation="asr", level=0, voxel_size=[ATLAS_UM] * 3),
              open(folder / "input" / "input_info.json", "w"))
    return atlas


def test_registration_ready(project, registered):
    assert cf.registration_ready("B1", project)


def test_cells_to_atlas(project, registered):
    cells = pd.DataFrame({"z": [5, 5, 0, 50], "y": [5, 5, 0, 50], "x": [3, 9, 0, 50]})
    table = cf.cells_to_atlas(cells, "B1", project, atlas=registered)
    assert list(table.structure_id) == [1, 2, 0, 0]
    assert list(table.hemisphere) == ["right", "left", "outside", "outside"]


def test_cells_to_atlas_flipped_orientation(project, registered):
    """Same brain, but our images have x flipped ("asl"): a cell at x=3 is now at atlas x=10."""
    info_file = cf.registration_folder("B1", project) / "input" / "input_info.json"
    info = json.load(open(info_file))
    info["orientation"] = "asl"
    json.dump(info, open(info_file, "w"))
    cells = pd.DataFrame({"z": [5], "y": [5], "x": [3]})
    table = cf.cells_to_atlas(cells, "B1", project, atlas=registered)
    assert table.atlas_x.iloc[0] == ATLAS_SHAPE[2] - 1 - 3
    assert table.structure_id.iloc[0] == 2


def test_per_region_table_counts_parents(project, registered):
    cells = pd.DataFrame({"z": [5, 5, 6], "y": [5, 5, 6], "x": [3, 9, 9]})
    in_atlas = cf.cells_to_atlas(cells, "B1", project, atlas=registered)
    volumes = cf.region_volumes("B1", project)
    table = cf.per_region_table(in_atlas, volumes, registered).set_index("acronym")
    assert table.loc["A", "cells"] == 1 and table.loc["B", "cells"] == 2
    assert table.loc["grey", "cells"] == 3 and table.loc["root", "cells"] == 3
    voxel_mm3 = (ATLAS_UM / 1000) ** 3
    assert table.loc["A", "volume_mm3"] == pytest.approx(6 * 8 * 5 * voxel_mm3)
    assert table.loc["B", "cells_per_mm3"] == pytest.approx(2 / (6 * 8 * 5 * voxel_mm3))
    assert table.loc["B", "left_cells"] == 2 and table.loc["B", "right_cells"] == 0


def test_count_and_summary(project, registered, monkeypatch):
    run = cf.registration_folder("B1", project).parent / "full_brain" / "net"
    run.mkdir(parents=True)
    pd.DataFrame({"z": [5, 5], "y": [5, 5], "x": [3, 9]}).to_csv(run / "cells.csv", index=False)
    monkeypatch.setattr(cf.atlas, "load_atlas", lambda settings: registered)
    table = cf.count_cells_per_region("B1", "net", project)
    assert (run / "regions" / "cells_per_region.csv").exists()
    assert table.set_index("acronym").loc["root", "cells"] == 2
    wide = cf.summary_table("net", project)
    assert "B1 cells_per_mm3" in wide.columns
