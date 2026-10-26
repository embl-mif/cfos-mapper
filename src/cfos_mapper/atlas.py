"""
Register a brain to an atlas (brainreg) and count the cells in every atlas region.

register_brain(brain, settings)
    1. takes the autofluorescence (and cFos) at the pymif resolution level just
       finer than half the atlas voxel (e.g. ~12 um for the 25 um Allen atlas)
    2. runs brainreg (CPU, ~30-60 min) -> results/<brain>/registration/
         downsampled.tiff              the brain at atlas resolution and orientation
         registered_atlas.tiff         the atlas regions laid on that brain
         registered_hemispheres.tiff   1 = left, 2 = right
         registered_atlas_original_orientation.tiff  same, in the orientation of our images
         deformation_field_*.tiff      where every brain voxel is in the atlas
         boundaries.tiff, brainreg.json, volumes.csv (brainreg's own outputs)

count_cells_per_region(brain, network, settings)
    moves every cell of a whole-brain run into the atlas and counts them per
    region and hemisphere -> results/<brain>/full_brain/<network>/regions/
         cells_in_atlas.csv     every cell: position in the brain and in the atlas, region, hemisphere
         cells_per_region.csv   every region (and all its parent regions): cells, volume (mm^3),
                                cells per mm^3, for left, right and both hemispheres

summary_table(network, settings)
    the cells_per_region tables of all brains side by side
    -> reports/<network>/cells_per_region_all_brains.csv

Left / right: the brain is nearly mirror-symmetric, so registration cannot
tell the hemispheres apart. They come from settings["atlas"]["orientation"]
(last letter: "l" = x starts at the animal's left).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .io import get_channel, load_into_memory, open_brain, voxel_size

LEFT, RIGHT = 1, 2  # hemisphere values in brainglobe atlases


# ---------------------------------------------------------------------------
# the atlas
# ---------------------------------------------------------------------------


def atlas_folder(settings):
    return Path(settings["atlas"]["folder"])


def brainglobe_config(settings):
    """
    A brainglobe configuration that keeps the atlases in settings atlas.folder
    (not in the home folder). Returns the folder to put in BRAINGLOBE_CONFIG_DIR.
    """
    import configparser

    config_dir = atlas_folder(settings) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config["default_dirs"] = {"brainglobe_dir": str(atlas_folder(settings)),
                              "interm_download_dir": str(atlas_folder(settings))}
    with open(config_dir / "bg_config.conf", "w") as f:
        config.write(f)
    return config_dir


def load_atlas(settings):
    """The brainglobe atlas of the settings (downloaded into atlas.folder the first time)."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas_folder(settings).mkdir(parents=True, exist_ok=True)
    return BrainGlobeAtlas(settings["atlas"]["name"], brainglobe_dir=atlas_folder(settings))


# ---------------------------------------------------------------------------
# 1. registration
# ---------------------------------------------------------------------------


def registration_folder(brain_name, settings):
    return Path(settings["results_folder"]) / brain_name / "registration"


def registration_ready(brain_name, settings):
    folder = registration_folder(brain_name, settings)
    return all((folder / f).exists() for f in
               ["registered_atlas.tiff", "registered_hemispheres.tiff", "downsampled.tiff",
                "deformation_field_0.tiff", "deformation_field_1.tiff", "deformation_field_2.tiff"])


def registration_level(dataset, atlas_um):
    """The coarsest pymif level whose voxels are all at most half the atlas voxel."""
    levels = [level for level in range(len(dataset.data)) if max(voxel_size(dataset, level)) <= atlas_um / 2]
    return levels[-1] if levels else 0


def register_brain(brain_name, settings, extra_brainreg_args=()):
    """
    Register a brain to the atlas with brainreg (see the top of this file).
    brainreg runs as its own program; its log goes to the registration folder.
    Returns the registration folder.
    """
    import tifffile

    atlas_name = settings["atlas"]["name"]
    atlas_um = float(atlas_name.rsplit("_", 1)[-1].replace("um", ""))
    out = registration_folder(brain_name, settings)
    inputs = out / "input"
    inputs.mkdir(parents=True, exist_ok=True)

    # 1. the brain at a resolution a bit finer than the atlas (small: ~1 GB)
    brain = open_brain(brain_name, settings)
    level = registration_level(brain, atlas_um)
    size = voxel_size(brain, level)
    images = {}
    for label, channel in [("autofluorescence", settings["background_channel"]),
                           ("cFos", settings["signal_channel"])]:
        images[label] = inputs / f"{label}.tiff"
        if not images[label].exists():
            print(f"{brain_name}: saving {label} at level {level} ({size} um) for brainreg", flush=True)
            tifffile.imwrite(images[label], load_into_memory(get_channel(brain, channel, level)))
    json.dump(dict(level=level, voxel_size=size, full_resolution_shape=list(brain.data[0].shape[-3:]),
                   full_resolution_voxel_size=voxel_size(brain), atlas=atlas_name,
                   orientation=settings["atlas"]["orientation"]),
              open(out / "input" / "input_info.json", "w"), indent=2)

    # 2. brainreg (its own program, with the atlas folder of the settings)
    command = [sys.executable, "-c", "import sys; from brainreg.core.cli import main; sys.exit(main())",
               str(images["autofluorescence"]), str(out),
               "-a", str(images["cFos"]),
               "-v", *[str(s) for s in size],
               "--orientation", settings["atlas"]["orientation"],
               "--atlas", atlas_name,
               "--n-free-cpus", str(settings["atlas"]["n_free_cpus"]),
               "--save-original-orientation",
               *extra_brainreg_args]
    environment = {**os.environ, "BRAINGLOBE_CONFIG_DIR": str(brainglobe_config(settings))}
    print(f"{brain_name}: running brainreg (~30-60 min) ...", flush=True)
    subprocess.run(command, check=True, env=environment)
    return out


# ---------------------------------------------------------------------------
# 2. cells per region
# ---------------------------------------------------------------------------


def cells_to_atlas(cells, brain_name, settings, atlas=None):
    """
    Positions of cells (table z, y, x in full-resolution voxels) in the atlas.
    Returns a table with the input columns plus atlas_z, atlas_y, atlas_x
    (atlas voxels), structure_id (0 = outside the brain) and hemisphere
    ("left" / "right" / "outside").
    """
    import tifffile
    from brainglobe_space import AnatomicalSpace

    atlas = atlas or load_atlas(settings)
    folder = registration_folder(brain_name, settings)
    info = json.load(open(folder / "input" / "input_info.json"))

    # brain voxels -> "downsampled" space (atlas orientation and voxel size, same brain)
    with tifffile.TiffFile(folder / "downsampled.tiff") as tif:
        downsampled_shape = tif.series[0].shape
    source = AnatomicalSpace(info["orientation"], shape=info["full_resolution_shape"],
                             resolution=info["full_resolution_voxel_size"])
    target = AnatomicalSpace(atlas.metadata["orientation"], shape=downsampled_shape,
                             resolution=atlas.resolution)
    # brainglobe-space works with positions where voxel i spans [i, i+1): use voxel centres (i + 0.5),
    # otherwise a flipped axis is off by one voxel
    points = source.map_points_to(target, cells[["z", "y", "x"]].to_numpy(dtype=float) + 0.5) - 0.5

    # downsampled space -> atlas, with brainreg's deformation fields (atlas position in mm per voxel)
    index = np.round(points).astype(int)
    inside = np.all((index >= 0) & (index < np.array(downsampled_shape)), axis=1)
    atlas_points = np.full((len(cells), 3), -1, dtype=int)
    for axis in range(3):
        field = tifffile.imread(folder / f"deformation_field_{axis}.tiff")
        mm = field[tuple(index[inside].T)]
        atlas_points[inside, axis] = np.round(mm * 1000 / atlas.resolution[axis]).astype(int)

    in_atlas = np.all((atlas_points >= 0) & (atlas_points < np.array(atlas.annotation.shape)), axis=1)
    structure = np.zeros(len(cells), dtype=int)
    hemisphere = np.zeros(len(cells), dtype=int)
    where = tuple(atlas_points[in_atlas].T)
    structure[in_atlas] = np.asarray(atlas.annotation)[where]
    hemisphere[in_atlas] = np.asarray(atlas.hemispheres)[where]

    table = cells.copy()
    table[["atlas_z", "atlas_y", "atlas_x"]] = atlas_points
    table["structure_id"] = structure
    table["hemisphere"] = np.select([hemisphere == LEFT, hemisphere == RIGHT], ["left", "right"], "outside")
    table.loc[structure == 0, "hemisphere"] = "outside"
    return table


def region_volumes(brain_name, settings):
    """Volume (mm^3) of every atlas label in this brain, per hemisphere: table structure_id, left, right."""
    import tifffile

    folder = registration_folder(brain_name, settings)
    labels = tifffile.imread(folder / "registered_atlas.tiff").ravel()
    hemispheres = tifffile.imread(folder / "registered_hemispheres.tiff").ravel()
    atlas_um = json.load(open(folder / "input" / "input_info.json"))["atlas"].rsplit("_", 1)[-1]
    voxel_mm3 = (float(atlas_um.replace("um", "")) / 1000) ** 3
    volumes = {}
    for side, value in [("left", LEFT), ("right", RIGHT)]:
        ids, counts = np.unique(labels[hemispheres == value], return_counts=True)
        volumes[side] = pd.Series(counts * voxel_mm3, index=ids)
    table = pd.DataFrame(volumes).fillna(0.0)
    table.index.name = "structure_id"
    return table.drop(index=0, errors="ignore").reset_index()


def per_region_table(cells_in_atlas, volumes, atlas):
    """
    Cells, volume and density for every region of the atlas, counting each
    cell (and voxel) also in all the parent regions of its region
    (e.g. a cell in "Primary motor area, layer 5" also counts for
    "Primary motor area", "Isocortex", ...).
    """
    structures = atlas.structures
    rows = {}

    def add(structure_id, column, amount):
        for parent in structures[structure_id]["structure_id_path"]:
            row = rows.setdefault(parent, dict.fromkeys(
                ["left_cells", "right_cells", "left_volume_mm3", "right_volume_mm3"], 0.0))
            row[column] += amount

    counted = cells_in_atlas[cells_in_atlas.structure_id > 0]
    for (structure_id, side), n in counted.groupby(["structure_id", "hemisphere"]).size().items():
        if side in ("left", "right") and structure_id in structures:
            add(int(structure_id), f"{side}_cells", n)
    for _, volume in volumes.iterrows():
        if int(volume.structure_id) in structures:
            add(int(volume.structure_id), "left_volume_mm3", volume.left)
            add(int(volume.structure_id), "right_volume_mm3", volume.right)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "structure_id"
    table = table.reset_index()
    table.insert(1, "acronym", [structures[i]["acronym"] for i in table.structure_id])
    table.insert(2, "name", [structures[i]["name"] for i in table.structure_id])
    table.insert(3, "depth", [len(structures[i]["structure_id_path"]) - 1 for i in table.structure_id])
    table["cells"] = table.left_cells + table.right_cells
    table["volume_mm3"] = table.left_volume_mm3 + table.right_volume_mm3
    for side in ["left_", "right_", ""]:
        volume = table[f"{side}volume_mm3"]
        table[f"{side}cells_per_mm3"] = np.where(volume > 0, table[f"{side}cells"] / volume.where(volume > 0), np.nan)
    for column in ["left_cells", "right_cells", "cells"]:
        table[column] = table[column].astype(int)
    return table.sort_values(["depth", "acronym"]).reset_index(drop=True)


def count_cells_per_region(brain_name, network, settings, run=None):
    """
    Count the cells of a whole-brain run (results/<brain>/full_brain/<run>/cells.csv,
    run = network name by default) per atlas region (see the top of this file).
    Returns the per-region table.
    """
    run_folder = Path(settings["results_folder"]) / brain_name / "full_brain" / (run or network)
    if not registration_ready(brain_name, settings):
        raise FileNotFoundError(f"{brain_name} is not registered yet: run `cfos-mapper register {brain_name}`")
    atlas = load_atlas(settings)
    out = run_folder / "regions"
    out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(run_folder / "cells.csv")
    in_atlas = cells_to_atlas(cells, brain_name, settings, atlas)
    in_atlas["acronym"] = [atlas.structures[i]["acronym"] if i in atlas.structures else "outside"
                           for i in in_atlas.structure_id]
    in_atlas.to_csv(out / "cells_in_atlas.csv", index=False)

    table = per_region_table(in_atlas, region_volumes(brain_name, settings), atlas)
    table.insert(0, "brain", brain_name)
    table.to_csv(out / "cells_per_region.csv", index=False)
    outside = int((in_atlas.structure_id == 0).sum())
    json.dump(dict(cells=len(cells), in_a_region=len(cells) - outside, outside_the_atlas=outside,
                   atlas=settings["atlas"]["name"], orientation=settings["atlas"]["orientation"]),
              open(out / "summary.json", "w"), indent=2)
    print(f"{brain_name} {run or network}: {len(cells) - outside} of {len(cells)} cells in an atlas region "
          f"-> {out / 'cells_per_region.csv'}", flush=True)
    return table


def summary_table(network, settings, brains=None, run=None):
    """
    The per-region tables of all brains with this run, side by side: one row
    per region, columns <brain> cells / <brain> cells_per_mm3.
    -> reports/<network>/cells_per_region_all_brains.csv
    """
    results = Path(settings["results_folder"])
    files = sorted(results.glob(f"*/full_brain/{run or network}/regions/cells_per_region.csv"))
    tables = [pd.read_csv(f) for f in files if brains is None or f.parents[3].name in brains]
    if not tables:
        raise FileNotFoundError(f"no cells_per_region.csv for '{run or network}': run `cfos-mapper count` first")
    keys = ["structure_id", "acronym", "name", "depth"]
    wide = None
    for t in tables:
        brain = t.brain.iloc[0]
        part = t[keys + ["cells", "volume_mm3", "cells_per_mm3"]].rename(
            columns={c: f"{brain} {c}" for c in ["cells", "volume_mm3", "cells_per_mm3"]})
        wide = part if wide is None else wide.merge(part, on=keys, how="outer")
    out = Path(settings["reports_folder"]) / (run or network)
    out.mkdir(parents=True, exist_ok=True)
    wide = wide.sort_values(["depth", "acronym"]).reset_index(drop=True)
    wide.to_csv(out / "cells_per_region_all_brains.csv", index=False)
    return wide
