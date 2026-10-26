"""
Train our own cell / not-cell network from the manual annotations.

  1. collects all annotations: results/<brain>/test_regions/<region>_annotations.csv
  2. keeps 20% of them aside (TEST set, never used for training) to measure
     how good the network is at the end
  3. corrects the brightness of every brain (brightness.json, one factor per
     brain and channel), so all brains look alike to the network
  4. saves the small 3D image cube around every training annotation
  5. trains, starting from cellfinder's pretrained network
  6. tests the new network on the TEST set, on random spots in plain tissue,
     and draws a picture per region

Everything goes to networks/<name>/:
  network.keras      the trained network
  test_results.csv   accuracy per region, test_results.png, pictures per region
  training.csv       loss / accuracy per epoch

Choose the GPU before anything uses it: command line --gpu, or in Python set
os.environ["CUDA_VISIBLE_DEVICES"] before importing cfos_mapper.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .brightness import brightness_factors, correct_brightness
from .classify import classify_candidates, save_cubes
from .detect import far_from_candidates, random_positions
from .evaluate import compare_with_annotations, draw_calls
from .io import get_channel, load_into_memory, voxel_size
from .training_regions import region_folder


def load_region(brain_name, region_name, settings, corrected=True):
    """cFos and autofluorescence of a test region (brightness-corrected by default) and its voxel size."""
    import pymif.microscope_manager as mm

    region = mm.ZarrManager(str(region_folder(brain_name, settings) / f"{region_name}.ome.zarr"))
    signal = load_into_memory(get_channel(region, settings["signal_channel"]))
    background = load_into_memory(get_channel(region, settings["background_channel"]))
    if corrected:
        factors = brightness_factors(brain_name, settings)
        signal = correct_brightness(signal, factors["signal"])
        background = correct_brightness(background, factors["background"])
    return signal, background, voxel_size(region)


def split_train_test(annotations, test_fraction, seed=0):
    """Random split, separately for cells and not-cells (same ratio in both)."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(annotations), dtype=bool)
    for label in ["cell", "not_cell"]:
        index = np.flatnonzero((annotations.label == label).to_numpy())
        n_test = int(round(test_fraction * len(index)))
        is_test[rng.choice(index, n_test, replace=False)] = True
    return annotations[~is_test], annotations[is_test]


def keep_cubes_in_memory(folder):
    """
    Read every training cube file once and keep it in memory. Otherwise
    cellfinder reads 64 small files from the network disk at every training
    step, which is very slow when the machine is busy (the GPU then sits idle,
    waiting for data). cellfinder reads all cubes in one thread of this
    process, so replacing its file reader here is enough.
    """
    import tifffile
    from cellfinder.core.classify import cube_generator

    cubes = {os.path.normpath(f): tifffile.imread(f) for f in Path(folder).rglob("*.tif")}

    def read(path, *args, **kwargs):
        cube = cubes.get(os.path.normpath(str(path)))
        return cube.copy() if cube is not None else tifffile.imread(path, *args, **kwargs)

    cube_generator.imread = read
    print(f"{len(cubes)} cube files kept in memory ({sum(c.nbytes for c in cubes.values()) / 1e9:.2f} GB)")


def collect_annotations(settings, test_fraction):
    """[(brain, region, train table, test table)] for every region with annotations."""
    regions = []
    for file in sorted(Path(settings["results_folder"]).glob("*/test_regions/*_annotations.csv")):
        brain_name = file.parent.parent.name
        region_name = file.name.replace("_annotations.csv", "")
        annotations = pd.read_csv(file)
        if len(annotations) == 0:
            continue
        train, test = split_train_test(annotations, test_fraction)
        regions.append((brain_name, region_name, train, test))
        print(f"{brain_name} {region_name}: train {dict(train.label.value_counts())}, "
              f"test {dict(test.label.value_counts())}")
    return regions


def train_network(name, settings, epochs=60, test_fraction=0.2):
    """Train networks/<name> from all saved annotations and test it (see the top of this file)."""
    import yaml

    out = Path(settings["networks_folder"]) / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(settings, open(out / "settings_used.json", "w"), indent=2)

    # 1-2. annotations, split into training and test
    regions = collect_annotations(settings, test_fraction)
    if not regions:
        raise ValueError("no annotations found (results/<brain>/test_regions/<region>_annotations.csv)")
    pd.concat([t.assign(brain=b, region=r, set="train") for b, r, t, _ in regions] +
              [t.assign(brain=b, region=r, set="test") for b, r, _, t in regions]
              ).to_csv(out / "annotations_split.csv", index=False)

    # 3-4. brightness-corrected cubes around the training annotations
    yaml_files = []
    for brain_name, region_name, train, _ in regions:
        signal, background, size = load_region(brain_name, region_name, settings)
        cube_folder = out / "training_cubes" / f"{brain_name}_{region_name}"
        sections = []
        for label, folder_name, cellfinder_type in [("cell", "cells", "cell"),
                                                     ("not_cell", "non_cells", "no_cell")]:
            points = train[train.label == label]
            n = save_cubes(points, signal, background, size, settings, cube_folder / folder_name)
            print(f"{brain_name} {region_name}: {n} {folder_name} cubes")
            sections.append(dict(cube_dir=str(cube_folder / folder_name), cell_def="",
                                 type=cellfinder_type, signal_channel=0, bg_channel=1))
        yaml_file = cube_folder / "training.yaml"
        yaml.safe_dump({"data": sections}, open(yaml_file, "w"))
        yaml_files.append(str(yaml_file))

    # 5. train (start from the pretrained network); 15% of the training
    #    cubes are used to pick the best epoch
    from cellfinder.core.train.train_yaml import run as run_training

    keep_cubes_in_memory(out / "training_cubes")
    run_training(
        out,
        yaml_files,
        n_free_cpus=settings["find_candidates"]["n_free_cpus"],
        continue_training=True,
        learning_rate=1e-4,
        test_fraction=0.15,
        batch_size=32,
        epochs=epochs,
        save_progress=True,
        max_workers=4,
    )

    # keep only the best epoch (lowest validation loss) as network.keras
    log = pd.read_csv(out / "training.csv")
    best_epoch = int(log.loc[log["val_loss"].idxmin(), "epoch"]) + 1
    checkpoints = sorted(out.glob("model-epoch.*.keras"))
    best = [c for c in checkpoints if c.name.startswith(f"model-epoch.{best_epoch:02d}-")]
    shutil.copy(best[0], out / "network.keras")
    for c in checkpoints:
        c.unlink()
    print(f"best epoch: {best_epoch} -> {out / 'network.keras'}")

    # 6. test: held-out annotations, random spots in plain tissue, pictures
    return test_network(name, settings, regions)


def test_network(name, settings, regions):
    """Test networks/<name> on the held-out annotations of every region; writes test_results.csv/png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(settings["networks_folder"]) / name
    network = out / "network.keras"
    rows = []
    for brain_name, region_name, _, test in regions:
        signal, background, size = load_region(brain_name, region_name, settings)
        candidates = pd.read_csv(region_folder(brain_name, settings) / f"{region_name}_candidates.csv")

        result = classify_candidates(candidates, signal, background, size, settings, model=network)
        result.to_csv(out / f"{brain_name}_{region_name}_calls.csv", index=False)
        a = compare_with_annotations(result, test)

        spots = random_positions(signal.shape, size, settings)
        control = classify_candidates(spots, signal, background, size, settings, model=network)
        plain_tissue = far_from_candidates(control, candidates, size)

        n_cells = a["cells_found"] + a["cells_missed"]
        n_not_cells = a["false_positives"] + a["correctly_rejected"]
        rows.append({
            "brain": brain_name, "region": region_name,
            "test cells": n_cells,
            "% cells found": round(100 * a["cells_found"] / n_cells, 1) if n_cells else np.nan,
            "test non-cells": n_not_cells,
            "% false positives (test)": round(100 * a["false_positives"] / n_not_cells, 1) if n_not_cells else np.nan,
            "% false positives on plain tissue": round(100 * plain_tissue.is_cell.mean(), 1),
            "% candidates called cell": round(100 * result.is_cell.mean(), 1),
        })
        print(rows[-1])

        raw_signal, _, _ = load_region(brain_name, region_name, settings, corrected=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_calls(ax, raw_signal, result, f"{name}: {brain_name} {region_name}")
        fig.tight_layout()
        fig.savefig(out / f"{brain_name}_{region_name}_calls.png", dpi=60)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out / "test_results.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(table)))
    ax.axis("off")
    t = ax.table(cellText=table.fillna("-").values,
                 colLabels=[c.replace(" ", "\n", 1) for c in table.columns],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.4)
    ax.set_title(f"{name}: results on the TEST annotations (not used for training)")
    fig.savefig(out / "test_results.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return table
