"""
Find cFos+ cells in a whole brain with a trained network.

  1. loads the cFos channel into memory (~270 GB for a whole brain)
  2. finds candidates in the whole brain (same settings as the test regions,
     plus the local threshold in settings "full_brain")
     -> results/<brain>/full_brain/candidates.csv   (made once, reused)
  3. loads the autofluorescence channel, corrects the brightness of both
     channels (brightness.json) and asks the network about every candidate
     -> results/<brain>/full_brain/<network>/
           cells.csv              the cells (z, y, x, full-resolution voxels)
           all_candidates.csv     every candidate with is_cell True/False
           cells_density.ome.zarr cells per 50 um cube (open in napari)
           overview.png           density map, seen from the top and the side
           run_info.json          numbers, timings, settings used

Memory: about 2 x 270 GB of RAM for a whole brain (ba-training has 1.5 TB).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .brightness import brightness_factors, correct_brightness_in_place
from .classify import classify_candidates
from .detect import find_candidates
from .io import get_channel, load_into_memory, open_brain, save_density_map, voxel_size
from .training_regions import network_file


def say(text):
    print(f"[{datetime.now():%H:%M:%S}] {text}", flush=True)


def run_folder(brain_name, network, settings, planes=None):
    """Output folder of a whole-brain run (planes: only part of the brain, for a quick test)."""
    part = f"_planes{planes[0]}-{planes[1]}" if planes else ""
    return Path(settings["results_folder"]) / brain_name / "full_brain" / f"{network}{part}"


def run_brain(brain_name, network, settings, planes=None):
    """
    Whole-brain cells of one brain with one of our networks (see the top of
    this file). planes = (first, last): only these z-planes, for a quick test.
    Returns the output folder.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = network_file(network, settings)
    if network != "pretrained" and not Path(model).exists():
        raise FileNotFoundError(f"network not found: {model}")

    brain = open_brain(brain_name, settings)
    size = voxel_size(brain)
    n_planes = brain.data[0].shape[-3]
    first, last = planes if planes else (0, n_planes)
    part = f"_planes{first}-{last}" if planes else ""

    out = Path(settings["results_folder"]) / brain_name / "full_brain"
    out_network = run_folder(brain_name, network, settings, planes)
    out_network.mkdir(parents=True, exist_ok=True)
    info = dict(brain=brain_name, network=network, planes=[first, last], voxel_size=size,
                started=f"{datetime.now():%Y-%m-%d %H:%M}", settings=settings)
    timings = info["minutes"] = {}

    def lap(name, t0):
        timings[name] = round((time.time() - t0) / 60, 1)
        json.dump(info, open(out_network / "run_info.json", "w"), indent=2, default=str)
        say(f"{name}: {timings[name]} min")

    # 1-2. candidates (found once per brain, then reused by every network)
    candidates_file = out / f"candidates{part}.csv"
    t0 = time.time()
    say(f"loading cFos of {brain_name}, planes {first}-{last} ...")
    signal = load_into_memory(get_channel(brain, settings["signal_channel"])[first:last])
    lap("load cFos", t0)

    if candidates_file.exists():
        say(f"candidates already found: {candidates_file}")
    else:
        t0 = time.time()
        say("finding candidates ...")
        find_settings = dict(settings)
        find_settings["find_candidates"] = {**settings["find_candidates"], **settings["full_brain"]}
        candidates = find_candidates(signal, size, find_settings)
        candidates["z"] += first  # back to whole-brain plane numbers
        candidates.to_csv(candidates_file, index=False)
        lap("find candidates", t0)
    candidates = pd.read_csv(candidates_file)
    info["candidates"] = len(candidates)
    say(f"{len(candidates)} candidates")

    # 3. brightness correction + classification
    t0 = time.time()
    say("loading autofluorescence and correcting brightness ...")
    background = load_into_memory(get_channel(brain, settings["background_channel"])[first:last])
    if network != "pretrained":  # our networks expect brightness-corrected images
        factors = brightness_factors(brain_name, settings)
        info["brightness_factors"] = factors
        correct_brightness_in_place(signal, factors["signal"])
        correct_brightness_in_place(background, factors["background"])
    lap("load autofluorescence + brightness", t0)

    t0 = time.time()
    say(f"classifying {len(candidates)} candidates with {network} ...")
    in_slab = candidates.assign(z=candidates.z - first)
    calls = classify_candidates(in_slab, signal, background, size, settings, model=model)
    calls["z"] += first
    lap("classify", t0)
    del signal, background

    calls.to_csv(out_network / "all_candidates.csv", index=False)
    cells = calls[calls.is_cell][["z", "y", "x"]]
    cells.to_csv(out_network / "cells.csv", index=False)
    info.update(classified=len(calls), cells=len(cells), not_cells=int((~calls.is_cell).sum()),
                too_close_to_border=len(candidates) - len(calls))
    say(f"{len(cells)} cells, {info['not_cells']} not cells "
        f"({info['too_close_to_border']} candidates too close to the border)")

    # outputs: density map (pymif) and overview picture
    density = save_density_map(cells, brain, out_network / "cells_density.ome.zarr")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw=dict(width_ratios=[2, 1]))
    axes[0].imshow(density.sum(0), cmap="magma")
    axes[0].set_title("cells, seen from the top (sum over z)")
    axes[1].imshow(density.sum(2).T, cmap="magma", aspect="auto")
    axes[1].set_title("seen from the side (sum over x)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{brain_name} - {network}: {len(cells)} cells (50 um bins)")
    fig.savefig(out_network / "overview.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    info["finished"] = f"{datetime.now():%Y-%m-%d %H:%M}"
    json.dump(info, open(out_network / "run_info.json", "w"), indent=2, default=str)
    say(f"done -> {out_network}")
    return out_network
