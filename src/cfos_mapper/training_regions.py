"""
Training (test) regions: small boxes cut out of a brain (regions.json), where candidates
are annotated by hand and networks are compared.

Files, per brain, in results/<brain>/test_regions/:
  <region>.ome.zarr               the region (all channels), cut out with pymif
  <region>_candidates.csv         its candidates
  <region>_annotations.csv        your annotations (napari widget 2)
  <region>_pretrained.csv         calls of cellfinder's pretrained network
  <region>_random_control.csv     ... on random spots (control)
Calls of our own networks go to networks/<network>/<brain>_<region>_calls.csv.
"""

import json
from pathlib import Path

import pandas as pd
import pymif.microscope_manager as mm

from .brightness import brightness_factors, correct_brightness
from .classify import classify_candidates
from .detect import find_candidates, random_positions, remove_border_candidates
from .io import get_channel, load_into_memory, open_brain, save_region, voxel_size


def region_folder(brain_name, settings):
    return Path(settings["results_folder"]) / brain_name / "test_regions"


def region_box(brain_name, region_name, settings, regions_file=None):
    """(start, end) per axis of a test region, from regions.json."""
    with open(regions_file or settings["regions_file"]) as f:
        return json.load(f)[brain_name][region_name]


def prepare_region(brain_name, region_name, settings, regions_file=None):
    """
    Make sure a test region exists (cut out of the brain with pymif) and has
    its candidates. Returns (signal, background, voxel_size, all_candidates).
    """
    folder = region_folder(brain_name, settings)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{region_name}.ome.zarr"
    if not path.exists():
        box = region_box(brain_name, region_name, settings, regions_file)
        save_region(open_brain(brain_name, settings), box["z"], box["y"], box["x"], path)
    region = mm.ZarrManager(str(path))
    signal = load_into_memory(get_channel(region, settings["signal_channel"]))
    background = load_into_memory(get_channel(region, settings["background_channel"]))
    size = voxel_size(region)
    candidates_file = folder / f"{region_name}_candidates.csv"
    if not candidates_file.exists():
        find_candidates(signal, size, settings, small_region=True).to_csv(candidates_file, index=False)
    return signal, background, size, pd.read_csv(candidates_file)


def region_ready(brain_name, region_name, settings):
    """True if a test region has been cut out of the brain and has its candidates."""
    folder = region_folder(brain_name, settings)
    return (folder / f"{region_name}.ome.zarr").exists() and (folder / f"{region_name}_candidates.csv").exists()


def annotations_file(brain_name, region_name, settings):
    return region_folder(brain_name, settings) / f"{region_name}_annotations.csv"


def network_file(network, settings):
    """The .keras file of one of our networks (or "pretrained")."""
    if network == "pretrained":
        return "pretrained"
    return Path(settings["networks_folder"]) / network / "network.keras"


def prediction_files(brain_name, region_name, network, settings):
    """Files holding a network's predictions on a test region: (calls, random-spot control)."""
    if network == "pretrained":
        folder = region_folder(brain_name, settings)
        return folder / f"{region_name}_pretrained.csv", folder / f"{region_name}_random_control.csv"
    folder = Path(settings["networks_folder"]) / network
    return folder / f"{brain_name}_{region_name}_calls.csv", folder / f"{brain_name}_{region_name}_random_control.csv"


def predictions_ready(brain_name, region_name, network, settings):
    return all(f.exists() for f in prediction_files(brain_name, region_name, network, settings))


def get_predictions(brain_name, region_name, network, settings, regions_file=None, max_workers=0):
    """
    Predictions of a network ("pretrained" or the name of one of ours) on a
    test region, plus the random-spot control. Computed once, then read from
    file. Returns dict(calls, control, candidates, signal, size).
    """
    signal, background, size, candidates = prepare_region(brain_name, region_name, settings, regions_file)
    calls_file, control_file = prediction_files(brain_name, region_name, network, settings)

    if not (calls_file.exists() and control_file.exists()):
        model = network_file(network, settings)
        s, b = signal, background
        if network != "pretrained":  # our networks expect brightness-corrected images
            factors = brightness_factors(brain_name, settings)
            s = correct_brightness(signal, factors["signal"])
            b = correct_brightness(background, factors["background"])
        calls_file.parent.mkdir(parents=True, exist_ok=True)
        if not calls_file.exists():
            inner = remove_border_candidates(candidates, signal.shape, size, settings)
            classify_candidates(inner, s, b, size, settings, model=model,
                                max_workers=max_workers).to_csv(calls_file, index=False)
        if not control_file.exists():
            spots = random_positions(signal.shape, size, settings)
            classify_candidates(spots, s, b, size, settings, model=model,
                                max_workers=max_workers).to_csv(control_file, index=False)
    return dict(calls=pd.read_csv(calls_file), control=pd.read_csv(control_file),
                candidates=candidates, signal=signal, size=size)
