"""
Brightness correction: all brains should look equally bright to the network
(e.g. one brain imaged with less laser power).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .io import get_channel, open_brain, voxel_size


def brightness_factors(brain_name, settings):
    """
    One brightness correction factor per channel for a brain.
    factor = reference / median brightness of the tissue, measured on a
    low-resolution copy of the brain. Saved in results/<brain>/brightness.json
    and reused.
    """
    from skimage.filters import threshold_otsu

    path = Path(settings["results_folder"]) / brain_name / "brightness.json"
    if path.exists():
        return json.load(open(path))

    brain = open_brain(brain_name, settings)
    level = min(3, len(brain.data) - 1)
    signal = np.asarray(get_channel(brain, settings["signal_channel"], level)).astype(np.float32)
    background = np.asarray(get_channel(brain, settings["background_channel"], level)).astype(np.float32)
    tissue = background > threshold_otsu(background)
    reference = settings["brightness_reference"]
    medians = dict(signal=float(np.median(signal[tissue])),
                   background=float(np.median(background[tissue])))
    factors = dict(signal=reference["signal"] / medians["signal"],
                   background=reference["background"] / medians["background"],
                   tissue_median=medians)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(factors, open(path, "w"), indent=2)
    return factors


def tissue_mask(brain_name, settings, level=3):
    """
    Where the brain is (True) at a low resolution, from the autofluorescence.
    Returns (mask, voxel_size_um of that level). Used for brain outlines and
    volumes (cells per mm^3).
    """
    from skimage.filters import threshold_otsu

    brain = open_brain(brain_name, settings)
    level = min(level, len(brain.data) - 1)
    background = np.asarray(get_channel(brain, settings["background_channel"], level)).astype(np.float32)
    return background > threshold_otsu(background), voxel_size(brain, level)


def correct_brightness(image, factor):
    """Multiply an image by its brightness factor (returns float32)."""
    return image.astype(np.float32) * np.float32(factor)


def correct_brightness_in_place(image, factor, block=4, threads=24):
    """
    Same as correct_brightness, but changes an integer image in place, a block
    of planes at a time (no extra memory: for whole brains). Values above the
    largest integer are capped. Blocks are done in parallel (~10x faster on
    ba-training: ~1 min instead of ~11 min per whole-brain channel).
    """
    def correct_block(z):
        planes = image[z:z + block].astype(np.float32) * np.float32(factor)
        np.clip(planes, 0, np.iinfo(image.dtype).max, out=planes)
        image[z:z + block] = np.rint(planes)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(correct_block, range(0, image.shape[0], block)))
    return image
