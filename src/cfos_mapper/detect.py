"""
Step 1 of cellfinder: find candidates (bright, cell-sized spots), plus the
positions used as controls.
"""

import numpy as np
import pandas as pd
from brainglobe_utils.cells.cells import Cell

from .tables import cells_to_table


def find_candidates(signal, voxel_size_um, settings, small_region=False):
    """
    Find bright, cell-sized spots in the cFos image.
    Returns a table with one row per candidate (z, y, x).

    small_region=True for regions cut out of a brain: cellfinder decides what
    is "outside the brain" by looking at the top-left corner of each plane
    and ignores everything not brighter than it. In a whole brain the corner
    is empty space, but in a small region it can be bright tissue, and then
    most of the region is ignored. So we make that corner (a few voxels)
    black.
    """
    from cellfinder.core.detect import detect

    if small_region:
        signal = signal.copy()
        signal[:, :32, :32] = 0

    found = detect.main(
        signal_array=signal,
        voxel_sizes=voxel_size_um,
        **settings["find_candidates"],
    )
    table = cells_to_table(found)
    # cellfinder marks very large blobs as artifacts (type 3): drop them
    return table[table["type"] == Cell.UNKNOWN].reset_index(drop=True)


def remove_border_candidates(candidates, image_shape, voxel_size_um, settings):
    """
    Remove candidates too close to the image border: the classifier needs a
    full cube of image around each candidate.
    """
    s = settings["classify"]
    half_cube = [
        int(np.ceil(size * net / vox / 2))  # half cube, in image voxels
        for size, net, vox in zip(s["cube_size"], s["network_voxel_size"], voxel_size_um)
    ]
    keep = np.ones(len(candidates), dtype=bool)
    for axis, half, n in zip("zyx", half_cube, image_shape):
        keep &= (candidates[axis] >= half).to_numpy() & (candidates[axis] < n - half).to_numpy()
    return candidates[keep].reset_index(drop=True)


def random_positions(image_shape, voxel_size_um, settings, n=300, seed=0):
    """
    n random positions (away from the border), as a control: a good
    classifier should call most random spots in the tissue "not a cell".
    """
    rng = np.random.default_rng(seed)
    margin = 40
    table = pd.DataFrame({
        axis: rng.integers(margin, size - margin, n * 2)
        for axis, size in zip("zyx", image_shape)
    })
    return remove_border_candidates(table, image_shape, voxel_size_um, settings).iloc[:n]


def far_from_candidates(spots, candidates, voxel_size_um, min_distance_um=10):
    """
    Spots farther than min_distance_um from every candidate: plain tissue,
    almost certainly not a cell.
    """
    from scipy.spatial import cKDTree

    um = np.asarray(voxel_size_um)
    tree = cKDTree(candidates[["z", "y", "x"]].to_numpy(dtype=float) * um)
    distance, _ = tree.query(spots[["z", "y", "x"]].to_numpy(dtype=float) * um)
    return spots[distance > min_distance_um].reset_index(drop=True)
