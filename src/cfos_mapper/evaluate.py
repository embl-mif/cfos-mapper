"""
How good is a network? Compare its calls with the manual annotations and with
random spots in plain tissue.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def compare_with_annotations(result, annotations, max_distance=3):
    """
    Compare network calls with manual annotations (same candidates, matched
    within max_distance voxels). Returns counts:
      cells_found        annotated cell, called cell
      cells_missed       annotated cell, called not a cell
      false_positives    annotated not-cell, called cell
      correctly_rejected annotated not-cell, called not a cell
    """
    from scipy.spatial import cKDTree

    counts = dict(cells_found=0, cells_missed=0, false_positives=0, correctly_rejected=0)
    if len(annotations) == 0 or len(result) == 0:
        return counts
    tree = cKDTree(result[["z", "y", "x"]].to_numpy(dtype=float))
    distance, index = tree.query(annotations[["z", "y", "x"]].to_numpy(dtype=float))
    matched = distance <= max_distance
    called_cell = result["is_cell"].to_numpy()[index[matched]]
    is_cell = (annotations["label"] == "cell").to_numpy()[matched]
    counts["cells_found"] = int((is_cell & called_cell).sum())
    counts["cells_missed"] = int((is_cell & ~called_cell).sum())
    counts["false_positives"] = int((~is_cell & called_cell).sum())
    counts["correctly_rejected"] = int((~is_cell & ~called_cell).sum())
    return counts


def network_says(points, calls, max_distance=3):
    """
    For each point (z, y, x), what the network said about the nearest
    candidate: "cell", "not_cell", or "no match" (no candidate within
    max_distance voxels, e.g. too close to the border to be classified).
    """
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return np.array([], dtype=object)
    if len(calls) == 0:
        return np.full(len(points), "no match", dtype=object)
    distance, index = cKDTree(calls[["z", "y", "x"]].to_numpy(dtype=float)).query(
        points[["z", "y", "x"]].to_numpy(dtype=float))
    says = np.where(calls["is_cell"].to_numpy()[index], "cell", "not_cell").astype(object)
    says[distance > max_distance] = "no match"
    return says


def used_in_training(points, network_folder, brain_name, region_name, max_distance=3):
    """True for the points that were used to train a network (annotations_split.csv)."""
    from scipy.spatial import cKDTree

    split_file = Path(network_folder) / "annotations_split.csv"
    if len(points) == 0 or not split_file.exists():
        return np.zeros(len(points), dtype=bool)
    split = pd.read_csv(split_file)
    train = split[(split.brain == brain_name) & (split.region == region_name) & (split.set == "train")]
    if len(train) == 0:
        return np.zeros(len(points), dtype=bool)
    distance, _ = cKDTree(train[["z", "y", "x"]].to_numpy(dtype=float)).query(
        points[["z", "y", "x"]].to_numpy(dtype=float))
    return distance <= max_distance


def prediction_statistics(calls, annotations, plain_tissue=None, trained_on=None):
    """
    Numbers to judge a network on one region (a dict, one table row):
      % of candidates called cell
      % false positives on plain tissue (random spots far from any candidate)
      your annotated cells found, your annotated non-cells called cell
    The last two also for the annotations NOT used to train the network
    (trained_on = True/False per annotation), which is the honest check.
    """
    def percent(part, total):
        return round(100 * part / total, 1) if total else np.nan

    row = {"candidates classified": len(calls),
           "% called cell": percent(calls.is_cell.sum(), len(calls))}
    if plain_tissue is not None:
        row["% false positives on plain tissue"] = percent(plain_tissue.is_cell.sum(), len(plain_tissue))
    groups = {"all annotations": np.ones(len(annotations), dtype=bool)}
    if trained_on is not None and np.any(trained_on):
        groups["not used in training"] = ~np.asarray(trained_on)
    for name, keep in groups.items():
        a = compare_with_annotations(calls, annotations[keep])
        n_cells = a["cells_found"] + a["cells_missed"]
        n_not = a["false_positives"] + a["correctly_rejected"]
        row[f"cells found ({name})"] = f"{a['cells_found']}/{n_cells}"
        row[f"% cells found ({name})"] = percent(a["cells_found"], n_cells)
        row[f"non-cells called cell ({name})"] = f"{a['false_positives']}/{n_not}"
        row[f"% false positives ({name})"] = percent(a["false_positives"], n_not)
    return row


def draw_calls(ax, signal, result, title, planes=(135, 165)):
    """
    Max projection of a few cFos planes, with the candidates whose centre is
    in those planes: yellow = called a cell, blue = called not a cell.
    """
    z0, z1 = planes
    ax.imshow(signal[z0:z1].max(0), cmap="gray",
              vmin=np.percentile(signal, 30), vmax=np.percentile(signal, 99.9))
    shown = result[(result.z >= z0) & (result.z < z1)]
    cells, not_cells = shown[shown.is_cell], shown[~shown.is_cell]
    ring = dict(s=110, facecolors="none", linewidths=1)
    ax.scatter(cells.x, cells.y, edgecolors="yellow", **ring)
    ax.scatter(not_cells.x, not_cells.y, edgecolors="deepskyblue", **ring)
    ax.set_title(f"{title}\nyellow = cell ({len(cells)}), blue = not cell ({len(not_cells)})")
    ax.axis("off")
