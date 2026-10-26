"""
Tables of points (pandas, columns z, y, x in full-resolution voxels) and the
manual annotations (columns z, y, x, label = "cell" / "not_cell").
"""

from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.cells.cells import Cell


def cells_to_table(cells):
    """cellfinder Cell objects -> table with columns z, y, x, type."""
    return pd.DataFrame(
        {
            "z": [c.z for c in cells],
            "y": [c.y for c in cells],
            "x": [c.x for c in cells],
            "type": [c.type for c in cells],
        }
    )


def table_to_cells(table):
    """Table with columns z, y, x -> cellfinder Cell objects."""
    return [Cell([x, y, z], Cell.UNKNOWN) for z, y, x in zip(table.z, table.y, table.x)]


def load_annotations(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["z", "y", "x", "label"])


def save_annotations(cells_zyx, not_cells_zyx, path):
    table = pd.concat([
        pd.DataFrame(np.asarray(cells_zyx).reshape(-1, 3), columns=["z", "y", "x"]).assign(label="cell"),
        pd.DataFrame(np.asarray(not_cells_zyx).reshape(-1, 3), columns=["z", "y", "x"]).assign(label="not_cell"),
    ])
    table.to_csv(path, index=False)
    return table


def remove_annotated(candidates, annotations, max_distance=3):
    """
    Candidates that are not yet annotated. A candidate counts as annotated if
    an annotation lies within max_distance voxels (points can get nudged a
    little when clicking them in napari).
    """
    from scipy.spatial import cKDTree

    if len(annotations) == 0 or len(candidates) == 0:
        return candidates.reset_index(drop=True)
    tree = cKDTree(annotations[["z", "y", "x"]].to_numpy(dtype=float))
    distance, _ = tree.query(candidates[["z", "y", "x"]].to_numpy(dtype=float))
    return candidates[distance > max_distance].reset_index(drop=True)
