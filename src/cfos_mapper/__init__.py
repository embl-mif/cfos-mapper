"""
cfos-mapper: find cFos+ cells in whole cleared mouse brains and count them per
atlas region.

    import cfos_mapper as cf
    settings = cf.load_settings("settings.json")
    brain = cf.open_brain("O15", settings)
    ...

All steps are plain functions on one brain (or one test region of a brain);
see the README for the order in which to use them.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # cellfinder's network runs on torch

import pymif.microscope_manager as mm  # noqa: E402

from .atlas import (  # noqa: E402
    cells_to_atlas,
    count_cells_per_region,
    describe_orientation,
    load_atlas,
    per_region_table,
    region_volumes,
    register_brain,
    registered_orientation,
    registration_folder,
    registration_ready,
    summary_table,
)
from .brightness import (  # noqa: E402
    brightness_factors,
    correct_brightness,
    correct_brightness_in_place,
    tissue_mask,
)
from .classify import classify_candidates, pretrained_weights, save_cubes  # noqa: E402
from .detect import far_from_candidates, find_candidates, random_positions, remove_border_candidates  # noqa: E402
from .evaluate import (  # noqa: E402
    compare_with_annotations,
    draw_calls,
    network_says,
    prediction_statistics,
    used_in_training,
)
from .io import (  # noqa: E402
    brain_names,
    get_channel,
    load_into_memory,
    open_brain,
    save_density_map,
    save_image,
    save_region,
    voxel_size,
)
from .settings import load_regions, load_settings, save_regions, write_templates  # noqa: E402
from .tables import (  # noqa: E402
    cells_to_table,
    load_annotations,
    remove_annotated,
    save_annotations,
    table_to_cells,
)
from .train import train_network  # noqa: E402
from .training_regions import (  # noqa: E402
    annotations_file,
    get_predictions,
    network_file,
    prediction_files,
    predictions_ready,
    prepare_region,
    region_box,
    region_folder,
    region_ready,
)
from .whole_brain import run_brain, run_folder  # noqa: E402

try:
    from ._version import __version__
except ImportError:  # not installed (running from a plain source folder)
    __version__ = "0.0.0"

__all__ = [name for name in dir() if not name.startswith("_")]
