"""
Step 2 of cellfinder: decide for every candidate if it is a cell, with a
network ("pretrained" = cellfinder's own, or one trained on our annotations).
"""

import gc
from pathlib import Path

from brainglobe_utils.cells.cells import Cell

from .tables import cells_to_table, table_to_cells


def pretrained_weights():
    """Path of cellfinder's pretrained network (downloaded the first time)."""
    from cellfinder.core.download.download import DEFAULT_DOWNLOAD_DIRECTORY
    from cellfinder.core.tools.prep import prep_model_weights

    return prep_model_weights(None, DEFAULT_DOWNLOAD_DIRECTORY, "resnet50_tv")


def classify_candidates(candidates, signal, background, voxel_size_um, settings, model=None,
                        max_workers=3):
    """
    Decide for every candidate if it is a cell, using a small 3D image cube
    around it (cFos + autofluorescence).
    Returns the candidates table with an extra column 'is_cell' (True/False).
    Candidates too close to the image border cannot be classified and are dropped.

    model: "pretrained" or the path of a trained network (.keras). Default:
    settings["classify"]["model"]. Our own networks expect brightness-corrected
    images (see correct_brightness).
    max_workers: extra processes preparing the cubes (0 = none, needed inside napari).
    """
    from cellfinder.core.classify import classify

    s = settings["classify"]
    model = model or s["model"]
    if model == "pretrained":
        model_file, weights_file = None, pretrained_weights()
    else:
        model_file, weights_file = str(model), None
    depth, height, width = s["cube_size"]
    points = table_to_cells(candidates)

    # keras (torch backend) leaves the GPU memory of every batch in reference
    # cycles that only Python's garbage collector frees, and with millions of
    # candidates in memory it runs too rarely: the GPU fills up after ~10000
    # batches. So we collect every 20 batches, and freeze the objects that
    # exist now (the candidates...) so the collector skips them and is fast.
    def free_gpu_memory(batch):
        if batch % 20 == 0:
            gc.collect()

    gc.collect()
    gc.freeze()
    try:
        classified = classify.main(
            points=points,
            signal_array=signal,
            background_array=background,
            n_free_cpus=settings["find_candidates"]["n_free_cpus"],
            voxel_sizes=voxel_size_um,
            network_voxel_sizes=s["network_voxel_size"],
            batch_size=s["batch_size"],
            cube_height=height,
            cube_width=width,
            cube_depth=depth,
            trained_model=model_file,
            model_weights=weights_file,
            network_depth="50",
            max_workers=max_workers,
            callback=free_gpu_memory,
        )
    finally:
        gc.unfreeze()
    table = cells_to_table(classified)
    table["is_cell"] = table["type"] == Cell.CELL
    return table.drop(columns="type")


def save_cubes(points, signal, background, voxel_size_um, settings, out_folder):
    """
    Save the small 3D image cube around each point (as the network sees it),
    one tiff per channel: pCellz<Z>y<Y>x<X>Ch0.tif (cFos), ...Ch1.tif
    (autofluorescence). Uses cellfinder's own cube extraction, the same as
    during classification. Returns the number of cubes saved.
    """
    import tifffile
    from cellfinder.core.classify.cube_generator import CuboidArrayDataset, CuboidBatchSampler

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    cells = table_to_cells(points)
    if not cells:
        return 0
    s = settings["classify"]
    dataset = CuboidArrayDataset(
        signal_array=signal,
        background_array=background,
        points=cells,
        data_voxel_sizes=voxel_size_um,
        network_voxel_sizes=s["network_voxel_size"],
        network_cuboid_voxels=tuple(s["cube_size"]),
        axis_order=("z", "y", "x"),
        output_axis_order=("z", "y", "x", "c"),
        target_output="index",
    )
    n = 0
    for batch in CuboidBatchSampler(dataset=dataset, batch_size=1, sort_by_axis="z"):
        cubes, indices = dataset[batch]
        for cube, i in zip(cubes, indices):
            c = cells[int(i.item())]
            name = f"pCellz{int(round(c.z))}y{int(round(c.y))}x{int(round(c.x))}"
            for channel in range(cube.shape[-1]):
                tifffile.imwrite(out_folder / f"{name}Ch{channel}.tif", cube[..., channel].numpy())
            n += 1
    return n
