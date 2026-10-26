"""
Opening and saving images with pymif (OME-Zarr).

Coordinates are always (z, y, x) in full-resolution voxels.
"""

from pathlib import Path

import numpy as np
import pymif.microscope_manager as mm
from tqdm import tqdm


def open_brain(brain_name, settings):
    """Open a brain, e.g. 'O15', with pymif (nothing is loaded yet)."""
    path = Path(settings["brains_folder"]) / f"{brain_name}.ome.zarr"
    return mm.ZarrManager(str(path))


def brain_names(settings):
    """Brains in the brains folder (<name>.ome.zarr)."""
    return sorted(p.name.replace(".ome.zarr", "") for p in Path(settings["brains_folder"]).glob("*.ome.zarr"))


def get_channel(dataset, channel_name, level=0):
    """
    One channel of a pymif dataset as a 3D (z, y, x) lazy array.
    channel_name only needs to be part of the name, e.g. 'cFos' or 'auto'.
    """
    names = dataset.metadata["channel_names"]
    matches = [i for i, n in enumerate(names) if channel_name.lower() in n.lower()]
    if len(matches) != 1:
        raise ValueError(f"channel '{channel_name}' not found (or not unique) in {names}")
    data = dataset.data[level]
    axes = dataset.metadata["axes"].lower()
    index = [0] * (len(axes) - 3)  # take t=0 if there is a time axis
    if "c" in axes:
        index[axes.index("c")] = matches[0]
    return data[tuple(index)]


def voxel_size(dataset, level=0):
    """(z, y, x) voxel size in micrometers."""
    return tuple(float(v) for v in dataset.metadata["scales"][level])


def save_region(dataset, z, y, x, out_path):
    """
    Save a box of the full-resolution image (all channels) with pymif.
    z, y, x are (start, end) pairs in full-resolution voxels.
    """
    axes = dataset.metadata["axes"].lower()
    box = {"z": slice(*z), "y": slice(*y), "x": slice(*x)}
    index = tuple(box.get(a, slice(None)) for a in axes)
    region = dataset.data[0][index]

    metadata = dict(dataset.metadata)
    metadata["size"] = [region.shape]
    metadata["scales"] = [metadata["scales"][0]]
    metadata["chunksize"] = [region.chunksize]
    out = mm.ArrayManager(region, metadata, chunks=None)
    out.to_zarr(str(out_path))
    return mm.ZarrManager(str(out_path))


def load_into_memory(lazy_array, block=122):
    """
    Read a (z, y, x) array from disk into memory (a numpy array), a block of
    planes at a time so memory never holds two copies (important for whole
    brains: ~270 GB per channel). block = 2 zarr chunks (61 planes each) reads
    ~35% faster than 1; bigger blocks don't help (the network disk is the limit).
    """
    out = np.empty(lazy_array.shape, dtype=lazy_array.dtype)
    n_planes = lazy_array.shape[0]
    with tqdm(total=n_planes, desc="Loading", unit="planes") as progress:
        for z in range(0, n_planes, block):
            out[z:z + block] = np.asarray(lazy_array[z:z + block].compute(scheduler="threads"))
            progress.update(min(block, n_planes - z))
    return out


def save_image(image, voxel_size_um, out_path, channel_names, like=None):
    """
    Save a (c, z, y, x) or (z, y, x) numpy image as OME-Zarr with pymif.
    like: a pymif dataset whose metadata (units, ...) is copied.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None]
    metadata = dict(like.metadata) if like is not None else {}
    metadata.update(
        axes="czyx", size=[image.shape], chunksize=[image.shape],
        scales=[tuple(float(s) for s in voxel_size_um)],
        channel_names=list(channel_names), channel_colors=["FFFFFF"] * image.shape[0],
        dtype=str(image.dtype),
    )
    for key, value in [("units", ("micrometer",) * 3), ("time_increment", None), ("time_increment_unit", None),
                       ("plane_files", None), ("data_type", "intensity"), ("is_label", False)]:
        metadata.setdefault(key, value)
    metadata.pop("shards", None)
    mm.ArrayManager(image, metadata, chunks=None).to_zarr(str(out_path))
    return mm.ZarrManager(str(out_path))


def save_density_map(cells, brain, out_path, bin_um=50):
    """
    Number of cells per cube of bin_um x bin_um x bin_um micrometers, saved as
    OME-Zarr with pymif (open it in napari like any image). Returns the array.
    """
    size = voxel_size(brain)
    shape = np.array(brain.data[0].shape[-3:])  # full-resolution (z, y, x)
    bin_voxels = np.maximum(1, np.round(bin_um / np.array(size))).astype(int)
    n_bins = np.ceil(shape / bin_voxels).astype(int)
    index = tuple(np.clip((cells[a].to_numpy() // b).astype(int), 0, n - 1)
                  for a, b, n in zip("zyx", bin_voxels, n_bins))
    density = np.zeros(n_bins, dtype=np.uint16)
    np.add.at(density, index, 1)

    metadata = dict(brain.metadata)
    metadata.update(
        axes="czyx", size=[(1, *density.shape)], chunksize=[(1, *density.shape)],
        scales=[tuple(float(b * s) for b, s in zip(bin_voxels, size))],
        channel_names=[f"cells per {bin_um} um"], channel_colors=["FFFF00"], dtype="uint16",
    )
    mm.ArrayManager(density[None], metadata, chunks=None).to_zarr(str(out_path))
    return density
