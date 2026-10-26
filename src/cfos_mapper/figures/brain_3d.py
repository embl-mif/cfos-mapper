"""
Simple 3D rendering of a brain without a graphics card (works in the background
on ba-training): the brain is resampled to cubes of `um` micrometers, turned to
the viewing angle and projected (brightest value along each line of sight).

Axes of the data: z = dorsal -> ventral, y = anterior -> posterior, x = left-right.
View axes: depth (towards the viewer), row (down the image), column (right).
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

# cell density on a dark background: dark blue -> light blue (one hue, brighter = more cells)
DENSITY_ON_BLACK = LinearSegmentedColormap.from_list(
    "density_on_black", ["#0d366b", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"])


def isotropic(volume, voxel_um, um):
    """Resample a (z, y, x) volume to cubic voxels of `um` micrometers (float32)."""
    return ndimage.zoom(volume.astype(np.float32), np.asarray(voxel_um, float) / um, order=1)


def density_volume(cells_um, shape, um, smooth_um=60):
    """Cells per cubic voxel of `um` (smoothed), from a table with columns z, y, x in micrometers."""
    index = tuple(np.clip((cells_um[a].to_numpy() / um).astype(int), 0, n - 1) for a, n in zip("zyx", shape))
    density = np.zeros(shape, np.float32)
    np.add.at(density, index, 1)
    return ndimage.gaussian_filter(density, smooth_um / um)


def view(azimuth=35, elevation=20):
    """
    Rotation from data axes (z, y, x) to view axes (depth, row, column).
    azimuth=0, elevation=0: seen from the side, dorsal up, anterior on the left.
    azimuth turns the brain around the vertical axis, elevation tilts it towards the viewer.
    """
    side = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)   # depth = x, row = z, column = y
    a, e = np.radians([azimuth, elevation])
    turn = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    tilt = np.array([[np.cos(e), np.sin(e), 0], [-np.sin(e), np.cos(e), 0], [0, 0, 1]])
    return tilt @ turn @ side


class Projector:
    """Projects volumes and points of one isotropic grid (shape) for one view (rotation)."""

    def __init__(self, shape, rotation):
        self.rotation = rotation
        self.centre_in = (np.asarray(shape) - 1) / 2
        size = int(np.ceil(np.linalg.norm(shape)))      # big enough for any angle
        self.shape_out = (size, size, size)
        self.centre_out = (np.asarray(self.shape_out) - 1) / 2

    def volume(self, volume):
        """Brightest value along each line of sight -> 2D image (rows, columns)."""
        inverse = self.rotation.T                        # view -> data coordinates
        offset = self.centre_in - inverse @ self.centre_out
        turned = ndimage.affine_transform(volume, inverse, offset=offset, output_shape=self.shape_out,
                                          order=1, cval=0.0)
        return turned.max(axis=0)

    def points(self, points_vox):
        """(n, 3) points in grid voxels (z, y, x) -> image (rows, columns)."""
        view_coords = (np.asarray(points_vox, float) - self.centre_in) @ self.rotation.T + self.centre_out
        return view_coords[:, 1], view_coords[:, 2]


def density_range(density_image, low_percent=60, high_percent=99.5):
    """
    Colour range for a PROJECTED density image: only the densest parts glow.
    (Each pixel is the highest density along its line of sight, so almost every
    pixel is high; scaling on the projected image keeps the differences visible.)
    """
    inside = density_image[density_image > 0]
    return np.percentile(inside, low_percent), np.percentile(inside, high_percent)


def composite(cfos_image, density_image, cfos_range, density_limits):
    """Gray cFos with the cell density as a blue glow on top (RGB image)."""
    low, high = cfos_range
    gray = np.clip((cfos_image - low) / (high - low), 0, 1) ** 0.7
    rgb = np.repeat(gray[..., None], 3, axis=2)
    low, high = density_limits
    amount = np.clip((density_image - low) / (high - low), 0, 1)
    alpha = (0.8 * amount ** 1.2)[..., None]
    return rgb * (1 - alpha) + DENSITY_ON_BLACK(amount)[..., :3] * alpha
