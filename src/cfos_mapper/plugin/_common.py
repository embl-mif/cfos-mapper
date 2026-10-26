"""Pieces shared by the widgets of the plugin."""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import cfos_mapper as cf
from cfos_mapper import settings as settings_module

# colours of the rings (same meaning in every widget)
ANNOTATION_CELL, ANNOTATION_NOT_CELL = "#00d000", "#ff2a2a"    # solid green / red
NETWORK_CELL, NETWORK_NOT_CELL = "#a8f0a8", "#ffb3b3"          # light green / red
LINE = 1.6                                                      # ring line thickness


def load_settings():
    """The project settings ($CFOS_MAPPER_SETTINGS, set by `cfos-mapper napari`, or ./settings.json)."""
    return cf.load_settings()


def load_regions():
    return settings_module.load_regions(load_settings())


def save_regions(regions):
    settings_module.save_regions(regions, load_settings())


def brain_names(settings):
    return cf.brain_names(settings)


def ring_style(size):
    """Thin ring (empty circle) of `size` data pixels, drawn like a sphere across planes."""
    return dict(size=size, symbol="disc", face_color="transparent", border_width=LINE / size,
                border_width_is_relative=True, projection_mode="rescale_spherical")


def brain_layer_name(brain_name, channel):
    return f"{brain_name}: {channel}"


def add_brain_layers(viewer, brain_name, settings):
    """
    Show a whole brain (all resolution levels, loaded only when needed) as two
    image layers: cFos (gray) and autofluorescence (green, hidden). Layers that
    are already in the viewer are reused. Returns (pymif dataset, voxel size).
    """
    dataset = cf.open_brain(brain_name, settings)
    size = cf.voxel_size(dataset)
    for channel, label, colormap, visible in [
        (settings["background_channel"], "autofluorescence", "green", False),
        (settings["signal_channel"], "cFos", "gray", True),
    ]:
        name = brain_layer_name(brain_name, label)
        if name in viewer.layers:
            continue
        levels = [cf.get_channel(dataset, channel, level) for level in range(len(dataset.data))]
        lowest = np.asarray(levels[-1])
        viewer.add_image(levels, multiscale=True, name=name, scale=size, colormap=colormap,
                         blending="additive", visible=visible,
                         contrast_limits=[float(np.percentile(lowest, 1)), float(np.percentile(lowest, 99.9))])
    return dataset, size


def focus(viewer, low_world, high_world):
    """Move the view to a box (world coordinates z, y, x): middle plane, centred, zoomed to fit."""
    low_world, high_world = np.asarray(low_world, float), np.asarray(high_world, float)
    centre = (low_world + high_world) / 2
    viewer.dims.set_point(0, centre[0])
    viewer.camera.center = (centre[1], centre[2])
    canvas = getattr(viewer, "_canvas_size", None)
    extent = np.maximum(high_world[1:] - low_world[1:], 1e-6)
    if canvas is not None and min(canvas) > 10:
        viewer.camera.zoom = 0.9 * min(canvas[0] / extent[0], canvas[1] / extent[1])


def remove_layers(viewer, names):
    for name in names:
        if name in viewer.layers:
            viewer.layers.remove(name)


def command(*args):
    """The command line that runs `cfos-mapper <args>` with this project's settings, in this Python."""
    return [sys.executable, "-m", "cfos_mapper", "--settings", load_settings()["settings_file"], *map(str, args)]


def log_file(name):
    folder = Path(load_settings()["logs_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"


class BackgroundJob:
    """
    Runs `cfos-mapper <args>` in its own process (all GPU work goes there, so a
    GPU problem can never crash napari) and calls on_done(ok, log_file) when it
    has finished. A failed job is tried again (up to `tries` times in all).
    The job keeps running if napari is closed. Keep a reference to the job while it runs.
    """

    running = {}  # args -> job: the same job is never started twice at the same time

    def __init__(self, args, on_done, tries=3):
        from qtpy.QtCore import QTimer

        self.args, self.on_done, self.tries_left = tuple(str(a) for a in args), on_done, tries
        self.log = log_file("job_" + "_".join(self.args).replace("-", "").replace(" ", ""))
        if self.args in BackgroundJob.running:  # already running: just wait for it too
            other = BackgroundJob.running[self.args]
            other.also.append(on_done)
            return
        self.also = []
        self._start()
        BackgroundJob.running[self.args] = self
        self.timer = QTimer()
        self.timer.timeout.connect(self._check)
        self.timer.start(1000)

    def _start(self):
        import subprocess

        self.tries_left -= 1
        self.process = subprocess.Popen(command(*self.args), stdout=open(self.log, "a"), stderr=subprocess.STDOUT,
                                        start_new_session=True, env={**os.environ})

    def _check(self):
        if self.process.poll() is None:
            return
        if self.process.returncode != 0 and self.tries_left > 0:
            with open(self.log, "a") as f:
                f.write(f"\n--- failed (exit code {self.process.returncode}), trying again ---\n")
            self._start()
            return
        self.timer.stop()
        BackgroundJob.running.pop(self.args, None)
        ok = self.process.returncode == 0
        for callback in [self.on_done] + self.also:
            callback(ok, self.log)
