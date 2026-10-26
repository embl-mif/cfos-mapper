"""
Widget 4 - atlas regions: register a brain to the atlas (brainreg), check the
registration on the brain, count the cells of a whole-brain run per region and
see the cell density of every region.

  0. "Show the brain"           the brain with its z, y, x axes, and what the orientation in
                                the box means for it (written on the canvas, follows the box)
  1. "Register this brain"      brainreg in a separate process (~30-60 min), with the image
                                orientation of the box (settings atlas.orientation by default)
  2. "Show atlas on the brain"  atlas regions over the brain (hover: region name in the status bar)
  3. "Count cells per region"   for the chosen whole-brain run (separate process)
  4. "Show density per region"  every region coloured by its cells per mm^3

The orientation a brain was registered with is written on the canvas with
its atlas layers.
"""

import json

import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

import cfos_mapper as cf

from . import _common as common

ORIENTATION_HELP = (
    "<b>Image orientation</b>: three letters, one for each image axis z, y, x (axes 0, 1, 2 of the "
    "OME-Zarr), saying where in the animal that axis <i>starts</i> (at index 0):<br><br>"
    "<b>s / i</b> &nbsp; superior (dorsal, top of the head) / inferior (ventral, the belly side)<br>"
    "<b>a / p</b> &nbsp; anterior (the nose, olfactory bulbs) / posterior (cerebellum, spinal cord)<br>"
    "<b>l / r</b> &nbsp; the animal's left / right (not yours, as you look at the image)<br><br>"
    "Each pair is used once. Example: <b>sal</b> = the first plane (z = 0) is dorsal, y runs from the "
    "nose backwards, x from the animal's left to its right.<br><br>"
    "Press <i>Show the brain</i>: in napari's default view the slider moves along z, y goes down "
    "the image and x goes to the right. The axes arrows show this too.<br><br>"
    "Registration cannot check l / r (the brain is nearly symmetric): a wrong letter only swaps the "
    "left and right columns. Check it once with a mark you know, e.g. an injection, a lesion or a cut."
)


class AtlasWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = common.load_settings()
        self.jobs = []
        self.my_layers = []
        self.preview_brain = None  # brain shown with "Show the brain": the canvas follows the box

        self.brain_box = QComboBox()
        self.brain_box.addItems(common.brain_names(self.settings))
        self.brain_box.currentTextChanged.connect(self.update_state)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        show_brain_button = QPushButton("Show the brain (to decide the orientation)")
        show_brain_button.clicked.connect(self.show_brain)
        orientation_label = QLabel("image orientation (where z, y, x start: s/i, a/p, l/r)  ⓘ")
        self.orientation_edit = QLineEdit(self.settings["atlas"]["orientation"])
        self.orientation_edit.textChanged.connect(self.update_orientation_text)
        self.orientation_text = QLabel("")
        self.orientation_text.setWordWrap(True)
        for w in [orientation_label, self.orientation_edit, self.orientation_text]:
            w.setToolTip(ORIENTATION_HELP)
        self.register_button = QPushButton("Register this brain")
        self.register_button.clicked.connect(self.register)
        show_atlas_button = QPushButton("Show atlas on the brain")
        show_atlas_button.clicked.connect(self.show_atlas)
        self.run_box = QComboBox()
        count_button = QPushButton("Count cells per region")
        count_button.clicked.connect(self.count)
        show_density_button = QPushButton("Show density per region")
        show_density_button.clicked.connect(self.show_density)
        self.top = QLabel("")
        self.top.setWordWrap(True)
        self.status = QLabel("Choose a brain.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._group("1. Brain", [
            self.brain_box, self.state, show_brain_button, orientation_label, self.orientation_edit,
            self.orientation_text, self.register_button]))
        layout.addWidget(self._group("2. Check the registration", [show_atlas_button]))
        layout.addWidget(self._group("3. Cells per region", [QLabel("whole-brain run"), self.run_box,
                                                             count_button, show_density_button, self.top]))
        layout.addWidget(self._group("Messages", [self.status]))
        layout.addWidget(QLabel(f"Atlas: {self.settings['atlas']['name']}, default image orientation "
                                f"'{self.settings['atlas']['orientation']}' (settings.json, atlas)."))
        layout.addStretch()
        self.update_state()

    @staticmethod
    def _group(title, widgets):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        for w in widgets:
            layout.addWidget(w)
        return box

    def say(self, text):
        self.status.setText(text)
        print(text)

    @property
    def brain(self):
        return self.brain_box.currentText()

    def runs(self):
        folder = cf.registration_folder(self.brain, self.settings).parent / "full_brain"
        return sorted(p.parent.name for p in folder.glob("*/cells.csv"))

    def counts_file(self, run):
        return cf.registration_folder(self.brain, self.settings).parent / "full_brain" / run / "regions" / \
            "cells_per_region.csv"

    def update_state(self):
        if not self.brain:
            return
        registered = cf.registration_ready(self.brain, self.settings)
        orientation = cf.registered_orientation(self.brain, self.settings) if registered else None
        self.state.setText(f"registered with orientation '{orientation}'" if registered else "not registered yet")
        self.register_button.setText("Register again" if registered else "Register this brain")
        # the box starts at the orientation of this brain's registration, or the settings
        self.orientation_edit.setText(orientation or self.settings["atlas"]["orientation"])
        self.update_orientation_text()
        self.run_box.clear()
        self.run_box.addItems(self.runs())

    def update_orientation_text(self):
        try:
            self.orientation_text.setText(cf.describe_orientation(self.orientation_edit.text()))
        except ValueError as e:
            self.orientation_text.setText(f"NOT VALID: {e}")
        if self.preview_brain is not None and self.preview_brain == self.brain:
            self.preview_orientation()

    def write_on_canvas(self, text):
        self.viewer.text_overlay.text = text
        self.viewer.text_overlay.position = "top_left"
        self.viewer.text_overlay.font_size = 14
        self.viewer.text_overlay.visible = True

    def show_orientation(self, brain):
        """Write on the canvas how the brain was registered."""
        self.preview_brain = None
        orientation = cf.registered_orientation(brain, self.settings)
        self.write_on_canvas(f"{brain}: registered with orientation '{orientation}'\n"
                             f"{cf.describe_orientation(orientation)}")

    def show_brain(self):
        """The brain with its axes, to decide the orientation; the canvas says what the box means."""
        brain = self.brain
        common.add_brain_layers(self.viewer, brain, self.settings)
        if self.viewer.dims.ndim == 3:
            self.viewer.dims.axis_labels = ("z", "y", "x")
        self.viewer.axes.visible = True
        self.viewer.axes.labels = True
        self.viewer.axes.arrows = True
        self.preview_brain = brain
        self.preview_orientation()
        self.say(f"{brain} shown. Move the z slider and look at the top and left of the image, then type the "
                 f"letters that fit in the orientation box (hover over it for help); the text on the canvas "
                 f"follows the box.")

    def preview_orientation(self):
        """Write on the canvas what the orientation in the box means for the shown brain."""
        orientation = self.orientation_edit.text().strip().lower()
        try:
            z, y, x = cf.orientation_words(orientation)
        except ValueError:
            self.write_on_canvas(f"{self.preview_brain}: orientation '{orientation}' is not valid "
                                 f"(one of s/i, a/p, l/r each)")
            return
        self.write_on_canvas(f"{self.preview_brain} with orientation '{orientation}' (the box), axes start at:\n"
                             f"z = 0 (first plane, slider at the start): {z}\n"
                             f"y = 0 (top of the image): {y}\n"
                             f"x = 0 (left of the image): {x}")

    # ------------------------------------------------------------------
    def register(self):
        brain = self.brain
        orientation = self.orientation_edit.text().strip().lower()
        try:
            words = cf.describe_orientation(orientation)
        except ValueError as e:
            self.say(str(e))
            return
        self.register_button.setEnabled(False)
        self.say(f"registering {brain} with orientation '{orientation}' ({words}) with brainreg in a separate "
                 f"process (~30-60 min). You can keep working; the log is in the logs folder.")

        def done(ok, log):
            self.register_button.setEnabled(True)
            self.update_state()
            self.say(f"{brain} registered: press 'Show atlas on the brain' to check it." if ok
                     else f"Registering {brain} FAILED, see {log}")

        self.jobs.append(common.BackgroundJob(["register", brain, "--orientation", orientation], done, tries=1))

    def show_atlas(self):
        brain = self.brain
        if not cf.registration_ready(brain, self.settings):
            self.say(f"{brain} is not registered yet.")
            return
        dataset, size = common.add_brain_layers(self.viewer, brain, self.settings)
        self.say(f"loading the atlas regions of {brain} ...")

        @thread_worker
        def work():
            import tifffile

            folder = cf.registration_folder(brain, self.settings)
            labels = tifffile.imread(folder / "registered_atlas_original_orientation.tiff")
            atlas = cf.load_atlas(self.settings)
            ids = np.unique(labels)
            names = ["outside the brain" if i == 0 else
                     f"{atlas.structures[i]['acronym']}: {atlas.structures[i]['name']}" if i in atlas.structures
                     else str(i) for i in ids]
            return labels, pd.DataFrame({"index": ids, "region": names})

        def show(result):
            labels, features = result
            name = f"{brain}: atlas regions"
            common.remove_layers(self.viewer, [name])
            scale, translate = self.atlas_scale(dataset, size, labels.shape)
            layer = self.viewer.add_labels(labels, name=name, scale=scale, translate=translate, opacity=0.35,
                                           features=features)
            layer.contour = 1
            self.my_layers.append(name)
            self.show_orientation(brain)
            self.say(f"{brain}: atlas regions shown (outlines). Hover over the brain to see the region name "
                     f"in the status bar. Do the outlines follow the brain?")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR: {e}"))
        worker.start()

    @staticmethod
    def atlas_scale(dataset, size, atlas_shape):
        """Scale / translate that lay an atlas-resolution image (our orientation) over the whole brain."""
        extent = np.array(dataset.data[0].shape[-3:]) * np.array(size)
        scale = extent / np.array(atlas_shape)
        return scale, scale / 2 - np.array(size) / 2

    # ------------------------------------------------------------------
    def count(self):
        brain, run = self.brain, self.run_box.currentText()
        if not run:
            self.say(f"{brain} has no whole-brain run yet (cfos-mapper run-brain).")
            return
        if not cf.registration_ready(brain, self.settings):
            self.say(f"Register {brain} first.")
            return
        info = json.load(open(cf.registration_folder(brain, self.settings).parent / "full_brain" / run
                              / "run_info.json"))
        self.say(f"counting the cells of {brain} {run} per region in a separate process ...")

        def done(ok, log):
            if ok:
                self.say(f"{brain} {run}: cells per region saved. Press 'Show density per region'.")
                self.show_top(run)
            else:
                self.say(f"Counting FAILED, see {log}")

        self.jobs.append(common.BackgroundJob(["count", brain, "--network", info.get("network", run), "--run", run],
                                              done, tries=1))

    def show_top(self, run):
        table = pd.read_csv(self.counts_file(run))
        top = table[(table.depth >= 5) & (table.volume_mm3 > 0.05)].nlargest(12, "cells_per_mm3")
        self.top.setText("highest density (cells/mm³):\n" + "\n".join(
            f"{r.acronym:>8}  {r.cells_per_mm3:8.0f}   ({r.cells} cells)" for r in top.itertuples()))

    def show_density(self):
        brain, run = self.brain, self.run_box.currentText()
        if not run or not self.counts_file(run).exists():
            self.say("Count the cells per region first.")
            return
        dataset, size = common.add_brain_layers(self.viewer, brain, self.settings)

        @thread_worker
        def work():
            import tifffile

            labels = tifffile.imread(cf.registration_folder(brain, self.settings)
                                     / "registered_atlas_original_orientation.tiff")
            table = pd.read_csv(self.counts_file(run)).set_index("structure_id")
            ids, inverse = np.unique(labels, return_inverse=True)
            density = table["cells_per_mm3"].reindex(ids).fillna(0).to_numpy(dtype=np.float32)
            return density[inverse].reshape(labels.shape)

        def show(image):
            name = f"{brain} {run}: cells per mm³ (per region)"
            common.remove_layers(self.viewer, [name])
            scale, translate = self.atlas_scale(dataset, size, image.shape)
            self.viewer.add_image(image, name=name, scale=scale, translate=translate, colormap="bop blue",
                                  blending="additive", opacity=0.7,
                                  contrast_limits=[0, float(np.percentile(image[image > 0], 99))])
            self.show_top(run)
            self.show_orientation(brain)
            self.say(f"{brain} {run}: each region coloured by its cells per mm³.")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR: {e}"))
        worker.start()
