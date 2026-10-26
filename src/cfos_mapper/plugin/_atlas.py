"""
Widget 4 - atlas regions: register a brain to the atlas (brainreg), check the
registration on the brain, count the cells of a whole-brain run per region and
see the cell density of every region.

  1. "Register this brain"      brainreg in a separate process (~30-60 min)
  2. "Show atlas on the brain"  atlas regions over the brain (hover: region name in the status bar)
  3. "Count cells per region"   for the chosen whole-brain run (separate process)
  4. "Show density per region"  every region coloured by its cells per mm^3
"""

import json

import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

import cfos_mapper as cf

from . import _common as common


class AtlasWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = common.load_settings()
        self.jobs = []
        self.my_layers = []

        self.brain_box = QComboBox()
        self.brain_box.addItems(common.brain_names(self.settings))
        self.brain_box.currentTextChanged.connect(self.update_state)
        self.state = QLabel("")
        self.state.setWordWrap(True)
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
        layout.addWidget(self._group("1. Brain", [self.brain_box, self.state, self.register_button]))
        layout.addWidget(self._group("2. Check the registration", [show_atlas_button]))
        layout.addWidget(self._group("3. Cells per region", [QLabel("whole-brain run"), self.run_box,
                                                             count_button, show_density_button, self.top]))
        layout.addWidget(self._group("Messages", [self.status]))
        layout.addWidget(QLabel(f"Atlas: {self.settings['atlas']['name']}, image orientation "
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
        self.state.setText("registered" if registered else "not registered yet")
        self.register_button.setText("Register again" if registered else "Register this brain")
        self.run_box.clear()
        self.run_box.addItems(self.runs())

    # ------------------------------------------------------------------
    def register(self):
        brain = self.brain
        self.register_button.setEnabled(False)
        self.say(f"registering {brain} with brainreg in a separate process (~30-60 min). "
                 f"You can keep working; the log is in the logs folder.")

        def done(ok, log):
            self.register_button.setEnabled(True)
            self.update_state()
            self.say(f"{brain} registered: press 'Show atlas on the brain' to check it." if ok
                     else f"Registering {brain} FAILED, see {log}")

        self.jobs.append(common.BackgroundJob(["register", brain], done, tries=1))

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
            self.say(f"{brain} {run}: each region coloured by its cells per mm³.")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR: {e}"))
        worker.start()
