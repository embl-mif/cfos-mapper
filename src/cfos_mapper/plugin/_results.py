"""
Widget 3 - inspect a whole-brain result of `cfos-mapper run-brain` in one window:
the brain (cFos, autofluorescence), what the network said about every candidate
(light green ring = cell, light red ring = not a cell) and the cell-density map.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymif.microscope_manager as mm
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QCheckBox, QComboBox, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from . import _common as common


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = common.load_settings()
        self.results = Path(self.settings["results_folder"])
        self.my_layers = []

        self.brain_box = QComboBox()
        self.brain_box.currentTextChanged.connect(self.list_runs)
        self.run_box = QComboBox()
        refresh_button = QPushButton("Refresh lists")
        refresh_button.clicked.connect(self.list_brains)
        self.show_cells = QCheckBox("cells (light green rings)")
        self.show_cells.setChecked(True)
        self.show_not_cells = QCheckBox("not cells (light red rings)")
        self.show_density = QCheckBox("cell density (blue)")
        self.show_density.setChecked(True)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.status = QLabel("Choose a brain and a run (= network used in `cfos-mapper run-brain`), press 'Load'.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._group("Whole-brain result", [QLabel("brain"), self.brain_box, QLabel("run"),
                                                            self.run_box, refresh_button]))
        layout.addWidget(self._group("Show", [self.show_cells, self.show_not_cells, self.show_density,
                                              load_button]))
        layout.addWidget(self._group("Numbers", [self.info]))
        layout.addWidget(self._group("Messages", [self.status]))
        layout.addWidget(QLabel("Tip: whole brains have millions of cells; zoom in to see the rings.\n"
                                "The 2D/3D button (bottom left) shows everything in 3D."))
        layout.addStretch()
        self.list_brains()

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

    def runs(self, brain):
        return sorted(p.parent.name for p in (self.results / brain / "full_brain").glob("*/all_candidates.csv"))

    def list_brains(self):
        brains = [b for b in common.brain_names(self.settings) if self.runs(b)]
        current = self.brain_box.currentText()
        self.brain_box.clear()
        self.brain_box.addItems(brains)
        if current in brains:
            self.brain_box.setCurrentText(current)
        self.list_runs()

    def list_runs(self):
        self.run_box.clear()
        if self.brain_box.currentText():
            self.run_box.addItems(self.runs(self.brain_box.currentText()))

    # ------------------------------------------------------------------
    def load(self):
        brain, run = self.brain_box.currentText(), self.run_box.currentText()
        if not brain or not run:
            self.say("No whole-brain results yet: run `cfos-mapper run-brain BRAIN --network NAME` first.")
            return
        folder = self.results / brain / "full_brain" / run
        self.say(f"loading {brain} {run} (millions of points: this takes a little while) ...")
        dataset, size = common.add_brain_layers(self.viewer, brain, self.settings)

        @thread_worker
        def work():
            calls = pd.read_csv(folder / "all_candidates.csv")
            density, density_scale = None, None
            if (folder / "cells_density.ome.zarr").exists():
                density_data = mm.ZarrManager(str(folder / "cells_density.ome.zarr"))
                density = np.squeeze(np.asarray(density_data.data[0]))
                density_scale = np.array(density_data.metadata["scales"][0], dtype=float)
            info = json.load(open(folder / "run_info.json")) if (folder / "run_info.json").exists() else {}
            return calls, density, density_scale, info

        def show(result):
            calls, density, density_scale, info = result
            common.remove_layers(self.viewer, self.my_layers)
            self.my_layers = []
            prefix = f"{brain} {run}"
            if density is not None and self.show_density.isChecked():
                name = f"{prefix}: cells per 50 µm cube"
                # a density pixel covers a whole cube: centre it on the cube
                translate = density_scale / 2 - np.array(size) / 2
                self.viewer.add_image(density, name=name, scale=density_scale, translate=translate,
                                      colormap="bop blue", blending="additive", opacity=0.7,
                                      contrast_limits=[0, float(np.percentile(density[density > 0], 99.5))])
                self.my_layers.append(name)
            for wanted, is_cell, label, color in [
                (self.show_not_cells.isChecked(), False, "not cells (network)", common.NETWORK_NOT_CELL),
                (self.show_cells.isChecked(), True, "cells (network)", common.NETWORK_CELL),
            ]:
                if not wanted:
                    continue
                name = f"{prefix}: {label}"
                points = calls[calls.is_cell == is_cell][["z", "y", "x"]].to_numpy(dtype=float)
                self.viewer.add_points(points, name=name, border_color=color, ndim=3, scale=size,
                                       **common.ring_style(14))
                self.my_layers.append(name)

            n_cells = int(calls.is_cell.sum())
            minutes = info.get("minutes", {})
            self.info.setText(
                f"{brain}, {run}\n"
                f"candidates: {info.get('candidates', len(calls)):,}\n"
                f"cells: {n_cells:,} ({100 * n_cells / max(len(calls), 1):.0f}% of classified)\n"
                f"not cells: {len(calls) - n_cells:,}\n"
                f"planes: {info.get('planes', '?')}\n"
                + (f"time: {sum(minutes.values()):.0f} min" if minutes else ""))
            full = np.array(dataset.data[0].shape[-3:]) * np.array(size)
            common.focus(self.viewer, [0, 0, 0], full)
            self.say(f"{brain} {run} loaded.")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR while loading: {e}"))
        worker.start()
