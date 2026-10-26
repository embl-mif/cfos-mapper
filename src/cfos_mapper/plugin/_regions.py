"""
Widget 1 - training regions: show the saved test regions of a brain and add new ones.

A new region is drawn as a rectangle in the current view (that gives two axes).
The third axis takes the default size, centred on the current plane - or rotate
the view (napari's "roll dims" button) and draw a second rectangle to set it.
"""

import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

import cfos_mapper as cf

from . import _common as common

SAVED = "training regions"
NAMES = "training region names"
DRAW = "draw a new region (rectangle)"
PREVIEW = "new region (preview)"


class RegionsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = common.load_settings()
        self.brain = None
        self.dataset = None
        self.jobs = []   # background jobs (kept here while they run)

        self.brain_box = QComboBox()
        self.brain_box.addItems(common.brain_names(self.settings))
        load_button = QPushButton("Load brain")
        load_button.clicked.connect(self.load_brain)
        self.saved_label = QLabel("")
        self.saved_label.setWordWrap(True)

        self.size_boxes = []
        size_row = QHBoxLayout()
        for label, default in [("depth z", 300), ("height y", 480), ("width x", 480)]:
            box = QSpinBox()
            box.setRange(60, 5000)
            box.setValue(default)
            box.valueChanged.connect(self.update_preview)
            size_row.addWidget(QLabel(label))
            size_row.addWidget(box)
            self.size_boxes.append(box)
        draw_button = QPushButton("Draw a new region")
        draw_button.clicked.connect(self.start_drawing)
        self.new_label = QLabel("")
        self.new_label.setWordWrap(True)
        self.name_edit = QLineEdit()
        save_button = QPushButton("Save region")
        save_button.clicked.connect(self.save_region)

        self.remove_box = QComboBox()
        remove_button = QPushButton("Remove region from the list")
        remove_button.clicked.connect(self.remove_region)

        self.status = QLabel("Choose a brain and press 'Load brain'.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._group("Brain", [self.brain_box, load_button, self.saved_label]))
        new_group = self._group("New region", [
            QLabel("1. press 'Draw a new region', then drag a rectangle in the image.\n"
                   "2. the third axis takes the size below, centred on the current plane - or rotate the view "
                   "(napari 'roll dims' button) and draw a second rectangle to set it.\n"
                   "3. check the yellow preview, give a name, press 'Save region'.\n"
                   "Sizes in full-resolution voxels (z: 3 µm, y/x: 1.46 µm)."),
            draw_button])
        new_group.layout().addLayout(size_row)
        for w in [self.new_label, QLabel("name"), self.name_edit, save_button]:
            new_group.layout().addWidget(w)
        layout.addWidget(new_group)
        layout.addWidget(self._group("Remove", [self.remove_box, remove_button]))
        layout.addWidget(self._group("Messages", [self.status]))
        layout.addStretch()

    @staticmethod
    def _group(title, widgets):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        for w in widgets:
            if isinstance(w, QLabel):
                w.setWordWrap(True)
            layout.addWidget(w)
        return box

    def say(self, text):
        self.status.setText(text)
        print(text)

    # ------------------------------------------------------------------
    def load_brain(self):
        self.brain = self.brain_box.currentText()
        self.dataset, self.size = common.add_brain_layers(self.viewer, self.brain, self.settings)
        self.full_shape = np.array(self.dataset.data[0].shape[-3:])
        # boxes are drawn on the lowest resolution level (enough to see them, and small)
        self.low_level = len(self.dataset.data) - 1
        self.low_shape = np.array(self.dataset.data[self.low_level].shape[-3:])
        self.low_size = cf.voxel_size(self.dataset, self.low_level)
        self.show_saved()
        common.focus(self.viewer, [0, 0, 0], self.full_shape * self.size)
        self.say(f"{self.brain} loaded. Saved regions are outlined; their names are written at their centre.")

    def boxes_image(self, boxes):
        """Label image (lowest resolution) with box i filled with value i + 1."""
        labels = np.zeros(self.low_shape, dtype=np.uint8)
        factor = self.full_shape / self.low_shape
        for i, box in enumerate(boxes):
            lo = np.floor(np.array([b[0] for b in box]) / factor).astype(int)
            hi = np.ceil(np.array([b[1] for b in box]) / factor).astype(int)
            labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = i + 1
        return labels

    def show_saved(self):
        regions = common.load_regions().get(self.brain, {})
        boxes = [(r["z"], r["y"], r["x"]) for r in regions.values()]
        common.remove_layers(self.viewer, [SAVED, NAMES])
        layer = self.viewer.add_labels(self.boxes_image(boxes), name=SAVED, scale=self.low_size, opacity=0.8)
        layer.contour = 1
        centres = np.array([[np.mean(b) for b in box] for box in boxes]).reshape(-1, 3)
        self.viewer.add_points(centres, name=NAMES, scale=self.size, size=1, face_color="transparent",
                               border_width=0, projection_mode="all",
                               text=dict(string=list(regions), color="yellow", size=12))
        self.saved_label.setText(f"{len(regions)} saved region(s): " + ", ".join(regions))
        self.remove_box.clear()
        self.remove_box.addItems(list(regions))
        n = len(regions)
        self.name_edit.setText(f"region{chr(ord('A') + n)}_" if n < 26 else "region_")

    # ------------------------------------------------------------------
    def start_drawing(self):
        if self.brain is None:
            self.say("Load a brain first.")
            return
        common.remove_layers(self.viewer, [DRAW, PREVIEW])
        layer = self.viewer.add_shapes(ndim=3, name=DRAW, scale=self.size, edge_color="yellow",
                                       face_color="transparent", edge_width=2)
        layer.mode = "add_rectangle"
        layer.events.data.connect(lambda event: self.update_preview())
        self.viewer.layers.selection.active = layer
        self.say("Drag a rectangle in the image. Rotate the view and draw another one to set the third axis.")

    def drawn_box(self):
        """Box (start, end per axis, full-resolution voxels) from the drawn rectangles, or None."""
        if DRAW not in self.viewer.layers or len(self.viewer.layers[DRAW].data) == 0:
            return None
        points = np.vstack(self.viewer.layers[DRAW].data)
        box = []
        for axis in range(3):
            low, high = points[:, axis].min(), points[:, axis].max()
            if high - low < 1:  # not drawn along this axis: default size around the plane
                half = self.size_boxes[axis].value() / 2
                low, high = low - half, low + half
            low = int(np.clip(round(low), 0, self.full_shape[axis] - 1))
            high = int(np.clip(round(high), low + 1, self.full_shape[axis]))
            box.append((low, high))
        return box

    def update_preview(self):
        box = self.drawn_box()
        if box is None:
            return
        common.remove_layers(self.viewer, [PREVIEW])
        preview = self.viewer.add_labels(self.boxes_image([box]), name=PREVIEW, scale=self.low_size, opacity=0.9,
                                         colormap=DirectLabelColormap(color_dict={None: "transparent", 1: "yellow"}))
        preview.contour = 1
        if DRAW in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[DRAW]
        size_um = [(b[1] - b[0]) * s for b, s in zip(box, self.size)]
        self.new_label.setText("new region: " + ", ".join(f"{a} {b[0]}-{b[1]}" for a, b in zip("zyx", box))
                               + f"\n= {size_um[0]:.0f} × {size_um[1]:.0f} × {size_um[2]:.0f} µm")

    def save_region(self):
        box = self.drawn_box()
        name = self.name_edit.text().strip()
        if box is None:
            self.say("Draw a region first.")
            return
        regions = common.load_regions()
        if not name or name in regions.get(self.brain, {}):
            self.say(f"Choose a new name ('{name}' exists already for {self.brain}).")
            return
        regions.setdefault(self.brain, {})[name] = {axis: [int(b[0]), int(b[1])] for axis, b in zip("zyx", box)}
        common.save_regions(regions)
        common.remove_layers(self.viewer, [DRAW, PREVIEW])
        self.new_label.setText("")
        self.show_saved()
        brain = self.brain
        self.say(f"{brain} {name} saved. Cutting it out of the brain and finding candidates in a separate "
                 f"process (a few minutes)...")

        def done(ok, log):
            if ok:
                self.say(f"{brain} {name} is ready: annotate it in widget 2 (press 'Refresh lists' there).")
            else:
                self.say(f"Preparing {brain} {name} FAILED, see {log}")

        self.jobs.append(common.BackgroundJob(["prepare", brain, name], done))

    def remove_region(self):
        name = self.remove_box.currentText()
        if not name:
            return
        answer = QMessageBox.question(
            self, "Remove region", f"Remove {self.brain} {name} from the list of training regions?\n"
            "(its image, candidates and annotations stay on disk)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        regions = common.load_regions()
        regions.get(self.brain, {}).pop(name, None)
        common.save_regions(regions)
        self.show_saved()
        self.say(f"{self.brain} {name} removed from the list.")
