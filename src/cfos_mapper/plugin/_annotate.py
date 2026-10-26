"""
Widget 2 - annotate and train.

  1. load a test region                (dropdown + "Load region")
  2. run a network on it               (dropdown + "Run network on this region")
     -> shows its prediction for every candidate, and statistics
     -> "Make report (all regions)": csv + png report of that network
  3. save your annotations             ("Save annotations")
  4. train a new network               (name + epochs + GPU + "Train new network")

Every candidate is shown as concentric rings (inside to outside):
  white                                the candidate
  solid green / solid red              your annotation: cell / not a cell
  light green / light red              network says cell / not a cell
  magenta                              you and the network disagree

Keys (act on the points selected in the "candidates" layer):
  c = cell, x = not a cell, d = delete the annotation
  n / Shift+N = select the next / previous disagreement between you and the network

Region layers are placed at their position in the whole brain, so they line up
with the brain loaded in the other widgets.
"""

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTextBrowser, QVBoxLayout, QWidget,
)
from scipy.spatial import cKDTree

import cfos_mapper as cf
from cfos_mapper.gpu import gpu_count

from . import _common as common

YES, NO, CANCEL = (QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No,
                   QMessageBox.StandardButton.Cancel)

CANDIDATES = "candidates"
DISAGREE = "disagreements"
RINGS = {                                                   # from outside to inside (top layer = inner)
    DISAGREE: ("magenta", 42),                              # you and the network disagree
    "network: cell": (common.NETWORK_CELL, 34),             # light green
    "network: not cell": (common.NETWORK_NOT_CELL, 34),     # light red
    "annotation: cell": (common.ANNOTATION_CELL, 27),       # solid green
    "annotation: not cell": (common.ANNOTATION_NOT_CELL, 27),  # solid red
    CANDIDATES: ("white", 20),                              # the candidate
}
IMAGES = ["region: autofluorescence", "region: cFos"]

INSTRUCTIONS = """
<h3>How to use: annotate and train</h3>

<p><b>Rings</b> - every candidate (bright, cell-sized spot) is shown as concentric rings, from inside to outside:<br>
&nbsp;&nbsp;1. <b>white</b> = the candidate (always there)<br>
&nbsp;&nbsp;2. your annotation:
<span style="color:green"><b>solid green</b></span> = cell,
<span style="color:red"><b>solid red</b></span> = not a cell<br>
&nbsp;&nbsp;3. what the network says:
<span style="color:#5cb85c"><b>light green</b></span> = cell,
<span style="color:#e08080"><b>light red</b></span> = not a cell<br>
&nbsp;&nbsp;4. <span style="color:magenta"><b>magenta</b></span> = you and the network disagree<br>
Rings are drawn like spheres: largest on the cell's own plane. Scroll through z with the slider.</p>

<p><b>1. Region</b> - choose a test region and press <i>Load region</i>. New regions made in widget 1
appear after <i>Refresh lists</i>. The region is shown at its place in the brain.</p>

<p><b>Annotating</b> - in the layer list, click the <b>candidates</b> layer, choose the select tool (arrow),
click a white ring (or drag a box around several), then press<br>
&nbsp;&nbsp;<b>c</b> = cell &nbsp;&nbsp; <b>x</b> = not a cell &nbsp;&nbsp; <b>d</b> = delete the annotation<br>
<b>c</b> / <b>x</b> on an annotated candidate changes it. The white ring never disappears.
Skip candidates you are unsure about. Aim for similar numbers of cells and non-cells,
and include difficult non-cells (dim blobs, haze, vessels).</p>

<p><b>2. Network</b> - choose a network (<i>pretrained</i> or one of ours) and press
<i>Run network on this region</i> (about 1 minute the first time, then instant): the light rings
appear, and a magenta ring wherever the network disagrees with you.<br>
<b>n</b> / <b>Shift+N</b> = select the next / previous disagreement. If the network is right,
press <b>c</b>/<b>x</b> to change your annotation; otherwise press <b>n</b> again.<br>
The statistics under the button tell you how good the network is on this region;
<i>not used in training</i> is the honest check.
<i>Make report (all regions)</i> saves a table (csv + png) and a picture per region in
<code>reports/&lt;network&gt;/</code>.</p>

<p><b>3. Save annotations</b> - saves this region's annotations
(<code>results/&lt;brain&gt;/test_regions/&lt;region&gt;_annotations.csv</code>). Save often.
Next time your annotations are loaded again, so you can continue over several days.</p>

<p><b>4. Train</b> - type a name for the new network (e.g. network_v3), choose epochs (150) and GPU (0, 1, ...),
press <i>Train new network</i>. It uses the saved annotations of all regions, keeps 20% aside
for testing, and runs in the background (30-60 min); you can keep annotating or close the window.
When it is finished it appears in the network list.</p>

<p><b>Typical round:</b> load a region &rarr; run the latest network &rarr; go through the
disagreements with <b>n</b> &rarr; annotate more &rarr; save &rarr; repeat for other regions/brains
&rarr; train a new network &rarr; make its report.</p>
"""


def format_statistics(row):
    """Statistics dict -> a few readable lines for the panel."""
    lines = [f"candidates called cell: {row['% called cell']}% of {row['candidates classified']}"]
    if "% false positives on plain tissue" in row:
        lines.append(f"false positives on plain tissue: {row['% false positives on plain tissue']}%")
    for group in ["all annotations", "not used in training"]:
        if f"cells found ({group})" in row:
            lines.append(f"{group}: cells found {row[f'cells found ({group})']} "
                         f"({row[f'% cells found ({group})']}%), non-cells called cell "
                         f"{row[f'non-cells called cell ({group})']} ({row[f'% false positives ({group})']}%)")
    return "\n".join(lines)


class AnnotationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = common.load_settings()
        self.results = Path(self.settings["results_folder"])
        self.networks = Path(self.settings["networks_folder"])
        self.reports = Path(self.settings["reports_folder"])
        self.region = None             # (brain, region) currently loaded
        self.points = np.empty((0, 3))  # candidate positions (z, y, x), region voxels
        self.labels = np.array([], dtype=object)       # your annotation per candidate: "", "cell", "not_cell"
        self.predictions = np.array([], dtype=object)  # network per candidate: "", "cell", "not_cell"
        self.extra = pd.DataFrame(columns=["z", "y", "x", "label"])  # annotations not on a candidate (kept)
        self.saved_labels = None
        self.current = -1               # candidate index of the last disagreement visited
        self.training = None            # (process, name, log file)
        self.help_window = None
        self.jobs = []                  # background GPU jobs (kept here while they run)

        # ---- widgets ----
        help_button = QPushButton("How to use")
        help_button.clicked.connect(self.show_instructions)
        self.region_box = QComboBox()
        load_button = QPushButton("Load region")
        load_button.clicked.connect(self.load_region)
        refresh_button = QPushButton("Refresh lists")
        refresh_button.clicked.connect(self.refresh_lists)
        self.network_box = QComboBox()
        run_button = QPushButton("Run network on this region")
        run_button.clicked.connect(self.run_network)
        self.statistics = QLabel("")
        self.statistics.setWordWrap(True)
        self.report_button = QPushButton("Make report (all regions)")
        self.report_button.clicked.connect(self.make_report)
        save_button = QPushButton("Save annotations")
        save_button.clicked.connect(self.save)
        self.counts = QLabel("")
        self.counts.setWordWrap(True)
        self.name_edit = QLineEdit()
        self.epochs_box = QSpinBox()
        self.epochs_box.setRange(5, 1000)
        self.epochs_box.setValue(150)
        # GPUs of this machine; the last of the first two by default (GPU 0 is often busy with run-brain)
        gpus = gpu_count()
        self.gpu_box = QSpinBox()
        self.gpu_box.setRange(0, max(gpus - 1, 0))
        self.gpu_box.setValue(min(1, max(gpus - 1, 0)))
        if gpus == 0:
            self.gpu_box.setEnabled(False)
            self.gpu_box.setToolTip("no NVIDIA GPU found: training runs on the CPU (many times slower)")
        self.train_button = QPushButton("Train new network")
        self.train_button.clicked.connect(self.train)
        self.status = QLabel("Choose a region and press 'Load region'.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(help_button)
        region_group = self._group("1. Region", [self.region_box])
        row = QHBoxLayout()
        row.addWidget(load_button)
        row.addWidget(refresh_button)
        region_group.layout().addLayout(row)
        layout.addWidget(region_group)
        layout.addWidget(self._group("2. Network", [self.network_box, run_button, self.statistics,
                                                    self.report_button]))
        layout.addWidget(self._group("3. Annotations", [self.counts, save_button]))
        train_row = QHBoxLayout()
        for label, widget in [("epochs", self.epochs_box), ("GPU", self.gpu_box)]:
            train_row.addWidget(QLabel(label))
            train_row.addWidget(widget)
        train_group = self._group("4. Train", [QLabel("new network name"), self.name_edit])
        train_group.layout().addLayout(train_row)
        train_group.layout().addWidget(self.train_button)
        layout.addWidget(train_group)
        layout.addWidget(self._group("Messages", [self.status]))
        layout.addWidget(QLabel("Keys (on selected candidates): c = cell, x = not a cell,\n"
                                "d = delete annotation, n / Shift+N = next / previous disagreement"))
        layout.addStretch()

        self.refresh_lists()
        self.viewer.bind_key("c", lambda v: self.annotate("cell"), overwrite=True)
        self.viewer.bind_key("x", lambda v: self.annotate("not_cell"), overwrite=True)
        self.viewer.bind_key("d", lambda v: self.annotate(""), overwrite=True)
        self.viewer.bind_key("n", lambda v: self.next_disagreement(1), overwrite=True)
        self.viewer.bind_key("Shift-N", lambda v: self.next_disagreement(-1), overwrite=True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_training)
        self.timer.start(10_000)

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
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

    def show_instructions(self):
        """Instructions in a separate window that does not block napari."""
        if self.help_window is None:
            text = QTextBrowser()
            text.setHtml(INSTRUCTIONS)
            self.help_window = QWidget()  # own top-level window
            self.help_window.setWindowTitle("How to use: annotate and train")
            QVBoxLayout(self.help_window).addWidget(text)
            self.help_window.resize(620, 780)
        self.help_window.show()
        self.help_window.raise_()

    def all_regions(self):
        regions = common.load_regions()
        return [(b, r) for b in regions if not b.startswith("_") for r in regions[b]]

    def refresh_lists(self):
        current = self.region_box.currentText()
        self.region_box.clear()
        self.region_box.addItems([f"{b} / {r}" for b, r in self.all_regions()])
        if current:
            self.region_box.setCurrentText(current)

        names = sorted(p.parent.name for p in self.networks.glob("*/network.keras"))
        current = self.network_box.currentText()
        self.network_box.clear()
        self.network_box.addItems(["pretrained"] + names)
        self.network_box.setCurrentText(current or (names[-1] if names else "pretrained"))
        n = 1
        while (self.networks / f"network_v{n}").exists():
            n += 1
        self.name_edit.setText(f"network_v{n}")

    def candidates_table(self):
        return pd.DataFrame(self.points, columns=["z", "y", "x"])

    def my_annotations(self):
        """Current annotations (including unsaved changes) as a table z, y, x, label."""
        annotated = self.labels != ""
        table = self.candidates_table()[annotated].assign(label=self.labels[annotated])
        return pd.concat([table, self.extra], ignore_index=True)

    def disagreements(self):
        return np.flatnonzero((self.labels != "") & (self.predictions != "") & (self.labels != self.predictions))

    def update_rings(self):
        """Redraw the annotation, network and disagreement rings from self.labels / self.predictions."""
        extra = self.extra
        content = {
            "annotation: cell": np.vstack([self.points[self.labels == "cell"],
                                           extra[extra.label == "cell"][["z", "y", "x"]].to_numpy(float)]),
            "annotation: not cell": np.vstack([self.points[self.labels == "not_cell"],
                                               extra[extra.label == "not_cell"][["z", "y", "x"]].to_numpy(float)]),
            "network: cell": self.points[self.predictions == "cell"],
            "network: not cell": self.points[self.predictions == "not_cell"],
            DISAGREE: self.points[self.disagreements()],
        }
        for name, data in content.items():
            if name in self.viewer.layers:
                self.viewer.layers[name].data = np.asarray(data, dtype=float).reshape(-1, 3)
        self.update_counts()

    def unsaved_changes(self):
        return self.saved_labels is not None and not np.array_equal(self.labels, self.saved_labels)

    def update_counts(self):
        mine = self.my_annotations()
        total = [pd.read_csv(f) for f in self.results.glob("*/test_regions/*_annotations.csv")]
        total = pd.concat(total) if total else pd.DataFrame(columns=["label"])
        self.counts.setText(
            f"this region: {(mine.label == 'cell').sum()} cells, {(mine.label == 'not_cell').sum()} not cells"
            f"{' (unsaved changes)' if self.unsaved_changes() else ''}\n"
            f"all regions (saved): {(total.label == 'cell').sum()} cells, "
            f"{(total.label == 'not_cell').sum()} not cells")

    def statistics_for(self, brain, region, network, predictions, annotations):
        """Statistics of a network on one region (see cf.prediction_statistics)."""
        plain_tissue = cf.far_from_candidates(predictions["control"], predictions["candidates"],
                                              predictions["size"])
        trained_on = None
        if network != "pretrained":
            trained_on = cf.used_in_training(annotations, self.networks / network, brain, region)
        return cf.prediction_statistics(predictions["calls"], annotations, plain_tissue, trained_on)

    # ------------------------------------------------------------------
    # 1. load a region
    # ------------------------------------------------------------------
    def load_region(self):
        if self.unsaved_changes():
            answer = QMessageBox.question(self, "Unsaved annotations",
                                          "Save the annotations of the current region first?",
                                          YES | NO | CANCEL)
            if answer == CANCEL:
                return
            if answer == YES:
                self.save()
        if not self.region_box.currentText():
            self.say("No regions: make one in widget 1, then press 'Refresh lists'.")
            return
        brain, region = [s.strip() for s in self.region_box.currentText().split("/")]
        if not cf.region_ready(brain, region, self.settings):
            # new region: cut it out + find candidates in a separate process, then load it
            self.say(f"{brain} {region} is not prepared yet: cutting it out and finding candidates in a separate "
                     f"process (a few minutes). It opens by itself when ready.")

            def done(ok, log):
                if ok:
                    self.region_box.setCurrentText(f"{brain} / {region}")
                    self.load_region()
                else:
                    self.say(f"Preparing {brain} {region} FAILED, see {log}")

            self.jobs.append(common.BackgroundJob(["prepare", brain, region], done))
            return
        self.say(f"loading {brain} {region} ...")

        @thread_worker
        def work():
            signal, background, size, candidates = cf.prepare_region(brain, region, self.settings)
            candidates = cf.remove_border_candidates(candidates, signal.shape, size, self.settings)
            annotations = cf.load_annotations(self.results / brain / "test_regions" / f"{region}_annotations.csv")
            return signal, background, size, candidates, annotations

        def show(result):
            signal, background, self.size, candidates, annotations = result
            self.region = (brain, region)
            box = common.load_regions()[brain][region]
            self.start = np.array([box["z"][0], box["y"][0], box["x"][0]], dtype=float)
            translate = self.start * np.array(self.size)   # place the region in the whole brain
            self.points = candidates[["z", "y", "x"]].to_numpy(dtype=float)
            self.labels = np.full(len(self.points), "", dtype=object)
            self.predictions = np.full(len(self.points), "", dtype=object)

            # each saved annotation -> its candidate (nearest within 3 voxels); others are kept as they are
            matched = np.zeros(len(annotations), dtype=bool)
            if len(annotations) and len(self.points):
                distance, index = cKDTree(self.points).query(annotations[["z", "y", "x"]].to_numpy(dtype=float))
                matched = distance <= 3
                self.labels[index[matched]] = annotations.label.to_numpy()[matched]
            self.extra = annotations[~matched][["z", "y", "x", "label"]].reset_index(drop=True)
            self.saved_labels = self.labels.copy()
            self.current = -1

            common.remove_layers(self.viewer, IMAGES + list(RINGS))
            self.viewer.add_image(background, name=IMAGES[0], scale=self.size, translate=translate,
                                  colormap="green", blending="additive", visible=False)
            self.viewer.add_image(signal, name=IMAGES[1], scale=self.size, translate=translate,
                                  colormap="gray", blending="additive")
            # the ring layers are only a display (redrawn from self.labels / self.predictions);
            # annotate by selecting points in the candidates layer
            for name, (color, size) in RINGS.items():
                self.viewer.add_points(np.empty((0, 3)), name=name, border_color=color, ndim=3,
                                       scale=self.size, translate=translate, **common.ring_style(size))
            self.viewer.layers[CANDIDATES].data = self.points
            self.viewer.layers.selection.active = self.viewer.layers[CANDIDATES]
            self.viewer.layers[CANDIDATES].mode = "select"
            self.update_rings()
            self.statistics.setText("")
            common.focus(self.viewer, translate, translate + np.array(signal.shape) * np.array(self.size))
            self.say(f"{brain} {region}: {len(self.points)} candidates, {int((self.labels != '').sum())} "
                     f"annotated. Press 'Run network on this region' to see what a network says.")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR while loading: {e}"))
        worker.start()

    # ------------------------------------------------------------------
    # annotating: keys c, x, d (on the candidates selected in the candidates layer)
    # ------------------------------------------------------------------
    def annotate(self, label):
        if self.region is None or CANDIDATES not in self.viewer.layers:
            return
        layer = self.viewer.layers[CANDIDATES]
        selected = sorted(layer.selected_data)
        if not selected:
            self.say("Select candidates first: 'candidates' layer, select tool (arrow), click a white ring.")
            return
        self.labels[selected] = label
        self.update_rings()
        self.viewer.layers.selection.active = layer
        layer.selected_data = set(selected)  # keep them selected
        what = {"cell": "cell", "not_cell": "not a cell", "": "annotation deleted"}[label]
        self.say(f"{len(selected)} candidate(s): {what}")

    def next_disagreement(self, step=1):
        """Select the next (step=1) or previous (step=-1) disagreement in the candidates layer."""
        if self.region is None or CANDIDATES not in self.viewer.layers:
            return
        found = self.disagreements()
        if len(found) == 0:
            self.say("No disagreements (run a network first, or you agree everywhere).")
            return
        if step > 0:
            later = found[found > self.current]
            self.current = later[0] if len(later) else found[0]
        else:
            earlier = found[found < self.current]
            self.current = earlier[-1] if len(earlier) else found[-1]
        i = int(self.current)
        z, y, x = (self.points[i] + self.start) * np.array(self.size)   # world coordinates
        self.viewer.dims.set_point(0, z)
        self.viewer.camera.center = (y, x)
        layer = self.viewer.layers[CANDIDATES]
        self.viewer.layers.selection.active = layer
        layer.mode = "select"
        layer.selected_data = {i}
        position = int(np.flatnonzero(found == i)[0]) + 1
        self.say(f"Disagreement {position}/{len(found)}: you said {self.labels[i]}, network says "
                 f"{self.predictions[i]}. Press c / x to change it, d to delete, n / Shift+N to move on.")

    # ------------------------------------------------------------------
    # 2. run a network on this region + statistics, and reports
    # ------------------------------------------------------------------
    def run_network(self):
        if self.region is None:
            self.say("Load a region first.")
            return
        brain, region = self.region
        network = self.network_box.currentText()
        if not cf.predictions_ready(brain, region, network, self.settings):
            # first time: the network runs in a separate process (GPU), then the result is shown
            self.say(f"running '{network}' on {brain} {region} in a separate process (about 1 minute) ...")

            def done(ok, log):
                if not ok:
                    self.say(f"Running '{network}' on {brain} {region} FAILED, see {log}")
                elif self.region == (brain, region):
                    self.network_box.setCurrentText(network)
                    self.run_network()

            self.jobs.append(common.BackgroundJob(["predict", brain, region, network], done))
            return
        self.say(f"showing '{network}' on {brain} {region} ...")

        @thread_worker
        def work():
            return cf.get_predictions(brain, region, network, self.settings, max_workers=0)

        def show(predictions):
            if self.region != (brain, region):
                return  # another region was loaded meanwhile
            says = cf.network_says(self.candidates_table(), predictions["calls"])
            self.predictions = np.where(says == "no match", "", says).astype(object)
            self.current = -1
            self.update_rings()
            row = self.statistics_for(brain, region, network, predictions, self.my_annotations())
            self.statistics.setText(f"'{network}' on {brain} {region}:\n" + format_statistics(row))
            self.say(f"'{network}': light green = says cell, light red = says not cell. It disagrees with "
                     f"{len(self.disagreements())} of your annotations (magenta): press n to go through them.")

        worker = work()
        worker.returned.connect(show)
        worker.errored.connect(lambda e: self.say(f"ERROR while running the network: {e}"))
        worker.start()

    def make_report(self):
        """csv + png report of the selected network on ALL test regions (saved annotations)."""
        network = self.network_box.currentText()
        out = self.reports / network
        missing = [(b, r) for b, r in self.all_regions()
                   if not (cf.region_ready(b, r, self.settings) and cf.predictions_ready(b, r, network, self.settings))]
        if missing:
            # first run the network on every region in a separate process (GPU), then make the report
            self.report_button.setEnabled(False)
            self.say(f"running '{network}' on {len(missing)} region(s) in a separate process first ...")

            def done(ok, log):
                self.report_button.setEnabled(True)
                if ok:
                    self.network_box.setCurrentText(network)
                    self.make_report()
                else:
                    self.say(f"Running '{network}' on all regions FAILED, see {log}")

            self.jobs.append(common.BackgroundJob(["predict-all", network], done))
            return
        self.report_button.setEnabled(False)

        @thread_worker
        def work():
            out.mkdir(parents=True, exist_ok=True)
            rows = []
            for brain, region in self.all_regions():
                yield f"report '{network}': {brain} {region} ..."
                predictions = cf.get_predictions(brain, region, network, self.settings,
                                                 max_workers=0)
                annotations = cf.load_annotations(self.results / brain / "test_regions" / f"{region}_annotations.csv")
                rows.append({"brain": brain, "region": region,
                             **self.statistics_for(brain, region, network, predictions, annotations)})
                fig = Figure(figsize=(8, 8))
                cf.draw_calls(fig.add_subplot(), predictions["signal"], predictions["calls"],
                              f"{network}: {brain} {region}")
                fig.tight_layout()
                fig.savefig(out / f"{brain}_{region}.png", dpi=60)
            table = pd.DataFrame(rows)
            table.to_csv(out / "report.csv", index=False)
            fig = Figure(figsize=(22, 2 + 0.6 * len(table)))
            ax = fig.add_subplot()
            ax.axis("off")
            t = ax.table(cellText=table.fillna("-").values,
                         colLabels=[c.replace(" (", "\n(").replace("% ", "%\n", 1).replace(" on plain", "\non plain")
                                    for c in table.columns],
                         loc="center", cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1, 3)
            ax.set_title(f"{network}: statistics per test region ({datetime.now():%Y-%m-%d %H:%M}). "
                         f"'not used in training' = honest check.")
            fig.savefig(out / "report.png", dpi=80, bbox_inches="tight")
            return out

        def done(folder):
            self.report_button.setEnabled(True)
            self.say(f"Report of '{network}' saved in {folder} (report.png, report.csv, one picture per region).")

        def failed(error):
            self.report_button.setEnabled(True)
            self.say(f"ERROR while making the report: {error}")

        worker = work()
        worker.yielded.connect(self.say)
        worker.returned.connect(done)
        worker.errored.connect(failed)
        worker.start()

    # ------------------------------------------------------------------
    # 3. save
    # ------------------------------------------------------------------
    def save(self):
        if self.region is None:
            self.say("Nothing to save: load a region first.")
            return
        brain, region = self.region
        file = self.results / brain / "test_regions" / f"{region}_annotations.csv"
        mine = self.my_annotations()
        cf.save_annotations(mine[mine.label == "cell"][["z", "y", "x"]].to_numpy(),
                            mine[mine.label == "not_cell"][["z", "y", "x"]].to_numpy(), file)
        self.saved_labels = self.labels.copy()
        self.update_counts()
        self.say(f"Saved {brain} {region}: {(mine.label == 'cell').sum()} cells, "
                 f"{(mine.label == 'not_cell').sum()} not cells -> {file}")

    # ------------------------------------------------------------------
    # 4. train (runs `cfos-mapper train` in the background)
    # ------------------------------------------------------------------
    def train(self):
        if self.training is not None:
            self.say(f"Training of '{self.training[1]}' is still running (log: {self.training[2]}).")
            return
        name = self.name_edit.text().strip()
        if not name or (self.networks / name).exists():
            self.say(f"Choose a new network name ('{name}' exists already).")
            return
        if self.unsaved_changes():
            if QMessageBox.question(self, "Unsaved annotations", "Save the current annotations before training?",
                                    YES | NO) == YES:
                self.save()
        log = common.log_file(f"train_{name}")
        gpu = ["--gpu", self.gpu_box.value()] if self.gpu_box.isEnabled() else []
        command = common.command("train", name, *gpu, "--epochs", self.epochs_box.value())
        process = subprocess.Popen(command, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                   start_new_session=True)  # keeps running if this window is closed
        self.training = (process, name, log)
        self.train_button.setEnabled(False)
        device = f"GPU {self.gpu_box.value()}" if gpu else "no GPU found: on the CPU, slow"
        self.say(f"Training '{name}' started in the background ({self.epochs_box.value()} epochs, "
                 f"{device}). You can keep annotating. Log: {log}")

    def check_training(self):
        if self.training is None:
            return
        process, name, log = self.training
        if process.poll() is None:
            return
        self.training = None
        self.train_button.setEnabled(True)
        if process.returncode == 0:
            self.refresh_lists()
            self.network_box.setCurrentText(name)
            self.say(f"Training of '{name}' finished. Results: {self.networks / name / 'test_results.png'}. "
                     f"It is now selected in 2.: press 'Run network on this region' or 'Make report'.")
        else:
            self.say(f"Training of '{name}' FAILED, see {log}")
