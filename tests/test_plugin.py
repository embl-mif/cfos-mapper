"""The napari widgets, without a screen (skipped when napari is not installed)."""

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("pytestqt")


@pytest.fixture
def viewer(project, monkeypatch, qtbot):
    from napari.components import ViewerModel

    monkeypatch.setenv("CFOS_MAPPER_SETTINGS", project["settings_file"])
    return ViewerModel()


def test_all_widgets_open(viewer, qtbot):
    from cfos_mapper.plugin import WIDGETS

    for widget_class, _ in WIDGETS:
        widget = widget_class(viewer)
        qtbot.addWidget(widget)


def test_regions_widget_draw_and_save(viewer, qtbot, project, monkeypatch):
    import cfos_mapper as cf
    from cfos_mapper.plugin import _common, _regions

    started = []
    monkeypatch.setattr(_common, "BackgroundJob", lambda args, done, **kw: started.append(args))
    widget = _regions.RegionsWidget(viewer)
    qtbot.addWidget(widget)
    widget.brain_box.setCurrentText("B1")
    widget.load_brain()
    assert "B1: cFos" in viewer.layers and _regions.SAVED in viewer.layers

    widget.start_drawing()
    rectangle = np.array([[20, 10, 10], [20, 10, 60], [20, 50, 60], [20, 50, 10]], float)
    viewer.layers[_regions.DRAW].add_rectangles(rectangle)
    box = widget.drawn_box()
    assert box[1] == (10, 50) and box[2] == (10, 60)

    widget.name_edit.setText("regionNew")
    widget.save_region()
    assert "regionNew" in cf.load_regions(project)["B1"]
    assert started == [["prepare", "B1", "regionNew"]]


def test_background_job_command_uses_the_settings(project, monkeypatch):
    from cfos_mapper.plugin import _common

    monkeypatch.setenv("CFOS_MAPPER_SETTINGS", project["settings_file"])
    command = _common.command("prepare", "B1", "regionA")
    assert command[1:3] == ["-m", "cfos_mapper"] and project["settings_file"] in command


def test_atlas_widget_registers_with_the_orientation_of_the_box(viewer, qtbot, project, monkeypatch):
    from cfos_mapper.plugin import _atlas, _common

    started = []
    monkeypatch.setattr(_common, "BackgroundJob", lambda args, done, **kw: started.append(args))
    widget = _atlas.AtlasWidget(viewer)
    qtbot.addWidget(widget)
    widget.brain_box.setCurrentText("B1")
    assert widget.orientation_edit.text() == project["atlas"]["orientation"]

    widget.orientation_edit.setText("xyz")
    assert "NOT VALID" in widget.orientation_text.text()
    widget.register()
    assert started == []

    widget.orientation_edit.setText("SAR")
    assert "the animal's right" in widget.orientation_text.text()
    widget.register()
    assert started == [["register", "B1", "--orientation", "sar"]]


def test_atlas_widget_show_brain_follows_the_orientation_box(viewer, qtbot, project):
    from cfos_mapper.plugin import _atlas

    widget = _atlas.AtlasWidget(viewer)
    qtbot.addWidget(widget)
    widget.brain_box.setCurrentText("B1")
    assert "s / i" in widget.orientation_edit.toolTip()
    widget.show_brain()
    assert "B1: cFos" in viewer.layers and viewer.axes.visible
    widget.orientation_edit.setText("pir")
    text = viewer.text_overlay.text
    assert "'pir'" in text and "posterior" in text and "the animal's right" in text
    widget.orientation_edit.setText("pi")
    assert "not valid" in viewer.text_overlay.text
