"""
cfos-mapper napari plugin: four widgets (Plugins menu -> cFos mapper)

  1. Training regions      - see the saved test regions of a brain, draw new ones
  2. Annotate and train    - annotate cells / non-cells, run and train networks
  3. Whole-brain results   - brain, network calls and cell density of a whole-brain run
  4. Atlas regions         - register a brain to the atlas, cells per region

Start it with `cfos-mapper napari --settings settings.json` (all four widgets in
one window), or from napari's Plugins menu after setting CFOS_MAPPER_SETTINGS.
"""

from ._annotate import AnnotationWidget
from ._atlas import AtlasWidget
from ._regions import RegionsWidget
from ._results import ResultsWidget

WIDGETS = [(RegionsWidget, "1. Training regions"), (AnnotationWidget, "2. Annotate and train"),
           (ResultsWidget, "3. Whole-brain results"), (AtlasWidget, "4. Atlas regions")]


def open_viewer():
    """A napari window with the four widgets as tabs (opens on 'Training regions')."""
    import napari

    viewer = napari.Viewer(title="cFos mapper")
    docks = [viewer.window.add_dock_widget(widget(viewer), name=name, area="right", tabify=True)
             for widget, name in WIDGETS]
    docks[0].raise_()
    return viewer


__all__ = ["RegionsWidget", "AnnotationWidget", "ResultsWidget", "AtlasWidget", "open_viewer"]
