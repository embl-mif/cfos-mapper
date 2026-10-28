"""Figure helpers that do not need an atlas download: the Beryl regions."""

from types import SimpleNamespace

import numpy as np

from cfos_mapper.figures import regions as fr

VISP, VISP_LAYER_1, FIBER_TRACTS = 385, 593, 1009
STRUCTURES = {
    997: {"structure_id_path": [997]},
    VISP: {"structure_id_path": [997, 315, VISP]},
    VISP_LAYER_1: {"structure_id_path": [997, 315, VISP, VISP_LAYER_1]},
    FIBER_TRACTS: {"structure_id_path": [997, FIBER_TRACTS]},
}


def test_beryl_ids():
    assert len(fr.BERYL_IDS) == 306
    assert len(set(fr.BERYL_IDS)) == 306
    assert VISP in fr.BERYL_IDS and 997 not in fr.BERYL_IDS


def test_to_beryl():
    atlas = SimpleNamespace(structures=STRUCTURES)
    labels = np.array([[VISP_LAYER_1, VISP], [FIBER_TRACTS, 0]])
    assert fr.to_beryl(labels, atlas).tolist() == [[VISP, VISP], [0, 0]]
