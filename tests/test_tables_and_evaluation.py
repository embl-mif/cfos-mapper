import numpy as np
import pandas as pd

import cfos_mapper as cf
from cfos_mapper.train import split_train_test


def calls():
    return pd.DataFrame({"z": [10, 20, 30, 40], "y": [10, 20, 30, 40], "x": [10, 20, 30, 40],
                         "is_cell": [True, True, False, False]})


def annotations():
    return pd.DataFrame({"z": [10, 21, 30, 40], "y": [10, 20, 30, 40], "x": [10, 20, 31, 40],
                         "label": ["cell", "not_cell", "cell", "not_cell"]})


def test_compare_with_annotations():
    counts = cf.compare_with_annotations(calls(), annotations())
    assert counts == dict(cells_found=1, cells_missed=1, false_positives=1, correctly_rejected=1)


def test_network_says_no_match_far_away():
    points = pd.DataFrame({"z": [10, 100], "y": [10, 100], "x": [10, 100]})
    assert list(cf.network_says(points, calls())) == ["cell", "no match"]


def test_prediction_statistics():
    row = cf.prediction_statistics(calls(), annotations(), plain_tissue=calls().iloc[2:])
    assert row["% called cell"] == 50.0
    assert row["% false positives on plain tissue"] == 0.0
    assert row["cells found (all annotations)"] == "1/2"


def test_annotations_round_trip(tmp_path):
    file = tmp_path / "a.csv"
    cf.save_annotations([[1, 2, 3]], [[4, 5, 6], [7, 8, 9]], file)
    table = cf.load_annotations(file)
    assert list(table.label) == ["cell", "not_cell", "not_cell"]
    assert len(cf.load_annotations(tmp_path / "missing.csv")) == 0


def test_remove_annotated():
    left = cf.remove_annotated(calls(), annotations().iloc[:1])
    assert len(left) == 3


def test_cells_tables_round_trip():
    table = calls()[["z", "y", "x"]]
    back = cf.cells_to_table(cf.table_to_cells(table))
    np.testing.assert_array_equal(back[["z", "y", "x"]].to_numpy(), table.to_numpy())


def test_split_train_test_keeps_the_ratio():
    table = pd.DataFrame({"z": range(20), "y": 0, "x": 0, "label": ["cell"] * 10 + ["not_cell"] * 10})
    train, test = split_train_test(table, 0.2)
    assert len(test) == 4 and (test.label == "cell").sum() == 2 and len(train) == 16


def test_border_and_random_positions(project):
    shape, size = (40, 160, 160), (3.0, 1.5, 1.5)
    candidates = pd.DataFrame({"z": [1, 20], "y": [80, 80], "x": [80, 80]})
    assert len(cf.remove_border_candidates(candidates, shape, size, project)) == 1
    spots = cf.random_positions((100, 400, 400), size, project, n=50)
    assert 0 < len(spots) <= 50
    far = cf.far_from_candidates(spots, spots.iloc[:1], size, min_distance_um=1)
    assert len(far) == len(spots) - 1


def test_brightness_correction_in_place():
    image = np.full((8, 4, 4), 1000, dtype=np.uint16)
    cf.correct_brightness_in_place(image, 70.0, threads=2)
    assert image.max() == 65535  # capped
    assert cf.correct_brightness(np.ones(3, np.uint16), 0.5).dtype == np.float32


def test_brightness_factors_saved(project):
    factors = cf.brightness_factors("B1", project)
    assert abs(factors["background"] - 700 / factors["tissue_median"]["background"]) < 1e-6
    assert cf.brightness_factors("B1", project) == factors  # read back from brightness.json
