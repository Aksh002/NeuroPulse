"""Synthetic truth, partition, preprocessing and causality checks before real training."""

import io
import zipfile

import numpy as np
import pytest

from neuropulse.data import acquire as acquisition
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import read_engines, read_test_labels
from neuropulse.data.splits import split_engines, validation_prefixes
from neuropulse.data.targets import rul_targets
from neuropulse.data.windows import WindowDataset, terminal_window


def engine(key, length=12):
    rows = np.zeros((length, 26))
    rows[:, 0] = key
    rows[:, 1] = np.arange(1, length + 1)
    rows[:, 5:] = np.arange(length)[:, None] + np.arange(21)[None, :] + key
    rows[:, 5] = 7  # Constant sensor must be removed from training only.
    return rows


def test_reader_rejects_bad_values_order_and_width(tmp_path):
    good = engine(1)
    cases = [good[:, :-1], good.copy(), good.copy(), good[::-1], good.copy()]
    cases[1][4, 3] = np.nan
    cases[2][4, 1] = 4
    cases[4][0, 0] = 1.5
    for bad in cases:
        path = tmp_path / "bad.txt"
        np.savetxt(path, bad)
        with pytest.raises(ValueError):
            read_engines(path)
    np.savetxt(path, good)
    assert len(read_engines(path)[1]) == 12


def test_splits_and_endpoints_are_engine_level_and_repeatable():
    engines = {i: engine(i) for i in range(1, 11)}
    train, validation = split_engines(engines)
    assert len(train) == 8 and len(validation) == 2
    assert not set(train) & set(validation)
    assert set(train + validation) == set(engines)
    assert (train, validation) == split_engines(engines)
    assert validation_prefixes(engines, validation) == validation_prefixes(engines, validation)
    assert all(6 <= t < 12 for t in validation_prefixes(engines, validation).values())


def test_scaler_never_fits_validation_and_round_trips(tmp_path):
    engines = {i: engine(i) for i in range(1, 4)}
    scaler = SensorScaler.fit(engines, [1, 2])
    assert 0 not in scaler.selected
    engines[3][:, 5:] = 1e9
    unchanged = SensorScaler.fit(engines, [1, 2])
    np.testing.assert_array_equal(scaler.mean, unchanged.mean)
    scaler.save(tmp_path / "scaler.npz")
    restored = SensorScaler.load(tmp_path / "scaler.npz")
    np.testing.assert_array_equal(
        restored.transform(engines[1][:, 5:]), scaler.transform(engines[1][:, 5:])
    )
    assert restored.training_ids == (1, 2)


def test_targets_window_endpoint_and_engine_boundaries():
    np.testing.assert_array_equal(rul_targets(np.arange(1.0, 6.0), 2), [2, 2, 2, 1, 0])
    dataset = WindowDataset({1: engine(1, 5), 2: engine(2, 6)}, length=3, cap=None)
    assert len(dataset) == 7
    x, y = dataset[0]
    np.testing.assert_array_equal(x, engine(1, 5)[:3, 5:])
    np.testing.assert_array_equal(y, [2])
    assert dataset.index[3] == (2, 2)
    assert len(WindowDataset({1: engine(1, 2)}, length=3)) == 0


def test_short_terminal_padding_and_prefix_invariance():
    rows = engine(1, 8)
    window, padded = terminal_window(rows[:2, 5:], 4)
    assert padded
    np.testing.assert_array_equal(window[:3], np.repeat(rows[:1, 5:], 3, axis=0))
    before, _ = terminal_window(rows[:5, 5:], 4)
    rows[5:, 5:] = -999
    after, _ = terminal_window(rows[:5, 5:], 4)
    np.testing.assert_array_equal(before, after)


def test_label_alignment_rejected(tmp_path):
    path = tmp_path / "RUL.txt"
    path.write_text("4\n9\n")
    np.testing.assert_array_equal(read_test_labels(path, [1, 2]), [4, 9])
    for ids in ([1], [2, 1], [1, 3]):
        with pytest.raises(ValueError):
            read_test_labels(path, ids)


def test_archive_rejects_tampering():
    with pytest.raises(ValueError, match="checksum"):
        acquisition.inspect_archive(b"corrupt archive")


def test_archive_rejects_unexpected_members_even_with_valid_hashes(monkeypatch):
    inner_io = io.BytesIO()
    with zipfile.ZipFile(inner_io, "w") as inner:
        inner.writestr("../unsafe.txt", "malicious")
    nested = inner_io.getvalue()
    outer_io = io.BytesIO()
    with zipfile.ZipFile(outer_io, "w") as outer:
        outer.writestr(acquisition.NESTED_PATH, nested)
    payload = outer_io.getvalue()
    monkeypatch.setattr(acquisition, "ARCHIVE_SHA256", acquisition.sha256(payload))
    monkeypatch.setattr(acquisition, "NESTED_SHA256", acquisition.sha256(nested))
    with pytest.raises(ValueError, match="member"):
        acquisition.inspect_archive(payload)
