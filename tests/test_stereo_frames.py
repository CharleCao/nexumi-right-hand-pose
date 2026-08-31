from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nexumi_marker_pose.stereo_frames import (
    decode_frames_by_index,
    iter_stereo_frames,
    pair_stereo_timestamps,
    pair_timestamp_csvs,
    read_timestamp_csv,
)


DATASET_71 = Path(
    "/home/charlie/nexumi/0829-01-TOYS(71-80)/71/"
    "EGO_AZER76400DX_20260829_153130/part0001"
)
PREFIX_71 = "EGO_AZER76400DX_20260829_153130_camera"
LEFT_VIDEO_71 = DATASET_71 / f"{PREFIX_71}_left_part0001.mp4"
RIGHT_VIDEO_71 = DATASET_71 / f"{PREFIX_71}_right_part0001.mp4"
LEFT_PTS_71 = DATASET_71 / f"{PREFIX_71}_left_part0001_pts.csv"
RIGHT_PTS_71 = DATASET_71 / f"{PREFIX_71}_right_part0001_pts.csv"


def test_reads_timestamp_column_without_losing_integer_precision(tmp_path: Path) -> None:
    path = tmp_path / "pts.csv"
    path.write_text(
        "timestamp_us,ignored\n1788017494342860,x\n1788017494376210,y\n",
        encoding="utf-8",
    )

    assert read_timestamp_csv(path) == (1788017494342860, 1788017494376210)


def test_timestamp_reader_rejects_clock_regression(tmp_path: Path) -> None:
    path = tmp_path / "pts.csv"
    path.write_text("timestamp_us\n20\n19\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        read_timestamp_csv(path)


def test_nearest_pairing_is_one_to_one_and_survives_a_dropped_frame() -> None:
    # Left frame 0 is far from every right frame; pairing it greedily would
    # shift all following associations.  The nearest monotonic result skips it.
    report = pair_stereo_timestamps(
        (0, 50, 100, 150),
        (49, 101, 151),
        max_difference_us=5,
    )

    assert [(p.left_index, p.right_index) for p in report.pairs] == [
        (1, 0),
        (2, 1),
        (3, 2),
    ]
    assert report.deltas_us == (-1, 1, 1)
    assert report.unmatched_left_indices == (0,)
    assert report.unmatched_right_indices == ()
    assert report.max_abs_delta_us == 1
    assert report.mean_abs_delta_us == pytest.approx(1.0)


def test_maximum_difference_reports_unmatched_frames() -> None:
    report = pair_stereo_timestamps((0, 100), (40, 101), max_difference_us=5)

    assert [(p.left_index, p.right_index) for p in report.pairs] == [(1, 1)]
    assert report.unmatched_left_indices == (0,)
    assert report.unmatched_right_indices == (0,)


def test_pair_timestamp_csvs_on_real_71_capture() -> None:
    if not (LEFT_PTS_71.is_file() and RIGHT_PTS_71.is_file()):
        pytest.skip("EGO capture 71 is not available")

    report = pair_timestamp_csvs(LEFT_PTS_71, RIGHT_PTS_71, max_difference_us=100)

    assert len(report.pairs) == 562
    assert not report.unmatched_left_indices
    assert not report.unmatched_right_indices
    # The actual GPIO-synchronized capture differs by -56..+86 us; keeping the
    # assertion at the measured order of magnitude catches millisecond/unit
    # mistakes while allowing another export of the same recording.
    assert report.max_abs_delta_us is not None
    assert report.max_abs_delta_us < 100
    assert report.mean_abs_delta_us is not None
    assert report.mean_abs_delta_us < 50


def test_pyav_decodes_real_71_stereo_as_1600x1300_bgr() -> None:
    required = (LEFT_VIDEO_71, RIGHT_VIDEO_71, LEFT_PTS_71, RIGHT_PTS_71)
    if not all(path.is_file() for path in required):
        pytest.skip("EGO capture 71 is not available")
    pytest.importorskip("av")

    report = pair_timestamp_csvs(LEFT_PTS_71, RIGHT_PTS_71, max_difference_us=100)
    frames = iter_stereo_frames(
        LEFT_VIDEO_71,
        RIGHT_VIDEO_71,
        report,
        color_order="BGR",
    )

    count = 0
    for frame in frames:
        assert frame.color_order == "BGR"
        assert frame.left.shape == (1300, 1600, 3)
        assert frame.right.shape == (1300, 1600, 3)
        assert frame.left.dtype == np.uint8
        assert frame.right.dtype == np.uint8
        count += 1
    assert count == 562


def test_rgb_and_bgr_contract_on_same_real_frame() -> None:
    if not LEFT_VIDEO_71.is_file():
        pytest.skip("EGO capture 71 is not available")
    pytest.importorskip("av")

    bgr = next(
        decode_frames_by_index(
            LEFT_VIDEO_71, (0,), color_order="BGR", expected_size=(1600, 1300)
        )
    ).image
    rgb = next(
        decode_frames_by_index(
            LEFT_VIDEO_71, (0,), color_order="RGB", expected_size=(1600, 1300)
        )
    ).image

    assert np.array_equal(bgr[..., ::-1], rgb)
