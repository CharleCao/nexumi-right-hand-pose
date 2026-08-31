from __future__ import annotations

from pathlib import Path

import pytest

from nexumi_marker_pose.color_detection import detect_colored_triangles
from nexumi_marker_pose.ego_calibration import load_ego_stereo_calibration
from nexumi_marker_pose.marker_pose import estimate_marker_pose
from nexumi_marker_pose.stereo_frames import decode_frames_by_index


def test_real_session_71_frame_440_pose() -> None:
    root = Path("/home/charlie/nexumi/0829-01-TOYS(71-80)/71")
    ego = next(root.glob("EGO_*"), None)
    model = Path(__file__).parents[1] / "outputs/right_hand_marker.json"
    if ego is None or not model.is_file():
        pytest.skip("real session 71 or marker model is unavailable")
    part = ego / "part0001"
    left_video = next(part.glob("*_camera_left_*.mp4"))
    right_video = next(part.glob("*_camera_right_*.mp4"))
    calibration_path = next(ego.glob("*_calibration_camera.yaml"))
    left_image = next(decode_frames_by_index(left_video, [440])).image
    right_image = next(decode_frames_by_index(right_video, [440])).image
    left = detect_colored_triangles(left_image, roi=(850, 760, 350, 300))
    right = detect_colored_triangles(right_image, roi=(780, 760, 360, 300))

    estimate = estimate_marker_pose(
        load_ego_stereo_calibration(calibration_path), model, left, right
    )

    assert {item.sticker_id for item in estimate.assignments} == {
        "F0", "F1", "F3", "F6"
    }
    assert estimate.stereo_pair_count == 2
    assert estimate.pose.rmse_px < 2.0
    assert estimate.corner_rmse_px < 5.0
