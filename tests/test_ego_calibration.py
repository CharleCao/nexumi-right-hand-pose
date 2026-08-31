from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nexumi_marker_pose.ego_calibration import load_ego_stereo_calibration


def _write_calibration(path: Path, *, model: str = "KB", cam1_tx: float = -120.0) -> None:
    path.write_text(
        f"""\
calibration_info:
  num_cameras: 2
  serial_number: TEST123
  reference_camera: cam_0
cameras:
  - id: cam_0
    name: left
    distortion_model: {model}
    image_width: 1600
    image_height: 1300
    intrinsics: {{fx: 500.0, fy: 501.0, cx: 800.0, cy: 650.0}}
    distortion: {{k1: 0.07, k2: -0.012, k3: 0.001, k4: -0.0002, k5: 0, k6: 0, p1: 0, p2: 0}}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [0, 0, 0]
  - id: cam_1
    name: right
    distortion_model: KB
    image_width: 1600
    image_height: 1300
    intrinsics: {{fx: 502.0, fy: 503.0, cx: 802.0, cy: 648.0}}
    distortion: {{k1: 0.071, k2: -0.013, k3: 0.0012, k4: -0.0002, k5: 0, k6: 0, p1: 0, p2: 0}}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [{cam1_tx}, -0.5, 0.25]
""",
        encoding="utf-8",
    )


def test_loads_units_extrinsic_semantics_and_opencv_parameters(tmp_path: Path) -> None:
    calibration_path = tmp_path / "camera.yaml"
    _write_calibration(calibration_path)

    stereo = load_ego_stereo_calibration(calibration_path)

    assert stereo.translation_unit == "mm"
    assert stereo.serial_number == "TEST123"
    assert stereo.baseline_mm == pytest.approx(np.linalg.norm([-120.0, -0.5, 0.25]))
    assert stereo.t_cam1_cam0_m == pytest.approx([-0.120, -0.0005, 0.00025])
    point_cam0_mm = np.array([10.0, 20.0, 1000.0])
    assert stereo.transform_cam0_to_cam1(point_cam0_mm) == pytest.approx(
        [-110.0, 19.5, 1000.25]
    )
    recovered = (
        stereo.T_cam0_cam1_mm
        @ np.append(stereo.transform_cam0_to_cam1(point_cam0_mm), 1.0)
    )
    assert recovered[:3] == pytest.approx(point_cam0_mm)

    K, D, image_size = stereo.cam0.opencv_fisheye_parameters()
    np.testing.assert_allclose(
        K, [[500.0, 0.0, 800.0], [0.0, 501.0, 650.0], [0, 0, 1]]
    )
    assert D.shape == (4, 1)
    assert D[:, 0] == pytest.approx([0.07, -0.012, 0.001, -0.0002])
    assert image_size == (1600, 1300)


def test_kb_projection_and_unprojection_round_trip(tmp_path: Path) -> None:
    calibration_path = tmp_path / "camera.yaml"
    _write_calibration(calibration_path)
    camera = load_ego_stereo_calibration(calibration_path).cam0

    points = np.array(
        [[0.0, 0.0, 1.0], [0.25, -0.15, 1.0], [-0.8, 0.3, 0.7]],
        dtype=np.float64,
    )
    pixels = camera.project(points)
    rays = camera.unproject(pixels)
    expected_rays = points / np.linalg.norm(points, axis=1, keepdims=True)

    assert pixels[0] == pytest.approx([camera.cx, camera.cy])
    assert rays == pytest.approx(expected_rays, abs=1.0e-11)
    assert camera.unproject([camera.cx, camera.cy]) == pytest.approx([0.0, 0.0, 1.0])


def test_rejects_invalid_schema_and_zero_baseline(tmp_path: Path) -> None:
    bad_model = tmp_path / "bad_model.yaml"
    _write_calibration(bad_model, model="pinhole")
    with pytest.raises(ValueError, match="distortion_model"):
        load_ego_stereo_calibration(bad_model)

    zero_baseline = tmp_path / "zero_baseline.yaml"
    _write_calibration(zero_baseline, cam1_tx=0.0)
    # The two smaller translation components still make this a non-zero baseline.
    text = zero_baseline.read_text(encoding="utf-8").replace(
        "translation: [0.0, -0.5, 0.25]", "translation: [0, 0, 0]"
    )
    zero_baseline.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        load_ego_stereo_calibration(zero_baseline)

    with pytest.raises(FileNotFoundError):
        load_ego_stereo_calibration(tmp_path / "missing.yaml")


def test_real_session_71_calibration() -> None:
    matches = list(
        Path("/home/charlie/nexumi/0829-01-TOYS(71-80)/71").glob(
            "EGO_*/*_calibration_camera.yaml"
        )
    )
    if not matches:
        pytest.skip("session 71 EGO camera calibration is not available")

    stereo = load_ego_stereo_calibration(matches[0])

    assert stereo.cam0.name == "IR_L"
    assert stereo.cam1.name == "IR_R"
    assert stereo.cam0.image_width == stereo.cam1.image_width == 1600
    assert stereo.cam0.image_height == stereo.cam1.image_height == 1300
    assert stereo.baseline_mm == pytest.approx(120.5215, abs=0.002)
    assert stereo.t_cam1_cam0_mm == pytest.approx(
        [-120.51941854, -0.68769444, -0.18432584]
    )
    assert np.linalg.det(stereo.R_cam1_cam0) == pytest.approx(1.0, abs=2.0e-5)

    pixels = np.array([[800.0, 650.0], [0.0, 0.0], [1599.0, 1299.0]])
    rays = stereo.cam0.unproject(pixels)
    assert np.linalg.norm(rays, axis=1) == pytest.approx(np.ones(3), abs=1.0e-12)
