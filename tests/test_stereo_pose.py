from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nexumi_marker_pose.ego_calibration import EGOStereoCalibration, KBFisheyeCamera
from nexumi_marker_pose.stereo_pose import PoseObservation, solve_stereo_pose


def _calibration() -> EGOStereoCalibration:
    def camera(camera_id: str, translation) -> KBFisheyeCamera:
        return KBFisheyeCamera(
            camera_id=camera_id,
            name=camera_id,
            image_width=1600,
            image_height=1300,
            fx=505.0,
            fy=503.0,
            cx=800.0,
            cy=650.0,
            kb_coefficients=(0.04, -0.006, 0.0007, -0.00005),
            R_camera_reference=np.eye(3),
            t_camera_reference_mm=np.asarray(translation, dtype=np.float64),
        )

    cam0 = camera("cam_0", [0, 0, 0])
    cam1 = camera("cam_1", [-120, 0, 0])
    return EGOStereoCalibration(
        source_path="synthetic",
        serial_number="TEST",
        cam0=cam0,
        cam1=cam1,
        R_cam1_cam0=np.eye(3),
        t_cam1_cam0_mm=np.array([-120.0, 0.0, 0.0]),
    )


def test_recovers_noisy_stereo_pose_with_one_outlier() -> None:
    calibration = _calibration()
    points = np.array(
        [
            [-55, -35, -20], [50, -30, -18], [48, 42, -12], [-52, 40, -10],
            [-42, -28, 35], [45, -25, 40], [38, 36, 48], [-46, 32, 42],
        ],
        dtype=np.float64,
    )
    true_rotation = Rotation.from_rotvec([0.22, -0.16, 0.09]).as_matrix()
    true_translation = np.array([35.0, -18.0, 720.0])
    cam0_points = points @ true_rotation.T + true_translation
    cam1_points = calibration.transform_cam0_to_cam1(cam0_points)
    rng = np.random.default_rng(7)
    observations = []
    for camera_id, camera_points, camera in (
        (0, cam0_points, calibration.cam0), (1, cam1_points, calibration.cam1)
    ):
        image_points = camera.project(camera_points) + rng.normal(0, 0.18, (len(points), 2))
        for index, (point, pixel) in enumerate(zip(points, image_points)):
            observations.append(PoseObservation(point, pixel, camera_id, label=f"P{index}"))
    observations[-1] = PoseObservation(
        observations[-1].point_marker_mm,
        observations[-1].pixel_xy + [38.0, -31.0],
        1,
        label="outlier",
    )

    result = solve_stereo_pose(calibration, observations, f_scale_px=2.0)

    rotation_error = Rotation.from_matrix(result.R_cam0_marker @ true_rotation.T).magnitude()
    assert result.success
    assert rotation_error < 0.01
    assert np.linalg.norm(result.t_cam0_marker_mm - true_translation) < 2.0
    assert result.median_error_px < 0.7
    assert result.inlier_mask.sum() == len(observations) - 1


def test_accepts_explicit_initial_pose_with_points_split_between_cameras() -> None:
    calibration = _calibration()
    points = np.array([[-40, -30, 0], [45, -20, 8], [30, 42, 15], [-35, 35, 25]])
    rotation = Rotation.from_rotvec([0.08, 0.13, -0.05]).as_matrix()
    translation = np.array([20.0, 15.0, 650.0])
    cam0 = points @ rotation.T + translation
    cam1 = calibration.transform_cam0_to_cam1(cam0)
    observations = [
        PoseObservation(points[i], calibration.cam0.project(cam0[i]), 0)
        for i in range(2)
    ] + [
        PoseObservation(points[i], calibration.cam1.project(cam1[i]), 1)
        for i in range(2, 4)
    ]

    result = solve_stereo_pose(
        calibration,
        observations,
        initial_R_cam0_marker=Rotation.from_rotvec([0.07, 0.12, -0.04]).as_matrix(),
        initial_t_cam0_marker_mm=[22.0, 13.0, 655.0],
    )

    assert result.rmse_px < 1.0e-7
    assert result.t_cam0_marker_mm == pytest.approx(translation, abs=1.0e-6)


def test_rejects_invalid_observations() -> None:
    with pytest.raises(ValueError, match="camera_id"):
        PoseObservation(np.zeros(3), np.zeros(2), 2)  # type: ignore[arg-type]
