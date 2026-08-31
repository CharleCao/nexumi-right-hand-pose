"""Robust 6DoF pose refinement from calibrated fisheye observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .ego_calibration import EGOStereoCalibration, KBFisheyeCamera


FloatArray = NDArray[np.float64]
CameraId = Literal[0, 1]


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """One known marker point observed in one EGO camera."""

    point_marker_mm: FloatArray
    pixel_xy: FloatArray
    camera_id: CameraId
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        point = np.asarray(self.point_marker_mm, dtype=np.float64).copy()
        pixel = np.asarray(self.pixel_xy, dtype=np.float64).copy()
        if point.shape != (3,) or pixel.shape != (2,):
            raise ValueError("point_marker_mm and pixel_xy must have shapes (3,) and (2,)")
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(pixel)):
            raise ValueError("observation contains a non-finite value")
        if self.camera_id not in (0, 1):
            raise ValueError("camera_id must be 0 or 1")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")
        point.setflags(write=False)
        pixel.setflags(write=False)
        object.__setattr__(self, "point_marker_mm", point)
        object.__setattr__(self, "pixel_xy", pixel)


@dataclass(frozen=True, slots=True)
class StereoPoseResult:
    """Marker pose with convention ``p_cam0 = R @ p_marker + t_mm``."""

    R_cam0_marker: FloatArray
    t_cam0_marker_mm: FloatArray
    rotation_vector: FloatArray
    reprojection_errors_px: FloatArray
    inlier_mask: NDArray[np.bool_]
    rmse_px: float
    median_error_px: float
    cam0_rmse_px: float | None
    cam1_rmse_px: float | None
    success: bool
    message: str
    nfev: int


def _camera(calibration: EGOStereoCalibration, camera_id: int) -> KBFisheyeCamera:
    return calibration.cam0 if camera_id == 0 else calibration.cam1


def _points_in_camera(
    calibration: EGOStereoCalibration,
    points_marker_mm: FloatArray,
    camera_ids: NDArray[np.int64],
    rotation_vector: FloatArray,
    translation_mm: FloatArray,
) -> FloatArray:
    points_cam0 = points_marker_mm @ Rotation.from_rotvec(rotation_vector).as_matrix().T
    points_cam0 += translation_mm
    result = points_cam0.copy()
    select_cam1 = camera_ids == 1
    if np.any(select_cam1):
        result[select_cam1] = calibration.transform_cam0_to_cam1(
            points_cam0[select_cam1]
        )
    return result


def _initial_pose(
    calibration: EGOStereoCalibration,
    observations: Sequence[PoseObservation],
) -> tuple[FloatArray, FloatArray]:
    """Initialize from one camera using undistorted unit rays and EPNP."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required for automatic pose initialization") from error

    by_camera = {
        camera_id: [item for item in observations if item.camera_id == camera_id]
        for camera_id in (0, 1)
    }
    camera_id = max(by_camera, key=lambda value: len(by_camera[value]))
    selected = by_camera[camera_id]
    if len(selected) < 4:
        # Three coloured face centres seen by both cameras are sufficient for
        # a metric 3-D/3-D initialization.  This matters when only three faces
        # of the convex marker are visible in one view.
        paired: list[tuple[np.ndarray, np.ndarray]] = []
        for first_index, first in enumerate(observations):
            if first.camera_id != 0:
                continue
            for second in observations[first_index + 1 :]:
                if second.camera_id != 1 or not np.allclose(
                    first.point_marker_mm, second.point_marker_mm, atol=1e-8
                ):
                    continue
                ray0 = calibration.cam0.unproject(first.pixel_xy)
                ray1 = calibration.R_cam1_cam0.T @ calibration.cam1.unproject(
                    second.pixel_xy
                )
                origin1 = -calibration.R_cam1_cam0.T @ calibration.t_cam1_cam0_mm
                system = np.array(
                    [[ray0 @ ray0, -(ray0 @ ray1)],
                     [ray0 @ ray1, -(ray1 @ ray1)]], dtype=np.float64
                )
                depth0, depth1 = np.linalg.solve(
                    system, np.array([ray0 @ origin1, ray1 @ origin1])
                )
                triangulated = 0.5 * (
                    depth0 * ray0 + origin1 + depth1 * ray1
                )
                paired.append((first.point_marker_mm, triangulated))
                break
        if len(paired) < 3:
            raise ValueError(
                "automatic initialization needs four points in one camera "
                "or three stereo point pairs"
            )
        source = np.asarray([item[0] for item in paired], dtype=np.float64)
        target = np.asarray([item[1] for item in paired], dtype=np.float64)
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        U, _, Vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0:
            Vt[-1] *= -1
            rotation = Vt.T @ U.T
        translation = target_center - rotation @ source_center
        return Rotation.from_matrix(rotation).as_rotvec(), translation
    object_points = np.asarray([item.point_marker_mm for item in selected], np.float64)
    rays = _camera(calibration, camera_id).unproject(
        [item.pixel_xy for item in selected]
    )
    if np.any(rays[:, 2] <= 0.05):
        raise ValueError("automatic initialization requires forward-facing observations")
    normalized_pixels = rays[:, :2] / rays[:, 2, None]
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        normalized_pixels,
        np.eye(3, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        raise RuntimeError("EPNP pose initialization failed")
    rotation = Rotation.from_rotvec(rvec[:, 0]).as_matrix()
    translation = tvec[:, 0]
    if camera_id == 1:
        transform = calibration.T_cam0_cam1_mm
        rotation = transform[:3, :3] @ rotation
        translation = transform[:3, :3] @ translation + transform[:3, 3]
    return Rotation.from_matrix(rotation).as_rotvec(), translation


def solve_stereo_pose(
    calibration: EGOStereoCalibration,
    observations: Sequence[PoseObservation],
    *,
    initial_R_cam0_marker: ArrayLike | None = None,
    initial_t_cam0_marker_mm: ArrayLike | None = None,
    loss: str = "huber",
    f_scale_px: float = 3.0,
    inlier_threshold_px: float = 4.0,
    max_nfev: int = 500,
) -> StereoPoseResult:
    """Estimate marker pose by robustly matching observed and predicted rays.

    The optimized state is marker-to-cam0: ``p_cam0 = R @ p_marker + t_mm``.
    Pixel errors are reported with each camera's native KB projection model.
    """

    items = tuple(observations)
    if len(items) < 4:
        raise ValueError("at least four observations are required")
    if f_scale_px <= 0 or inlier_threshold_px <= 0 or max_nfev <= 0:
        raise ValueError("solver scales and max_nfev must be positive")
    if (initial_R_cam0_marker is None) != (initial_t_cam0_marker_mm is None):
        raise ValueError("initial rotation and translation must be provided together")

    points = np.asarray([item.point_marker_mm for item in items], dtype=np.float64)
    pixels = np.asarray([item.pixel_xy for item in items], dtype=np.float64)
    camera_ids = np.asarray([item.camera_id for item in items], dtype=np.int64)
    weights = np.sqrt(np.asarray([item.weight for item in items], dtype=np.float64))
    observed_rays = np.empty((len(items), 3), dtype=np.float64)
    focal_scales = np.empty(len(items), dtype=np.float64)
    for camera_id in (0, 1):
        selected = camera_ids == camera_id
        if np.any(selected):
            camera = _camera(calibration, camera_id)
            observed_rays[selected] = camera.unproject(pixels[selected])
            focal_scales[selected] = 0.5 * (camera.fx + camera.fy)

    if initial_R_cam0_marker is None:
        initial_rvec, initial_translation = _initial_pose(calibration, items)
    else:
        initial_rotation = np.asarray(initial_R_cam0_marker, dtype=np.float64)
        initial_translation = np.asarray(initial_t_cam0_marker_mm, dtype=np.float64)
        if initial_rotation.shape != (3, 3) or initial_translation.shape != (3,):
            raise ValueError("initial rotation and translation must have shapes (3,3), (3,)")
        initial_rvec = Rotation.from_matrix(initial_rotation).as_rotvec()
    x0 = np.concatenate((initial_rvec, initial_translation))

    def residual(state: FloatArray, selected: NDArray[np.bool_] | None = None) -> FloatArray:
        predicted = _points_in_camera(
            calibration, points, camera_ids, state[:3], state[3:]
        )
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        ray_error = predicted - observed_rays
        values = ray_error * focal_scales[:, None] * weights[:, None]
        return values.ravel() if selected is None else values[selected].ravel()

    def reprojection_errors(state: FloatArray) -> FloatArray:
        camera_points = _points_in_camera(
            calibration, points, camera_ids, state[:3], state[3:]
        )
        projected = np.empty_like(pixels)
        for camera_id in (0, 1):
            selected = camera_ids == camera_id
            if np.any(selected):
                projected[selected] = _camera(calibration, camera_id).project(
                    camera_points[selected]
                )
        return np.linalg.norm(projected - pixels, axis=1)

    optimized = least_squares(
        residual,
        x0,
        loss=loss,
        f_scale=f_scale_px,
        max_nfev=max_nfev,
    )
    preliminary_errors = reprojection_errors(optimized.x)
    preliminary_inliers = preliminary_errors <= inlier_threshold_px
    total_nfev = int(optimized.nfev)
    if 4 <= int(preliminary_inliers.sum()) < len(items):
        refined = least_squares(
            lambda state: residual(state, preliminary_inliers),
            optimized.x,
            loss="huber",
            f_scale=f_scale_px,
            max_nfev=max_nfev,
        )
        total_nfev += int(refined.nfev)
        optimized = refined
    rotation = Rotation.from_rotvec(optimized.x[:3]).as_matrix()
    translation = optimized.x[3:].copy()
    errors = reprojection_errors(optimized.x)

    def camera_rmse(camera_id: int) -> float | None:
        selected = camera_ids == camera_id
        return float(np.sqrt(np.mean(errors[selected] ** 2))) if np.any(selected) else None

    return StereoPoseResult(
        R_cam0_marker=rotation,
        t_cam0_marker_mm=translation,
        rotation_vector=optimized.x[:3].copy(),
        reprojection_errors_px=errors,
        inlier_mask=errors <= inlier_threshold_px,
        rmse_px=float(np.sqrt(np.mean(errors**2))),
        median_error_px=float(np.median(errors)),
        cam0_rmse_px=camera_rmse(0),
        cam1_rmse_px=camera_rmse(1),
        success=bool(optimized.success),
        message=str(optimized.message),
        nfev=total_nfev,
    )


__all__ = ["PoseObservation", "StereoPoseResult", "solve_stereo_pose"]
