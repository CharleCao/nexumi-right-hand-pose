"""Load the EGO stereo Kannala--Brandt camera calibration.

The vendor YAML stores each camera extrinsic relative to ``reference_camera``.
For the files used by this project the reference is ``cam_0``; consequently the
``cam_1`` extrinsic has the unambiguous active-transform convention::

    p_cam1_mm = R_cam1_cam0 @ p_cam0_mm + t_cam1_cam0_mm

Translations in the EGO calibration are millimetres.  This module keeps that
unit explicit in every public name and only converts to metres through dedicated
properties.

The first four YAML distortion values are the OpenCV fisheye/Kannala--Brandt
coefficients ``(k1, k2, k3, k4)``.  The remaining rational/tangential fields in
the vendor schema must be zero because they are not part of that model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class KBFisheyeCamera:
    """One calibrated OpenCV-compatible Kannala--Brandt fisheye camera."""

    camera_id: str
    name: str
    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    kb_coefficients: tuple[float, float, float, float]
    R_camera_reference: FloatArray
    t_camera_reference_mm: FloatArray

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        intrinsic_values = np.asarray(
            (self.fx, self.fy, self.cx, self.cy, *self.kb_coefficients),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(intrinsic_values)):
            raise ValueError("intrinsics and KB coefficients must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("focal lengths must be positive")

        rotation = _as_float_array(
            self.R_camera_reference, (3, 3), "R_camera_reference"
        ).copy()
        translation = _as_float_array(
            self.t_camera_reference_mm, (3,), "t_camera_reference_mm"
        ).copy()
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-5):
            raise ValueError("R_camera_reference is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-5):
            raise ValueError("R_camera_reference must be a proper rotation")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "R_camera_reference", rotation)
        object.__setattr__(self, "t_camera_reference_mm", translation)

    @property
    def K(self) -> FloatArray:
        """Return the 3x3 OpenCV intrinsic matrix."""

        return np.array(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    @property
    def D(self) -> FloatArray:
        """Return OpenCV ``cv2.fisheye`` distortion as shape ``(4, 1)``."""

        return np.asarray(self.kb_coefficients, dtype=np.float64).reshape(4, 1)

    @property
    def t_camera_reference_m(self) -> FloatArray:
        """Return the reference-to-camera translation converted to metres."""

        return self.t_camera_reference_mm / 1000.0

    def opencv_fisheye_parameters(self) -> tuple[FloatArray, FloatArray, tuple[int, int]]:
        """Return ``(K, D, image_size)`` for the OpenCV fisheye API."""

        return self.K, self.D, (self.image_width, self.image_height)

    def _theta_limit(self) -> float:
        """Largest angle before the KB radial polynomial stops being monotonic."""

        k1, k2, k3, k4 = self.kb_coefficients
        derivative_coefficients = np.array(
            (1.0, 0.0, 3.0 * k1, 0.0, 5.0 * k2, 0.0, 7.0 * k3, 0.0, 9.0 * k4),
            dtype=np.float64,
        )
        roots = np.polynomial.polynomial.polyroots(derivative_coefficients)
        positive_real_roots = [
            float(root.real)
            for root in roots
            if abs(root.imag) < 1.0e-9 and root.real > 1.0e-9
        ]
        return min(positive_real_roots, default=float(np.pi))

    def _distort_theta(self, theta: FloatArray) -> FloatArray:
        k1, k2, k3, k4 = self.kb_coefficients
        theta2 = theta * theta
        return theta * (
            1.0
            + theta2
            * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
        )

    def project(self, points_camera: ArrayLike) -> FloatArray:
        """Project camera-frame 3-D points to pixels using the KB model.

        ``points_camera`` may have shape ``(3,)`` or ``(..., 3)`` and may use
        any length unit.  The returned array has the matching leading shape and
        final dimension 2.  Points outside the one-to-one angular portion of the
        calibrated radial model are rejected.
        """

        points = np.asarray(points_camera, dtype=np.float64)
        if points.shape == () or points.shape[-1] != 3:
            raise ValueError("points_camera must have shape (3,) or (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_camera contains a non-finite value")
        norm = np.linalg.norm(points, axis=-1)
        if np.any(norm == 0.0):
            raise ValueError("cannot project the camera origin")

        radial = np.hypot(points[..., 0], points[..., 1])
        theta = np.arctan2(radial, points[..., 2])
        if np.any(theta >= self._theta_limit()):
            raise ValueError("point is outside the monotonic KB field of view")
        theta_distorted = self._distort_theta(theta)
        scale = np.divide(
            theta_distorted,
            radial,
            out=np.full_like(theta_distorted, 1.0),
            where=radial > 1.0e-15,
        )
        u = self.fx * points[..., 0] * scale + self.cx
        v = self.fy * points[..., 1] * scale + self.cy
        return np.stack((u, v), axis=-1)

    def unproject(self, pixels: ArrayLike) -> FloatArray:
        """Back-project pixels to unit rays in the camera frame.

        The inverse radial polynomial is solved by bisection on its monotonic
        interval.  This is slower than a fixed polynomial approximation but is
        deterministic and accurate enough for calibration and pose refinement.
        """

        pixel_array = np.asarray(pixels, dtype=np.float64)
        if pixel_array.shape == () or pixel_array.shape[-1] != 2:
            raise ValueError("pixels must have shape (2,) or (..., 2)")
        if not np.all(np.isfinite(pixel_array)):
            raise ValueError("pixels contains a non-finite value")

        x_distorted = (pixel_array[..., 0] - self.cx) / self.fx
        y_distorted = (pixel_array[..., 1] - self.cy) / self.fy
        radius_distorted = np.hypot(x_distorted, y_distorted)
        upper_value = self._theta_limit() * (1.0 - 1.0e-12)
        max_radius = float(self._distort_theta(np.asarray(upper_value)))
        if np.any(radius_distorted > max_radius):
            raise ValueError("pixel is outside the invertible KB field of view")

        low = np.zeros_like(radius_distorted)
        high = np.full_like(radius_distorted, upper_value)
        for _ in range(64):
            middle = (low + high) * 0.5
            below_target = self._distort_theta(middle) < radius_distorted
            low = np.where(below_target, middle, low)
            high = np.where(below_target, high, middle)
        theta = (low + high) * 0.5

        xy_scale = np.divide(
            np.sin(theta),
            radius_distorted,
            out=np.ones_like(theta),
            where=radius_distorted > 1.0e-15,
        )
        rays = np.stack(
            (x_distorted * xy_scale, y_distorted * xy_scale, np.cos(theta)),
            axis=-1,
        )
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


@dataclass(frozen=True, slots=True)
class EGOStereoCalibration:
    """Calibrated EGO pair with explicit ``cam_0 -> cam_1`` semantics."""

    source_path: str
    serial_number: str
    cam0: KBFisheyeCamera
    cam1: KBFisheyeCamera
    R_cam1_cam0: FloatArray
    t_cam1_cam0_mm: FloatArray
    translation_unit: str = "mm"

    def __post_init__(self) -> None:
        if self.translation_unit != "mm":
            raise ValueError("EGO translation unit must be 'mm'")
        if self.cam0.camera_id != "cam_0" or self.cam1.camera_id != "cam_1":
            raise ValueError("calibration must contain cam_0 followed by cam_1")
        rotation = _as_float_array(self.R_cam1_cam0, (3, 3), "R_cam1_cam0").copy()
        translation = _as_float_array(
            self.t_cam1_cam0_mm, (3,), "t_cam1_cam0_mm"
        ).copy()
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-5):
            raise ValueError("R_cam1_cam0 is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-5):
            raise ValueError("R_cam1_cam0 must be a proper rotation")
        if np.linalg.norm(translation) <= 0.0:
            raise ValueError("stereo baseline must be non-zero")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "R_cam1_cam0", rotation)
        object.__setattr__(self, "t_cam1_cam0_mm", translation)

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.t_cam1_cam0_mm))

    @property
    def t_cam1_cam0_m(self) -> FloatArray:
        return self.t_cam1_cam0_mm / 1000.0

    @property
    def T_cam1_cam0_mm(self) -> FloatArray:
        """Homogeneous active transform: ``p_cam1 = T @ p_cam0`` in mm."""

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.R_cam1_cam0
        transform[:3, 3] = self.t_cam1_cam0_mm
        return transform

    @property
    def T_cam0_cam1_mm(self) -> FloatArray:
        """Inverse homogeneous active transform: ``p_cam0 = T @ p_cam1``."""

        rotation_inverse = self.R_cam1_cam0.T
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_inverse
        transform[:3, 3] = -rotation_inverse @ self.t_cam1_cam0_mm
        return transform

    def transform_cam0_to_cam1(self, points_cam0_mm: ArrayLike) -> FloatArray:
        """Apply ``p_cam1 = R_cam1_cam0 p_cam0 + t_cam1_cam0`` in millimetres."""

        points = np.asarray(points_cam0_mm, dtype=np.float64)
        if points.shape == () or points.shape[-1] != 3:
            raise ValueError("points_cam0_mm must have shape (3,) or (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_cam0_mm contains a non-finite value")
        return points @ self.R_cam1_cam0.T + self.t_cam1_cam0_mm


def _parse_camera(raw: Mapping[str, Any], reference_camera: str) -> KBFisheyeCamera:
    camera_id = str(raw.get("id", ""))
    if raw.get("distortion_model") != "KB":
        raise ValueError(f"{camera_id or 'camera'} distortion_model must be 'KB'")
    intrinsics = _require_mapping(raw.get("intrinsics"), f"{camera_id}.intrinsics")
    distortion = _require_mapping(raw.get("distortion"), f"{camera_id}.distortion")
    extrinsics = _require_mapping(raw.get("extrinsics"), f"{camera_id}.extrinsics")

    unsupported = tuple(float(distortion.get(key, 0.0)) for key in ("k5", "k6", "p1", "p2"))
    if not np.allclose(unsupported, 0.0, atol=1.0e-12):
        raise ValueError(
            f"{camera_id} has non-zero k5/k6/p1/p2, which are not part of the KB model"
        )

    camera = KBFisheyeCamera(
        camera_id=camera_id,
        name=str(raw.get("name", camera_id)),
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        kb_coefficients=tuple(
            float(distortion[key]) for key in ("k1", "k2", "k3", "k4")
        ),  # type: ignore[arg-type]
        R_camera_reference=np.asarray(extrinsics["rotation"], dtype=np.float64),
        t_camera_reference_mm=np.asarray(extrinsics["translation"], dtype=np.float64),
    )
    if camera_id == reference_camera:
        if not np.allclose(camera.R_camera_reference, np.eye(3), atol=1.0e-7):
            raise ValueError("reference camera rotation must be identity")
        if not np.allclose(camera.t_camera_reference_mm, 0.0, atol=1.0e-7):
            raise ValueError("reference camera translation must be zero")
    return camera


def load_ego_stereo_calibration(path: str | Path) -> EGOStereoCalibration:
    """Parse and validate a two-camera EGO KB calibration YAML.

    The schema has no inline unit key, so this loader applies the EGO dataset's
    millimetre convention explicitly.  It never silently converts the source
    translation.  Since ``cam_0`` must be the reference camera, the second
    camera's vendor extrinsic directly represents ``cam_0 -> cam_1``.
    """

    calibration_path = Path(path).expanduser()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "PyYAML is required to load EGO calibration; install the 'vision' extras"
        ) from exc

    with calibration_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _require_mapping(document, "calibration document")
    info = _require_mapping(root.get("calibration_info"), "calibration_info")
    if int(info.get("num_cameras", -1)) != 2:
        raise ValueError("EGO stereo calibration must declare exactly two cameras")
    reference_camera = str(info.get("reference_camera", ""))
    if reference_camera != "cam_0":
        raise ValueError("reference_camera must be cam_0 to define cam_0 -> cam_1")
    raw_cameras = root.get("cameras")
    if not isinstance(raw_cameras, list) or len(raw_cameras) != 2:
        raise ValueError("cameras must be a list containing exactly two entries")
    parsed = {
        str(_require_mapping(raw, "camera").get("id", "")): _parse_camera(
            _require_mapping(raw, "camera"), reference_camera
        )
        for raw in raw_cameras
    }
    if set(parsed) != {"cam_0", "cam_1"}:
        raise ValueError("camera ids must be exactly cam_0 and cam_1")
    cam0 = parsed["cam_0"]
    cam1 = parsed["cam_1"]
    return EGOStereoCalibration(
        source_path=str(calibration_path.resolve()),
        serial_number=str(info.get("serial_number", "")),
        cam0=cam0,
        cam1=cam1,
        R_cam1_cam0=cam1.R_camera_reference,
        t_cam1_cam0_mm=cam1.t_camera_reference_mm,
    )
