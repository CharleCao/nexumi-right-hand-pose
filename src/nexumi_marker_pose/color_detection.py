"""Detect the right-hand marker's coloured triangular regions.

The printed sRGB values are useful labels, but raw RGB distance is brittle in
the EGO recordings: exposure changes mainly shorten colour saturation while
the direction of the Lab ``a/b`` chroma vector remains comparatively stable.
This detector therefore classifies chromatic stickers primarily by Lab hue,
then uses chroma, morphology and triangle geometry to reject background tape.

This module detects colour/shape candidates only.  Repeated colours (red,
yellow and blue) are intentionally not assigned to F1/F7 etc.; that requires
the 3-D face topology and pose solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import cv2
import numpy as np

from .sticker_spec import COLOR_RGB


ROI = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ColorDetectionConfig:
    """Thresholds expressed in pixels and OpenCV's 8-bit Lab coordinates."""

    min_chroma: float = 10.0
    max_hue_error_deg: float = 31.0
    hue_sigma_deg: float = 18.0
    gray_max_chroma: float = 18.0
    gray_l_range: tuple[float, float] = (55.0, 235.0)
    morphology_kernel: int = 3
    morphology_iterations: int = 1
    min_area_px: float = 24.0
    max_area_fraction: float = 0.08
    polygon_epsilon_fraction: float = 0.035
    min_solidity: float = 0.72
    min_triangle_fill: float = 0.62
    neutral_adaptation_limit: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            self.min_chroma,
            self.max_hue_error_deg,
            self.hue_sigma_deg,
            self.gray_max_chroma,
            self.min_area_px,
        )
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("positive detector thresholds must be finite")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        if self.morphology_iterations < 0:
            raise ValueError("morphology_iterations must be non-negative")
        if not 0 < self.max_area_fraction <= 1:
            raise ValueError("max_area_fraction must be in (0, 1]")
        if not 0 < self.polygon_epsilon_fraction < 0.25:
            raise ValueError("polygon_epsilon_fraction must be in (0, 0.25)")
        if not 0 <= self.min_solidity <= 1 or not 0 <= self.min_triangle_fill <= 1:
            raise ValueError("shape thresholds must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ColorTriangleCandidate:
    """One connected, triangle-like region in full-image coordinates."""

    color_name: str
    reference_rgb: tuple[int, int, int]
    contour_xy: np.ndarray
    triangle_xy: np.ndarray
    component_mask: np.ndarray
    mask_origin_xy: tuple[int, int]
    centroid_xy: tuple[float, float]
    area_px: float
    color_confidence: float
    shape_confidence: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ColorDetectionResult:
    """Cleaned full-resolution masks and ranked triangle candidates."""

    masks: Mapping[str, np.ndarray]
    candidates: tuple[ColorTriangleCandidate, ...]
    neutral_ab: tuple[float, float]
    roi: ROI


_COLOR_NAMES = tuple(COLOR_RGB)
_GRAY_NAME = "浅灰"
_CHROMATIC_NAMES = tuple(name for name in _COLOR_NAMES if name != _GRAY_NAME)


def _reference_lab() -> dict[str, np.ndarray]:
    rgb = np.asarray([[COLOR_RGB[name] for name in _COLOR_NAMES]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    return {name: value for name, value in zip(_COLOR_NAMES, lab)}


_REFERENCE_LAB = _reference_lab()


def _normalise_roi(shape: tuple[int, ...], roi: ROI | None) -> ROI:
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width, height
    if len(roi) != 4 or any(isinstance(value, bool) for value in roi):
        raise ValueError("roi must be (x, y, width, height)")
    x, y, roi_width, roi_height = (int(value) for value in roi)
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("roi must have non-negative origin and positive size")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError("roi extends outside the image")
    return x, y, roi_width, roi_height


def _estimate_neutral_ab(lab: np.ndarray, limit: float) -> tuple[float, float]:
    """Estimate a small per-frame AWB offset from low-chroma, usable pixels."""

    ab = lab[:, :, 1:3]
    provisional_chroma = np.linalg.norm(ab - 128.0, axis=2)
    usable = (provisional_chroma < 13.0) & (lab[:, :, 0] > 45) & (lab[:, :, 0] < 240)
    if np.count_nonzero(usable) < 64:
        return 128.0, 128.0
    centre = np.median(ab[usable], axis=0)
    centre = np.clip(centre, 128.0 - limit, 128.0 + limit)
    return float(centre[0]), float(centre[1])


def _angular_difference(first: np.ndarray, second: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(first - second), np.cos(first - second)))


def _colour_maps(
    image_bgr: np.ndarray, config: ColorDetectionConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[float, float]]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    neutral_ab = _estimate_neutral_ab(lab, config.neutral_adaptation_limit)
    da = lab[:, :, 1] - neutral_ab[0]
    db = lab[:, :, 2] - neutral_ab[1]
    chroma = np.hypot(da, db)
    hue = np.arctan2(db, da)

    errors: list[np.ndarray] = []
    proto_chroma: list[float] = []
    for name in _CHROMATIC_NAMES:
        proto = _REFERENCE_LAB[name]
        proto_da, proto_db = float(proto[1] - 128.0), float(proto[2] - 128.0)
        errors.append(_angular_difference(hue, float(np.arctan2(proto_db, proto_da))))
        proto_chroma.append(float(np.hypot(proto_da, proto_db)))
    error_stack = np.stack(errors, axis=2)
    winner = np.argmin(error_stack, axis=2)

    masks: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    sigma = np.deg2rad(config.hue_sigma_deg)
    max_error = np.deg2rad(config.max_hue_error_deg)
    for index, name in enumerate(_CHROMATIC_NAMES):
        angular_score = np.exp(-0.5 * np.square(error_stack[:, :, index] / sigma))
        saturation_score = np.clip(chroma / (0.45 * proto_chroma[index]), 0.0, 1.0)
        score = angular_score * np.sqrt(saturation_score)
        selected = (
            (winner == index)
            & (error_stack[:, :, index] <= max_error)
            & (chroma >= config.min_chroma)
        )
        masks[name] = np.where(selected, 255, 0).astype(np.uint8)
        confidence[name] = score.astype(np.float32)

    gray_l_min, gray_l_max = config.gray_l_range
    gray_selected = (
        (chroma <= config.gray_max_chroma)
        & (lab[:, :, 0] >= gray_l_min)
        & (lab[:, :, 0] <= gray_l_max)
    )
    gray_chroma_score = np.exp(-0.5 * np.square(chroma / (0.65 * config.gray_max_chroma)))
    gray_l_score = np.exp(-0.5 * np.square((lab[:, :, 0] - _REFERENCE_LAB[_GRAY_NAME][0]) / 75.0))
    masks[_GRAY_NAME] = np.where(gray_selected, 255, 0).astype(np.uint8)
    confidence[_GRAY_NAME] = (gray_chroma_score * gray_l_score).astype(np.float32)
    return masks, confidence, neutral_ab


def _clean_mask(mask: np.ndarray, config: ColorDetectionConfig) -> np.ndarray:
    if config.morphology_iterations == 0 or config.morphology_kernel == 1:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel, config.morphology_kernel),
    )
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=config.morphology_iterations
    )
    return cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=config.morphology_iterations
    )


def _triangle_for_contour(
    contour: np.ndarray, config: ColorDetectionConfig
) -> tuple[np.ndarray, float] | None:
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    contour_area = float(cv2.contourArea(contour))
    if hull_area <= 0:
        return None
    solidity = contour_area / hull_area
    if solidity < config.min_solidity:
        return None

    perimeter = float(cv2.arcLength(hull, True))
    polygon = cv2.approxPolyDP(hull, config.polygon_epsilon_fraction * perimeter, True)
    if len(polygon) == 3:
        triangle = polygon.reshape(3, 2).astype(np.float32)
        triangle_area = abs(float(cv2.contourArea(triangle)))
    else:
        triangle_area, triangle = cv2.minEnclosingTriangle(hull)
        if triangle is None:
            return None
        triangle = triangle.reshape(3, 2).astype(np.float32)
        triangle_area = float(triangle_area)
    if triangle_area <= 0:
        return None
    triangle_fill = contour_area / triangle_area
    if triangle_fill < config.min_triangle_fill:
        return None
    fill_score = np.clip(
        (triangle_fill - config.min_triangle_fill) / (1.0 - config.min_triangle_fill),
        0.0,
        1.0,
    )
    shape_confidence = float(np.sqrt(solidity * (0.35 + 0.65 * fill_score)))
    return triangle, shape_confidence


def detect_colored_triangles(
    image_bgr: np.ndarray,
    *,
    roi: ROI | None = None,
    config: ColorDetectionConfig | None = None,
) -> ColorDetectionResult:
    """Detect colour-labelled triangular regions in a BGR uint8 image.

    ``roi`` is optional and only limits work; all returned contours, triangle
    corners and centroids remain in full-image pixel coordinates.  With no ROI
    the complete frame is processed, so this detector contains no scene-specific
    screen location.
    """

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3) and dtype uint8")
    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        raise ValueError("image_bgr must not be empty")
    detector_config = config or ColorDetectionConfig()
    x0, y0, width, height = _normalise_roi(image_bgr.shape, roi)
    crop = np.ascontiguousarray(image_bgr[y0 : y0 + height, x0 : x0 + width])
    raw_masks, confidence_maps, neutral_ab = _colour_maps(crop, detector_config)

    full_masks: dict[str, np.ndarray] = {}
    candidates: list[ColorTriangleCandidate] = []
    # Keep this threshold independent of an optional tight ROI.  Otherwise an
    # identical marker could be accepted globally but rejected merely because
    # a caller cropped closely around it.
    max_area = detector_config.max_area_fraction * image_bgr.shape[0] * image_bgr.shape[1]
    for color_name in _COLOR_NAMES:
        clean = _clean_mask(raw_masks[color_name], detector_config)
        full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        full_mask[y0 : y0 + height, x0 : x0 + width] = clean
        full_masks[color_name] = full_mask
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < detector_config.min_area_px or area > max_area:
                continue
            triangle_result = _triangle_for_contour(contour, detector_config)
            if triangle_result is None:
                continue
            triangle, shape_confidence = triangle_result
            bx, by, bw, bh = cv2.boundingRect(contour)
            local_contour = contour - np.asarray([[[bx, by]]], dtype=contour.dtype)
            component_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(component_mask, [local_contour], -1, 255, cv2.FILLED)
            score_patch = confidence_maps[color_name][by : by + bh, bx : bx + bw]
            color_confidence = float(np.mean(score_patch[component_mask != 0]))
            moments = cv2.moments(contour)
            centroid = (
                float(moments["m10"] / moments["m00"] + x0),
                float(moments["m01"] / moments["m00"] + y0),
            )
            contour_xy = contour.reshape(-1, 2).astype(np.float32)
            contour_xy += np.asarray([x0, y0], dtype=np.float32)
            triangle += np.asarray([x0, y0], dtype=np.float32)
            confidence_value = float(np.sqrt(color_confidence * shape_confidence))
            candidates.append(
                ColorTriangleCandidate(
                    color_name=color_name,
                    reference_rgb=COLOR_RGB[color_name],
                    contour_xy=contour_xy,
                    triangle_xy=triangle,
                    component_mask=component_mask,
                    mask_origin_xy=(bx + x0, by + y0),
                    centroid_xy=centroid,
                    area_px=area,
                    color_confidence=color_confidence,
                    shape_confidence=shape_confidence,
                    confidence=confidence_value,
                )
            )
    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    return ColorDetectionResult(
        masks=full_masks,
        candidates=tuple(candidates),
        neutral_ab=neutral_ab,
        roi=(x0, y0, width, height),
    )


__all__ = [
    "ColorDetectionConfig",
    "ColorDetectionResult",
    "ColorTriangleCandidate",
    "detect_colored_triangles",
]
