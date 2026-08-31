"""Initialize a marker pose by enumerating coloured-face hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
import json

import numpy as np

from .color_detection import ColorDetectionResult, ColorTriangleCandidate
from .ego_calibration import EGOStereoCalibration
from .stereo_pose import PoseObservation, StereoPoseResult, solve_stereo_pose


@dataclass(frozen=True, slots=True)
class MarkerPoseConfig:
    min_area_px: float = 400.0
    min_confidence: float = 0.5
    max_candidates_per_color: int = 2
    hypothesis_face_count: int = 4
    min_stereo_pairs: int = 2
    max_stereo_ray_distance_mm: float = 5.0
    min_depth_mm: float = 100.0
    max_depth_mm: float = 2000.0
    max_corner_rmse_px: float = 15.0
    corner_score_weight: float = 0.2


@dataclass(frozen=True, slots=True)
class FaceAssignment:
    sticker_id: str
    color_name: str
    primary_candidate_index: int
    secondary_candidate_index: int | None


@dataclass(frozen=True, slots=True)
class MarkerPoseEstimate:
    pose: StereoPoseResult
    assignments: tuple[FaceAssignment, ...]
    score: float
    corner_rmse_px: float
    stereo_pair_count: int
    hypotheses_evaluated: int


def _eligible(
    result: ColorDetectionResult, config: MarkerPoseConfig
) -> list[tuple[int, ColorTriangleCandidate]]:
    grouped: dict[str, list[tuple[int, ColorTriangleCandidate]]] = {}
    for index, candidate in enumerate(result.candidates):
        if candidate.area_px < config.min_area_px or candidate.confidence < config.min_confidence:
            continue
        grouped.setdefault(candidate.color_name, []).append((index, candidate))
    selected: list[tuple[int, ColorTriangleCandidate]] = []
    for values in grouped.values():
        values.sort(key=lambda item: item[1].confidence, reverse=True)
        selected.extend(values[: config.max_candidates_per_color])
    return selected


def _triangulation_error(
    calibration: EGOStereoCalibration,
    left: ColorTriangleCandidate,
    right: ColorTriangleCandidate,
) -> tuple[float, float, float]:
    ray0 = calibration.cam0.unproject(left.centroid_xy)
    ray1 = calibration.R_cam1_cam0.T @ calibration.cam1.unproject(right.centroid_xy)
    origin1 = -calibration.R_cam1_cam0.T @ calibration.t_cam1_cam0_mm
    system = np.array(
        [[ray0 @ ray0, -(ray0 @ ray1)], [ray0 @ ray1, -(ray1 @ ray1)]],
        dtype=np.float64,
    )
    depth0, depth1 = np.linalg.solve(
        system, np.array([ray0 @ origin1, ray1 @ origin1])
    )
    point0 = depth0 * ray0
    point1 = origin1 + depth1 * ray1
    return float(np.linalg.norm(point0 - point1)), float(depth0), float(depth1)


def _corner_rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return min(
        float(np.sqrt(np.mean(np.sum((predicted - observed[list(order)]) ** 2, axis=1))))
        for order in permutations(range(3))
    )


def estimate_marker_pose(
    calibration: EGOStereoCalibration,
    marker_model: str | Path | dict,
    left: ColorDetectionResult,
    right: ColorDetectionResult,
    *,
    config: MarkerPoseConfig | None = None,
) -> MarkerPoseEstimate:
    """Find a cam0 marker pose without hard-coded face IDs or image positions."""

    cfg = config or MarkerPoseConfig()
    if isinstance(marker_model, (str, Path)):
        with Path(marker_model).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    else:
        document = marker_model
    faces = {item["sticker_id"]: item for item in document["faces"]}
    by_color: dict[str, list[str]] = {}
    for sticker_id, face in faces.items():
        by_color.setdefault(face["color"]["name_zh"], []).append(sticker_id)

    primary = _eligible(right, cfg)
    secondary = _eligible(left, cfg)
    if len(primary) < cfg.hypothesis_face_count:
        raise ValueError("not enough eligible right-camera colour candidates")

    stereo_matches: dict[int, tuple[int, ColorTriangleCandidate, float]] = {}
    for primary_index, primary_candidate in primary:
        possible = []
        for secondary_index, secondary_candidate in secondary:
            if secondary_candidate.color_name != primary_candidate.color_name:
                continue
            distance, depth0, depth1 = _triangulation_error(
                calibration, secondary_candidate, primary_candidate
            )
            if (
                distance <= cfg.max_stereo_ray_distance_mm
                and cfg.min_depth_mm <= depth0 <= cfg.max_depth_mm
                and cfg.min_depth_mm <= depth1 <= cfg.max_depth_mm
            ):
                possible.append((distance, secondary_index, secondary_candidate))
        if possible:
            distance, secondary_index, secondary_candidate = min(possible)
            stereo_matches[primary_index] = (
                secondary_index,
                secondary_candidate,
                distance,
            )

    estimates: list[tuple[float, float, StereoPoseResult, tuple[FaceAssignment, ...]]] = []
    evaluated = 0
    for subset in combinations(primary, cfg.hypothesis_face_count):
        if len({candidate.color_name for _, candidate in subset}) != len(subset):
            continue
        if sum(index in stereo_matches for index, _ in subset) < cfg.min_stereo_pairs:
            continue
        face_options = [by_color.get(candidate.color_name, []) for _, candidate in subset]
        if any(not values for values in face_options):
            continue
        for sticker_ids in product(*face_options):
            if len(set(sticker_ids)) != len(sticker_ids):
                continue
            observations: list[PoseObservation] = []
            assignments: list[FaceAssignment] = []
            corner_items: list[tuple[int, str, ColorTriangleCandidate]] = []
            for (primary_index, candidate), sticker_id in zip(subset, sticker_ids):
                face = faces[sticker_id]["step_face"]
                point = np.asarray(face["centroid_mm"], dtype=np.float64)
                observations.append(PoseObservation(point, candidate.centroid_xy, 1))
                corner_items.append((1, sticker_id, candidate))
                secondary_index = None
                if primary_index in stereo_matches:
                    secondary_index, secondary_candidate, _ = stereo_matches[primary_index]
                    observations.append(
                        PoseObservation(point, secondary_candidate.centroid_xy, 0)
                    )
                    corner_items.append((0, sticker_id, secondary_candidate))
                assignments.append(
                    FaceAssignment(
                        sticker_id=sticker_id,
                        color_name=candidate.color_name,
                        primary_candidate_index=primary_index,
                        secondary_candidate_index=secondary_index,
                    )
                )
            evaluated += 1
            try:
                pose = solve_stereo_pose(
                    calibration,
                    observations,
                    loss="linear",
                    inlier_threshold_px=100.0,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            corner_errors = []
            for camera_id, sticker_id, candidate in corner_items:
                vertices = np.asarray(
                    faces[sticker_id]["step_face"]["vertices_mm"], dtype=np.float64
                )
                points_cam0 = vertices @ pose.R_cam0_marker.T + pose.t_cam0_marker_mm
                if camera_id == 1:
                    predicted = calibration.cam1.project(
                        calibration.transform_cam0_to_cam1(points_cam0)
                    )
                else:
                    predicted = calibration.cam0.project(points_cam0)
                corner_errors.append(_corner_rmse(predicted, candidate.triangle_xy))
            mean_corner_error = float(np.mean(corner_errors))
            if mean_corner_error > cfg.max_corner_rmse_px:
                continue
            score = pose.rmse_px + cfg.corner_score_weight * mean_corner_error
            estimates.append((score, mean_corner_error, pose, tuple(assignments)))

    if not estimates:
        raise RuntimeError(f"no valid marker pose among {evaluated} hypotheses")
    score, corner_error, pose, assignments = min(estimates, key=lambda item: item[0])
    return MarkerPoseEstimate(
        pose=pose,
        assignments=assignments,
        score=score,
        corner_rmse_px=corner_error,
        stereo_pair_count=sum(item.secondary_candidate_index is not None for item in assignments),
        hypotheses_evaluated=evaluated,
    )


__all__ = [
    "FaceAssignment",
    "MarkerPoseConfig",
    "MarkerPoseEstimate",
    "estimate_marker_pose",
]
