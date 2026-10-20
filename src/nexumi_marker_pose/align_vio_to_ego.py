"""Align monocular VIO camera poses to EGO stereo marker poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


EGO_LOCAL_TO_UTC_US = 8 * 60 * 60 * 1_000_000


def _load_mono(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    times = np.asarray([int(row["timestamp_us"]) for row in rows], np.int64)
    positions = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows])
    rotations = Rotation.from_quat(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows]
    )
    return rows, times, positions, rotations


def _load_measured_stereo(path: Path, utc_offset_us: int):
    rows = [
        row for row in csv.DictReader(path.open(encoding="utf-8"))
        if row["valid"] == "1" and row["interpolated"] == "0" and row["tx_mm"]
    ]
    times = np.asarray([int(row["timestamp_us"]) - utc_offset_us for row in rows], np.int64)
    positions = np.asarray(
        [[float(row[key]) / 1000.0 for key in ("tx_mm", "ty_mm", "tz_mm")] for row in rows]
    )
    rotations = Rotation.from_quat(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows]
    )
    return times, positions, rotations


def _auto_trim(positions: np.ndarray, threshold_m: float) -> int:
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    candidates = np.flatnonzero(
        (steps > threshold_m) & (np.arange(len(steps)) > 0.75 * len(steps))
    )
    return int(candidates[0] + 1) if len(candidates) else len(positions)


def _sample_stereo(mono_times_s, stereo_times_s, positions, rotations,
                   time_offset_s: float, max_gap_s: float):
    query = mono_times_s + time_offset_s
    right = np.searchsorted(stereo_times_s, query)
    valid = (right > 0) & (right < len(stereo_times_s))
    mono_indices = np.flatnonzero(valid)
    left = right[valid] - 1
    right = right[valid]
    short = stereo_times_s[right] - stereo_times_s[left] <= max_gap_s
    mono_indices, left, right = mono_indices[short], left[short], right[short]
    query = query[mono_indices]
    alpha = (query - stereo_times_s[left]) / (stereo_times_s[right] - stereo_times_s[left])
    sampled_positions = positions[left] * (1 - alpha[:, None]) + positions[right] * alpha[:, None]
    sampled_quaternions = [
        Slerp(stereo_times_s[[lo, hi]], rotations[[lo, hi]])([when]).as_quat()[0]
        for lo, hi, when in zip(left, right, query)
    ]
    return mono_indices, sampled_positions, Rotation.from_quat(sampled_quaternions)


def _initial_guess(mono_positions, mono_rotations, ego_positions, ego_rotations):
    source_vectors, target_vectors = [], []
    separation = 8
    for index in range(len(mono_positions) - separation):
        mono_relative = (mono_rotations[index].inv() * mono_rotations[index + separation]).as_rotvec()
        ego_relative = (ego_rotations[index].inv() * ego_rotations[index + separation]).as_rotvec()
        if np.linalg.norm(mono_relative) > 0.015 and np.linalg.norm(ego_relative) > 0.015:
            source_vectors.append(ego_relative)
            target_vectors.append(mono_relative)
    camera_to_marker, _ = Rotation.align_vectors(target_vectors, source_vectors)
    candidates = ego_rotations * camera_to_marker.inv() * mono_rotations.inv()
    reference = candidates[0]
    ego_from_world = reference * Rotation.from_rotvec(
        np.mean((reference.inv() * candidates).as_rotvec(), axis=0)
    )

    rotation = ego_from_world.as_matrix()
    system, values = [], []
    for mono_position, mono_rotation, ego_position in zip(
        mono_positions, mono_rotations, ego_positions
    ):
        system.append(
            np.column_stack(
                [rotation @ mono_position, np.eye(3), rotation @ mono_rotation.as_matrix()]
            )
        )
        values.append(ego_position)
    linear = np.linalg.lstsq(np.vstack(system), np.hstack(values), rcond=None)[0]
    return np.r_[
        ego_from_world.as_rotvec(), camera_to_marker.as_rotvec(),
        linear[1:4], linear[4:7], np.log(max(float(linear[0]), 0.05)),
    ]


def _predict(parameters, mono_positions, mono_rotations):
    ego_from_world = Rotation.from_rotvec(parameters[:3])
    camera_to_marker = Rotation.from_rotvec(parameters[3:6])
    translation = parameters[6:9]
    camera_to_marker_translation = parameters[9:12]
    scale = float(np.exp(parameters[12]))
    predicted_rotations = ego_from_world * mono_rotations * camera_to_marker
    predicted_positions = (
        scale * ego_from_world.apply(mono_positions)
        + translation
        + (ego_from_world * mono_rotations).apply(
            np.tile(camera_to_marker_translation, (len(mono_positions), 1))
        )
    )
    return predicted_positions, predicted_rotations


def _solve_at_offset(mono_positions, mono_rotations, ego_positions, ego_rotations):
    initial = _initial_guess(mono_positions, mono_rotations, ego_positions, ego_rotations)

    def errors(parameters, selection=None):
        predicted_positions, predicted_rotations = _predict(
            parameters, mono_positions, mono_rotations
        )
        position_errors = ego_positions - predicted_positions
        rotation_errors = (ego_rotations.inv() * predicted_rotations).as_rotvec()
        if selection is not None:
            position_errors, rotation_errors = position_errors[selection], rotation_errors[selection]
        return position_errors, rotation_errors

    def residual(parameters, selection=None):
        position_errors, rotation_errors = errors(parameters, selection)
        return np.r_[
            position_errors.ravel() / 0.025,
            rotation_errors.ravel() / np.deg2rad(6.0),
        ]

    fitted = least_squares(residual, initial, loss="soft_l1", f_scale=1.2, max_nfev=160)
    position_errors, rotation_errors = errors(fitted.x)
    inliers = (
        (np.linalg.norm(position_errors, axis=1) < 0.08)
        & (np.linalg.norm(rotation_errors, axis=1) < np.deg2rad(20))
    )
    fitted = least_squares(
        lambda parameters: residual(parameters, inliers),
        fitted.x, loss="soft_l1", f_scale=1.0, max_nfev=120,
    )
    position_errors, rotation_errors = errors(fitted.x)
    position_norms = np.linalg.norm(position_errors, axis=1)
    rotation_norms = np.linalg.norm(rotation_errors, axis=1)
    score = float(np.median(position_norms) + 0.15 * np.median(rotation_norms))
    return fitted.x, inliers, position_norms, rotation_norms, score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono", required=True, type=Path)
    parser.add_argument("--stereo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-points", type=int, default=400)
    parser.add_argument("--late-jump-m", type=float, default=0.08)
    parser.add_argument("--divergence-m", type=float, default=0.05)
    parser.add_argument("--max-stereo-gap-s", type=float, default=0.15)
    args = parser.parse_args(argv)

    mono_rows, mono_times, mono_positions, mono_rotations = _load_mono(args.mono)
    stereo_times, stereo_positions, stereo_rotations = _load_measured_stereo(
        args.stereo, EGO_LOCAL_TO_UTC_US
    )
    trim_count = _auto_trim(mono_positions, args.late_jump_m)
    fit_count = min(args.fit_points, trim_count)
    fit_times_s = mono_times[:fit_count].astype(np.float64) / 1e6
    stereo_times_s = stereo_times.astype(np.float64) / 1e6

    candidates = []
    offsets = np.arange(-0.20, 0.2001, 0.02)
    for offset in offsets:
        indices, ego_positions, ego_rotations = _sample_stereo(
            fit_times_s, stereo_times_s, stereo_positions, stereo_rotations,
            float(offset), args.max_stereo_gap_s,
        )
        result = _solve_at_offset(
            mono_positions[:fit_count][indices], mono_rotations[:fit_count][indices],
            ego_positions, ego_rotations,
        )
        candidates.append((result[-1], float(offset), indices, result))
    _, coarse_offset, _, _ = min(candidates, key=lambda item: item[0])
    for offset in np.arange(coarse_offset - 0.02, coarse_offset + 0.0201, 0.005):
        indices, ego_positions, ego_rotations = _sample_stereo(
            fit_times_s, stereo_times_s, stereo_positions, stereo_rotations,
            float(offset), args.max_stereo_gap_s,
        )
        result = _solve_at_offset(
            mono_positions[:fit_count][indices], mono_rotations[:fit_count][indices],
            ego_positions, ego_rotations,
        )
        candidates.append((result[-1], float(offset), indices, result))
    _, time_offset_s, fit_indices, result = min(candidates, key=lambda item: item[0])
    parameters, inliers, position_errors, rotation_errors, _ = result

    # A failed VIO tail may drift smoothly before its first large single-frame
    # jump. Detect that failure against measured EGO poses using a persistent
    # 20-correspondence median, rather than relying on VIO step size alone.
    step_trim_count = trim_count
    validation_indices, validation_positions, validation_rotations = _sample_stereo(
        mono_times.astype(np.float64) / 1e6,
        stereo_times_s, stereo_positions, stereo_rotations,
        time_offset_s, args.max_stereo_gap_s,
    )
    predicted_validation, _ = _predict(
        parameters, mono_positions[validation_indices], mono_rotations[validation_indices]
    )
    validation_errors = np.linalg.norm(
        validation_positions - predicted_validation, axis=1
    )
    residual_trim_count = len(mono_positions)
    window = 20
    for index in range(len(validation_errors) - window + 1):
        if validation_indices[index] < fit_count:
            continue
        if np.median(validation_errors[index:index + window]) > args.divergence_m:
            residual_trim_count = int(validation_indices[index])
            break
    trim_count = min(step_trim_count, residual_trim_count)

    output_positions, output_rotations = _predict(
        parameters, mono_positions[:trim_count], mono_rotations[:trim_count]
    )
    output_times = mono_times[:trim_count] + int(round(time_offset_s * 1e6))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pure VIO camera optical centre expressed in EGO cam0.  This applies only
    # the fitted world Sim(3), without the camera-to-marker hand-eye offset.
    ego_from_world = Rotation.from_rotvec(parameters[:3])
    scale = float(np.exp(parameters[12]))
    camera_positions = (
        scale * ego_from_world.apply(mono_positions[:trim_count])
        + parameters[6:9]
    )
    camera_rotations = ego_from_world * mono_rotations[:trim_count]
    camera_path = args.output_dir / "mono_camera_in_ego_trimmed.csv"
    with camera_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_us", "time_s", "x", "y", "z", "qx", "qy", "qz", "qw", "source", "pose_frame"]
        )
        for timestamp, position, quaternion in zip(
            output_times, camera_positions, camera_rotations.as_quat()
        ):
            writer.writerow(
                [timestamp, f"{(timestamp-output_times[0])/1e6:.6f}",
                 *(f"{value:.9f}" for value in position),
                 *(f"{value:.9f}" for value in quaternion),
                 "mono_vio_sim3_coordinate_change_only", "ego_cam0_camera_optical_center"]
            )

    trajectory_path = args.output_dir / "mono_marker_fitted_ego_trimmed.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_us", "time_s", "x", "y", "z", "qx", "qy", "qz", "qw", "source", "pose_frame"]
        )
        quaternions = output_rotations.as_quat()
        for timestamp, position, quaternion in zip(output_times, output_positions, quaternions):
            writer.writerow(
                [timestamp, f"{(timestamp-output_times[0])/1e6:.6f}",
                 *(f"{value:.9f}" for value in position),
                 *(f"{value:.9f}" for value in quaternion),
                 "mono_vio_hand_eye_fitted_marker", "ego_cam0"]
            )

    report = {
        "method": "timestamp interpolation + robust Sim(3) hand-eye trajectory alignment",
        "ego_timestamp_correction_us": -EGO_LOCAL_TO_UTC_US,
        "optimized_time_offset_s_added_to_mono": time_offset_s,
        "fit_points_requested": fit_count,
        "fit_correspondences": len(fit_indices),
        "fit_inliers": int(inliers.sum()),
        "trimmed_output_points": trim_count,
        "removed_late_points": len(mono_positions) - trim_count,
        "step_based_trim_points": step_trim_count,
        "ego_residual_based_trim_points": residual_trim_count,
        "divergence_window_median_threshold_m": args.divergence_m,
        "scale": float(np.exp(parameters[12])),
        "R_ego_world": Rotation.from_rotvec(parameters[:3]).as_matrix().tolist(),
        "t_ego_world_m": parameters[6:9].tolist(),
        "R_camera_marker": Rotation.from_rotvec(parameters[3:6]).as_matrix().tolist(),
        "t_camera_marker_m": parameters[9:12].tolist(),
        "position_error_m_p50_p90": np.percentile(position_errors, [50, 90]).tolist(),
        "rotation_error_deg_p50_p90": np.rad2deg(np.percentile(rotation_errors, [50, 90])).tolist(),
        "trajectory": str(trajectory_path),
        "pure_vio_camera_trajectory_in_ego": str(camera_path),
    }
    report_path = args.output_dir / "hand_eye_alignment.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
