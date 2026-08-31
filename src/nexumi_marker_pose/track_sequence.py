"""Generate a continuous right-hand marker trajectory for one EGO session."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp

from .color_detection import ColorDetectionConfig, detect_colored_triangles
from .ego_calibration import load_ego_stereo_calibration
from .inspect_pose import _overlay
from .stereo_frames import iter_stereo_frames, pair_timestamp_csvs
from .stereo_pose import PoseObservation, StereoPoseResult, solve_stereo_pose


@dataclass(slots=True)
class TrackSample:
    valid: bool
    pose: StereoPoseResult | None
    matched_faces: int = 0
    reason: str = ""
    interpolated: bool = False
    matches: list[dict] | None = None


def _discover(session: Path) -> dict[str, Path]:
    ego_dirs = sorted(path for path in session.glob("EGO_*") if path.is_dir())
    if len(ego_dirs) != 1:
        raise ValueError(f"expected one EGO_* directory in {session}, found {len(ego_dirs)}")
    ego = ego_dirs[0]
    part = ego / "part0001"
    def one(pattern: str, root: Path = part) -> Path:
        values = sorted(root.glob(pattern))
        if len(values) != 1:
            raise ValueError(f"expected one {pattern} in {root}, found {len(values)}")
        return values[0]
    return {
        "calibration": one("*_calibration_camera.yaml", ego),
        "left_video": one("*_camera_left_*.mp4"),
        "right_video": one("*_camera_right_*.mp4"),
        "left_pts": one("*_camera_left_*_pts.csv"),
        "right_pts": one("*_camera_right_*_pts.csv"),
    }


def _roi_from_pose(image: np.ndarray, camera_id: int, calibration, model, pose,
                   padding: int) -> tuple[int, int, int, int]:
    points = np.concatenate([
        np.asarray(face["step_face"]["vertices_mm"], np.float64)
        for face in model["faces"]
    ])
    points = points @ pose.R_cam0_marker.T + pose.t_cam0_marker_mm
    if camera_id == 1:
        points = calibration.transform_cam0_to_cam1(points)
        pixels = calibration.cam1.project(points)
    else:
        pixels = calibration.cam0.project(points)
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(pixels[:, 0].min())) - padding)
    y0 = max(0, int(np.floor(pixels[:, 1].min())) - padding)
    x1 = min(width, int(np.ceil(pixels[:, 0].max())) + padding)
    y1 = min(height, int(np.ceil(pixels[:, 1].max())) + padding)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _project_face(face, camera_id, calibration, pose) -> np.ndarray:
    points = np.asarray(face["step_face"]["vertices_mm"], np.float64)
    points = points @ pose.R_cam0_marker.T + pose.t_cam0_marker_mm
    if camera_id == 1:
        return calibration.cam1.project(calibration.transform_cam0_to_cam1(points))
    return calibration.cam0.project(points)


def _best_corner_order(predicted: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float]:
    choices = []
    for order in permutations(range(3)):
        ordered = observed[list(order)]
        rmse = float(np.sqrt(np.mean(np.sum((predicted - ordered) ** 2, axis=1))))
        choices.append((rmse, ordered))
    rmse, ordered = min(choices, key=lambda item: item[0])
    return ordered, rmse


def _guided_observations(image, camera_id, calibration, model, prior,
                         padding: int, max_corner_error: float):
    roi = _roi_from_pose(image, camera_id, calibration, model, prior, padding)
    detected = detect_colored_triangles(
        image, roi=roi,
        config=ColorDetectionConfig(min_area_px=20.0),
    )
    candidates = [c for c in detected.candidates if c.area_px >= 60 and c.confidence >= .32]
    observations: list[PoseObservation] = []
    matched: list[str] = []
    records: list[dict] = []
    for color in {face["color"]["name_zh"] for face in model["faces"]}:
        faces = [face for face in model["faces"] if face["color"]["name_zh"] == color]
        values = [candidate for candidate in candidates if candidate.color_name == color]
        if not faces or not values:
            continue
        cost = np.full((len(faces), len(values)), 1e6, np.float64)
        orders: dict[tuple[int, int], np.ndarray] = {}
        for i, face in enumerate(faces):
            predicted = _project_face(face, camera_id, calibration, prior)
            for j, candidate in enumerate(values):
                ordered, error = _best_corner_order(predicted, candidate.triangle_xy)
                centroid_error = np.linalg.norm(predicted.mean(axis=0) - candidate.centroid_xy)
                if error <= max_corner_error and centroid_error <= 1.5 * max_corner_error:
                    cost[i, j] = error + 0.15 * centroid_error
                    orders[i, j] = ordered
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows, columns):
            if cost[i, j] >= 1e5:
                continue
            face = faces[i]
            vertices = np.asarray(face["step_face"]["vertices_mm"], np.float64)
            for point, pixel in zip(vertices, orders[i, j]):
                observations.append(PoseObservation(point, pixel, camera_id,
                                                    label=face["sticker_id"]))
            matched.append(face["sticker_id"])
            predicted_record = _project_face(face, camera_id, calibration, prior)
            matched_candidate = values[j]
            _, matched_error = _best_corner_order(predicted_record, matched_candidate.triangle_xy)
            records.append({"sticker_id": face["sticker_id"],
                            "color": color, "observed_triangle": matched_candidate.triangle_xy.tolist(),
                            "predicted_triangle": predicted_record.tolist(), "corner_error_px": float(matched_error)})
    return observations, matched, roi, records


def _track_one(left, right, calibration, model, prior, padding=85) -> TrackSample:
    try:
        observations = []
        matched = set()
        records: list[dict] = []
        for camera_id, image in ((0, left), (1, right)):
            obs, ids, _, camera_records = _guided_observations(
                image, camera_id, calibration, model, prior, padding, 48.0
            )
            observations.extend(obs)
            matched.update(ids)
            for record in camera_records:
                record["camera_id"] = camera_id
            records.extend(camera_records)
        if len(observations) < 6 or len(matched) < 2:
            return TrackSample(False, None, len(matched), "too few matched faces", matches=records)
        pose = solve_stereo_pose(
            calibration, observations,
            initial_R_cam0_marker=prior.R_cam0_marker,
            initial_t_cam0_marker_mm=prior.t_cam0_marker_mm,
            loss="huber", f_scale_px=3.0, inlier_threshold_px=8.0,
        )
        translation_jump = np.linalg.norm(pose.t_cam0_marker_mm - prior.t_cam0_marker_mm)
        rotation_jump = (Rotation.from_matrix(pose.R_cam0_marker)
                         * Rotation.from_matrix(prior.R_cam0_marker).inv()).magnitude()
        inlier_count = int(pose.inlier_mask.sum())
        # One partly occluded triangle contributes three large residuals, but
        # must not invalidate an otherwise well-constrained robust solution.
        if (translation_jump > 70 or rotation_jump > np.deg2rad(35)
                or pose.median_error_px > 10 or inlier_count < 6):
            return TrackSample(False, None, len(matched),
                               f"quality gate: dt={translation_jump:.1f}mm, "
                               f"dr={np.rad2deg(rotation_jump):.1f}deg, "
                               f"median={pose.median_error_px:.1f}px, inliers={inlier_count}", matches=records)
        return TrackSample(True, pose, len(matched), matches=records)
    except Exception as error:
        return TrackSample(False, None, 0, str(error), matches=records)


def _pose_from_report(path: Path) -> StereoPoseResult:
    value = json.loads(path.read_text(encoding="utf-8"))
    rotation = np.asarray(value["R_cam0_marker"], np.float64)
    translation = np.asarray(value["t_cam0_marker_mm"], np.float64)
    return StereoPoseResult(rotation, translation, Rotation.from_matrix(rotation).as_rotvec(),
                            np.empty(0), np.empty(0, bool),
                            float(value["centroid_rmse_px"]), 0.0, None, None,
                            True, "anchor", 0)


def _encode_frames(paths, pairing):
    encoded = []
    for index, frame in enumerate(iter_stereo_frames(paths["left_video"], paths["right_video"], pairing)):
        left_ok, left = cv2.imencode(".jpg", frame.left, [cv2.IMWRITE_JPEG_QUALITY, 96])
        right_ok, right = cv2.imencode(".jpg", frame.right, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not left_ok or not right_ok:
            raise RuntimeError(f"failed to buffer frame {index}")
        encoded.append((left, right))
    return encoded


def _decode(pair):
    return cv2.imdecode(pair[0], cv2.IMREAD_COLOR), cv2.imdecode(pair[1], cv2.IMREAD_COLOR)


def _synthetic_pose(rotation: np.ndarray, translation: np.ndarray) -> StereoPoseResult:
    return StereoPoseResult(rotation, translation, Rotation.from_matrix(rotation).as_rotvec(),
                            np.empty(0), np.empty(0, bool), float("nan"), float("nan"),
                            None, None, True, "interpolated", 0)


def _fill_short_gaps(samples: list[TrackSample], max_gap: int = 5) -> None:
    """Fill brief detector dropouts so every frame has a usable 6-DoF pose."""
    index = 0
    while index < len(samples):
        if samples[index].pose is not None:
            index += 1
            continue
        start = index
        while index < len(samples) and samples[index].pose is None:
            index += 1
        end = index
        if end - start > max_gap:
            continue
        before = start - 1
        after = end if end < len(samples) else None
        if before >= 0 and after is not None:
            first, second = samples[before].pose, samples[after].pose
            assert first is not None and second is not None
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix(
                [first.R_cam0_marker, second.R_cam0_marker]))
            for current in range(start, end):
                alpha = (current - before) / (after - before)
                translation = ((1-alpha) * first.t_cam0_marker_mm
                               + alpha * second.t_cam0_marker_mm)
                samples[current] = TrackSample(
                    True, _synthetic_pose(slerp([alpha]).as_matrix()[0], translation),
                    reason="interpolated", interpolated=True)
        elif before >= 1:
            older, first = samples[before-1].pose, samples[before].pose
            assert older is not None and first is not None
            velocity = first.t_cam0_marker_mm - older.t_cam0_marker_mm
            delta_rotation = (Rotation.from_matrix(first.R_cam0_marker)
                              * Rotation.from_matrix(older.R_cam0_marker).inv())
            for current in range(start, end):
                steps = current - before
                rotation = (delta_rotation ** steps
                            * Rotation.from_matrix(first.R_cam0_marker)).as_matrix()
                samples[current] = TrackSample(
                    True, _synthetic_pose(rotation, first.t_cam0_marker_mm + steps*velocity),
                    reason="extrapolated", interpolated=True)


def _write_outputs(output: Path, samples, pairing, calibration, model, encoded, fps):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "trajectory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["frame_index", "timestamp_us", "left_index", "right_index", "sync_delta_us",
                  "valid", "interpolated", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw",
                  "rmse_px", "matched_faces", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for index, (sample, pair) in enumerate(zip(samples, pairing.pairs)):
            row = {"frame_index": index, "timestamp_us": pair.timestamp_us,
                   "left_index": pair.left_index, "right_index": pair.right_index,
                   "sync_delta_us": pair.delta_us, "valid": int(sample.valid),
                   "interpolated": int(sample.interpolated),
                   "matched_faces": sample.matched_faces, "reason": sample.reason}
            if sample.pose is not None:
                q = Rotation.from_matrix(sample.pose.R_cam0_marker).as_quat()
                row.update(dict(zip(("tx_mm","ty_mm","tz_mm"), sample.pose.t_cam0_marker_mm)))
                row.update(dict(zip(("qx","qy","qz","qw"), q)))
                if not sample.interpolated:
                    row["rmse_px"] = sample.pose.rmse_px
            writer.writerow(row)
    valid = sum(sample.valid for sample in samples)
    interpolated = sum(sample.interpolated for sample in samples)
    report = {"frames": len(samples), "valid_frames": valid,
              "measured_frames": valid - interpolated,
              "interpolated_frames": interpolated,
              "valid_fraction": valid / len(samples), "fps": fps,
              "pose_convention": "p_cam0_mm = R_cam0_marker @ p_marker_mm + t_cam0_marker_mm",
              "trajectory_csv": str(csv_path)}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                         encoding="utf-8")
    with (output / "frame_matches.jsonl").open("w", encoding="utf-8") as stream:
        for index, (sample, pair) in enumerate(zip(samples, pairing.pairs)):
            stream.write(json.dumps({"frame_index": index, "timestamp_us": pair.timestamp_us,
                                     "sync_delta_us": pair.delta_us, "valid": sample.valid,
                                     "interpolated": sample.interpolated,
                                     "matches": sample.matches or [], "reason": sample.reason},
                                    ensure_ascii=False) + "\n")
    writer = cv2.VideoWriter(str(output / "trajectory_preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 650))
    last_pose = None
    for index, (sample, packed) in enumerate(zip(samples, encoded)):
        left, right = _decode(packed)
        if sample.pose is not None:
            last_pose = sample.pose
        if last_pose is not None:
            proxy = type("Estimate", (), {"pose": last_pose})()
            left = _overlay(left, 0, calibration, model, proxy)
            right = _overlay(right, 1, calibration, model, proxy)
        view = np.hstack((cv2.resize(left, (800,650)), cv2.resize(right, (800,650))))
        state = "INTERP" if sample.interpolated else ("VALID" if sample.valid else "LOST")
        color = (40,180,240) if sample.interpolated else ((40,220,40) if sample.valid else (30,30,240))
        cv2.putText(view, f"frame {index:04d}  {state}",
                    (25,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        writer.write(view)
    writer.release()
    import av
    match_container = av.open(str(output / "matching_preview.mp4"), "w",
                              options={"movflags": "+faststart"})
    match_stream = match_container.add_stream("libx264", rate=int(round(fps)))
    match_stream.width = 1600
    match_stream.height = 650
    match_stream.pix_fmt = "yuv420p"
    match_stream.options = {"preset": "fast", "crf": "22"}
    for index, (sample, packed) in enumerate(zip(samples, encoded)):
        left, right = _decode(packed)
        if sample.pose is not None:
            proxy = type("Estimate", (), {"pose": sample.pose})()
            left = _overlay(left, 0, calibration, model, proxy)
            right = _overlay(right, 1, calibration, model, proxy)
        for record in sample.matches or []:
            image = left if record["camera_id"] == 0 else right
            observed = np.rint(record["observed_triangle"]).astype(np.int32)
            predicted = np.rint(record["predicted_triangle"]).astype(np.int32)
            cv2.polylines(image, [observed], True, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.polylines(image, [predicted], True, (255, 255, 0), 2, cv2.LINE_AA)
            centre = tuple(np.rint(observed.mean(axis=0)).astype(int))
            cv2.putText(image, f'{record["sticker_id"]} {record["corner_error_px"]:.1f}px',
                        centre, cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(image, f'{record["sticker_id"]} {record["corner_error_px"]:.1f}px',
                        centre, cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
        view = np.hstack((cv2.resize(left, (800,650)), cv2.resize(right, (800,650))))
        cv2.putText(view, f"frame {index:04d}  matches={len(sample.matches or [])}",
                    (25,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,220,220), 3, cv2.LINE_AA)
        video_frame = av.VideoFrame.from_ndarray(view, format="bgr24")
        for packet in match_stream.encode(video_frame):
            match_container.mux(packet)
    for packet in match_stream.encode():
        match_container.mux(packet)
    match_container.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--anchor-pose", required=True, type=Path)
    parser.add_argument("--anchor-frame", type=int, default=440)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    paths = _discover(args.session)
    calibration = load_ego_stereo_calibration(paths["calibration"])
    model = json.loads(args.model.read_text(encoding="utf-8"))
    pairing = pair_timestamp_csvs(paths["left_pts"], paths["right_pts"], max_difference_us=200)
    encoded = _encode_frames(paths, pairing)
    if not 0 <= args.anchor_frame < len(encoded):
        raise ValueError("anchor frame is outside the recording")
    samples = [TrackSample(False, None, reason="not processed") for _ in encoded]
    anchor_pose = _pose_from_report(args.anchor_pose)
    samples[args.anchor_frame] = TrackSample(True, anchor_pose, 4, "anchor")
    prior = anchor_pose
    for index in range(args.anchor_frame + 1, len(encoded)):
        sample = _track_one(*_decode(encoded[index]), calibration, model, prior)
        samples[index] = sample
        if sample.pose is not None:
            prior = sample.pose
    prior = anchor_pose
    for index in range(args.anchor_frame - 1, -1, -1):
        sample = _track_one(*_decode(encoded[index]), calibration, model, prior)
        samples[index] = sample
        if sample.pose is not None:
            prior = sample.pose
    _fill_short_gaps(samples)
    report = _write_outputs(args.output_dir, samples, pairing, calibration, model,
                            encoded, args.fps)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
