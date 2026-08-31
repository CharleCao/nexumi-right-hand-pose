"""CLI for estimating and visually checking one EGO stereo marker pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .color_detection import detect_colored_triangles
from .ego_calibration import load_ego_stereo_calibration
from .marker_pose import estimate_marker_pose
from .stereo_frames import decode_frames_by_index


def _roi(value: str) -> tuple[int, int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return result  # type: ignore[return-value]


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _overlay(image, camera_id, calibration, model, estimate):
    output = image.copy()
    for face in model["faces"]:
        vertices = np.asarray(face["step_face"]["vertices_mm"], dtype=np.float64)
        points = vertices @ estimate.pose.R_cam0_marker.T + estimate.pose.t_cam0_marker_mm
        if camera_id == 1:
            points = calibration.transform_cam0_to_cam1(points)
            pixels = calibration.cam1.project(points)
        else:
            pixels = calibration.cam0.project(points)
        polygon = np.rint(pixels).astype(np.int32)
        rgb = face["color"]["srgb8"]
        bgr = tuple(int(value) for value in reversed(rgb))
        cv2.polylines(output, [polygon], True, bgr, 3, cv2.LINE_AA)
        centre = tuple(np.rint(pixels.mean(axis=0)).astype(int))
        cv2.putText(output, face["sticker_id"], centre, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(output, face["sticker_id"], centre, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--left-video", required=True, type=Path)
    parser.add_argument("--right-video", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--left-roi", required=True, type=_roi)
    parser.add_argument("--right-roi", required=True, type=_roi)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    left_image = next(decode_frames_by_index(args.left_video, [args.frame])).image
    right_image = next(decode_frames_by_index(args.right_video, [args.frame])).image
    calibration = load_ego_stereo_calibration(args.calibration)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    estimate = estimate_marker_pose(
        calibration,
        model,
        detect_colored_triangles(left_image, roi=args.left_roi),
        detect_colored_triangles(right_image, roi=args.right_roi),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{args.frame:06d}"
    cv2.imwrite(str(args.output_dir / f"{stem}_left.png"),
                _overlay(left_image, 0, calibration, model, estimate))
    cv2.imwrite(str(args.output_dir / f"{stem}_right.png"),
                _overlay(right_image, 1, calibration, model, estimate))
    report = {
        "frame_index": args.frame,
        "pose_convention": "p_cam0_mm = R_cam0_marker @ p_marker_mm + t_cam0_marker_mm",
        "R_cam0_marker": estimate.pose.R_cam0_marker.tolist(),
        "t_cam0_marker_mm": estimate.pose.t_cam0_marker_mm.tolist(),
        "centroid_rmse_px": estimate.pose.rmse_px,
        "corner_rmse_px": estimate.corner_rmse_px,
        "score": estimate.score,
        "hypotheses_evaluated": estimate.hypotheses_evaluated,
        "assignments": [
            {
                "sticker_id": item.sticker_id,
                "detected_color": item.color_name,
                "right_candidate": item.primary_candidate_index,
                "left_candidate": item.secondary_candidate_index,
            }
            for item in estimate.assignments
        ],
    }
    _write_json_atomic(args.output_dir / f"{stem}_pose.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
