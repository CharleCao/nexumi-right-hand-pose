"""Build the authoritative right-hand coloured marker model.

The STEP model is the metric geometry source.  The right-hand PDF contributes
the F0--F9 semantic labels, print colours, and rounded edge lengths used to
identify the recessed faces.  No face index is hard-coded: every assignment is
recovered by a global one-to-one edge-length match.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

from .face_matching import match_triangular_faces
from .step_geometry import read_step_geometry
from .sticker_spec import (
    RIGHT_HAND_STICKERS,
    validate_right_hand_pdf_layout,
)


SCHEMA_VERSION = "1.0"
MARKER_NAME = "right_hand_v4_0822"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _file_sha256(resolved),
    }


def build_marker_model(
    sticker_pdf: str | Path,
    step_path: str | Path,
    *,
    tolerance_mm: float = 0.25,
    ambiguity_margin_mm: float = 0.10,
) -> dict[str, Any]:
    """Validate both design sources and build a JSON-serializable model."""

    pdf_path = Path(sticker_pdf).expanduser().resolve()
    cad_path = Path(step_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if not cad_path.is_file():
        raise FileNotFoundError(cad_path)

    print_check = validate_right_hand_pdf_layout(pdf_path)
    geometry = read_step_geometry(cad_path)
    result = match_triangular_faces(
        {record.face_id: record.sides_mm for record in RIGHT_HAND_STICKERS},
        geometry.triangular_faces,
        tolerance_mm=tolerance_mm,
        ambiguity_margin_mm=ambiguity_margin_mm,
    )
    if not result.complete:
        missing = ", ".join(result.unmatched_sticker_ids)
        raise ValueError(
            "could not match every right-hand sticker to a STEP recessed face; "
            f"unmatched: {missing}"
        )
    if result.ambiguities or result.globally_ambiguous:
        details = ", ".join(item.sticker_id for item in result.ambiguities)
        raise ValueError(f"ambiguous sticker-to-STEP assignment: {details or 'global'}")

    faces_by_index = {face.face_index: face for face in geometry.triangular_faces}
    matches_by_id = {match.sticker_id: match for match in result.matches}
    faces: list[dict[str, Any]] = []
    for sticker in RIGHT_HAND_STICKERS:
        match = matches_by_id[sticker.face_id]
        face = faces_by_index[match.step_face_id]
        faces.append(
            {
                "sticker_id": sticker.face_id,
                "name_zh": sticker.name_zh,
                "color": {
                    "name_zh": sticker.color_zh,
                    "srgb8": list(sticker.rgb),
                },
                "pdf_sides_mm": list(sticker.sides_mm),
                "step_face": face.to_dict(),
                "match": {
                    "step_face_index": match.step_face_id,
                    "edge_residuals_mm": list(match.residuals_mm),
                    "rmse_mm": match.rmse_mm,
                    "max_abs_error_mm": match.max_abs_error_mm,
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "marker_name": MARKER_NAME,
        "handedness": "right",
        "coordinate_frame": {
            "name": "step_native",
            "length_unit": "mm",
            "hand_or_tool_transform_defined": False,
        },
        "sources": {
            "right_hand_sticker_pdf": _source_record(pdf_path),
            "step_geometry": _source_record(cad_path),
        },
        "pdf_print_check": {
            "page_count": print_check.page_count,
            "page_sizes_mm": [list(size) for size in print_check.page_sizes_mm],
            "calibration_rulers_mm": list(print_check.calibration_rulers_mm),
        },
        "step_summary": {
            "total_face_count": geometry.total_face_count,
            "planar_face_count": geometry.planar_face_count,
            "triangular_candidate_count": len(geometry.triangular_faces),
        },
        "matching": {
            "method": "global_one_to_one_sorted_edge_lengths",
            "tolerance_mm": result.tolerance_mm,
            "ambiguity_margin_mm": result.ambiguity_margin_mm,
            "global_rmse_mm": result.global_rmse_mm,
            "second_best_global_rmse_mm": result.second_best_global_rmse_mm,
            "complete": result.complete,
        },
        "faces": faces,
    }


def write_marker_model(model: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write *model* as stable, UTF-8 JSON."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metric right-hand coloured marker model from PDF + STEP"
    )
    parser.add_argument("--sticker-pdf", required=True, type=Path)
    parser.add_argument("--step", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=0.25)
    parser.add_argument("--ambiguity-margin-mm", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = build_marker_model(
        args.sticker_pdf,
        args.step,
        tolerance_mm=args.tolerance_mm,
        ambiguity_margin_mm=args.ambiguity_margin_mm,
    )
    output = write_marker_model(model, args.output)
    mapping = " ".join(
        f"{face['sticker_id']}->{face['match']['step_face_index']}"
        for face in model["faces"]
    )
    print(f"Wrote {output}")
    print(f"Matched {mapping}")


if __name__ == "__main__":
    main()

