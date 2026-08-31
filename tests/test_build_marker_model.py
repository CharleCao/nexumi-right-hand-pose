from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexumi_marker_pose.build_marker_model import (
    build_marker_model,
    write_marker_model,
)


RIGHT_PDF = Path("/home/charlie/Downloads/右手_V4_0822彩色三角贴片_A4_100pct.pdf")
RIGHT_STEP = Path("/home/charlie/Downloads/上盖2.stp")
EXPECTED_MAPPING = {
    "F0": 25,
    "F1": 378,
    "F2": 54,
    "F3": 385,
    "F4": 389,
    "F5": 400,
    "F6": 404,
    "F7": 408,
    "F8": 412,
    "F9": 416,
}


@pytest.mark.skipif(
    not RIGHT_PDF.is_file() or not RIGHT_STEP.is_file(),
    reason="local right-hand design assets are unavailable",
)
def test_real_right_hand_design_builds_complete_model(tmp_path: Path) -> None:
    pytest.importorskip("OCC")
    model = build_marker_model(RIGHT_PDF, RIGHT_STEP)

    assert model["handedness"] == "right"
    assert model["matching"]["complete"] is True
    assert len(model["faces"]) == 10
    assert {
        face["sticker_id"]: face["match"]["step_face_index"]
        for face in model["faces"]
    } == EXPECTED_MAPPING
    assert max(
        face["match"]["max_abs_error_mm"] for face in model["faces"]
    ) <= 0.25

    output = write_marker_model(model, tmp_path / "right_hand_marker.json")
    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["sources"]["right_hand_sticker_pdf"]["sha256"]
    assert decoded["sources"]["step_geometry"]["sha256"]
    assert decoded["faces"][0]["color"]["name_zh"] == "品红"

