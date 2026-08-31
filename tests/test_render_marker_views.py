from __future__ import annotations

from nexumi_marker_pose.render_marker_views import render_marker_svg


def test_svg_contains_all_face_labels() -> None:
    faces = []
    for index in range(10):
        faces.append(
            {
                "sticker_id": f"F{index}",
                "color": {"srgb8": [index * 10, 100, 200]},
                "step_face": {
                    "vertices_mm": [
                        [index, 0, 0],
                        [index + 1, 0, 0],
                        [index, 1, 1],
                    ]
                },
            }
        )
    svg = render_marker_svg({"marker_name": "test", "faces": faces})
    assert svg.startswith("<svg")
    assert svg.count("<polygon") == 40
    for index in range(10):
        assert f">F{index}</text>" in svg

