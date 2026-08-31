"""Render lightweight SVG views of a built marker model for geometry QA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence


Point3 = tuple[float, float, float]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalise(v: Sequence[float]) -> Point3:
    norm = sqrt(_dot(v, v))
    if norm == 0:
        raise ValueError("view vector must be non-zero")
    return tuple(value / norm for value in v)  # type: ignore[return-value]


@dataclass(frozen=True)
class View:
    name: str
    sight: Point3
    up_hint: Point3

    def basis(self) -> tuple[Point3, Point3, Point3]:
        depth = _normalise(self.sight)
        right = _normalise(_cross(depth, self.up_hint))
        up = _normalise(_cross(right, depth))
        return right, up, depth


VIEWS = (
    View("等轴视图", (1.0, -1.4, 1.0), (0.0, 0.0, 1.0)),
    View("XY / 顶视", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    View("XZ / 侧视", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    View("YZ / 端视", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)


def _project(point: Point3, view: View) -> tuple[float, float, float]:
    right, up, depth = view.basis()
    return _dot(point, right), _dot(point, up), _dot(point, depth)


def _rgb(rgb: Sequence[int]) -> str:
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def render_marker_svg(model: dict, *, width: int = 1280, height: int = 900) -> str:
    faces = model.get("faces", [])
    if len(faces) != 10:
        raise ValueError(f"expected 10 marker faces, got {len(faces)}")

    panel_w, panel_h = width / 2, (height - 60) / 2
    all_points: list[Point3] = [
        tuple(float(value) for value in point)  # type: ignore[misc]
        for face in faces
        for point in face["step_face"]["vertices_mm"]
    ]
    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f7f9"/>',
        '<text x="28" y="34" font-family="sans-serif" font-size="22" font-weight="700">'
        f'{escape(model.get("marker_name", "right hand marker"))} - STEP native mm</text>',
    ]

    for view_index, view in enumerate(VIEWS):
        col, row = view_index % 2, view_index // 2
        ox, oy = col * panel_w, 60 + row * panel_h
        projected = [_project(point, view) for point in all_points]
        xs, ys = [p[0] for p in projected], [p[1] for p in projected]
        span_x = max(max(xs) - min(xs), 1.0)
        span_y = max(max(ys) - min(ys), 1.0)
        scale = min((panel_w - 90) / span_x, (panel_h - 90) / span_y)
        centre_x = (min(xs) + max(xs)) / 2
        centre_y = (min(ys) + max(ys)) / 2

        def screen(point: Point3) -> tuple[float, float]:
            x, y, _ = _project(point, view)
            return (
                ox + panel_w / 2 + (x - centre_x) * scale,
                oy + panel_h / 2 - (y - centre_y) * scale,
            )

        body.append(
            f'<rect x="{ox + 12:.1f}" y="{oy + 8:.1f}" width="{panel_w - 24:.1f}" '
            f'height="{panel_h - 16:.1f}" rx="8" fill="white" stroke="#d8dde5"/>'
        )
        body.append(
            f'<text x="{ox + 28:.1f}" y="{oy + 36:.1f}" font-family="sans-serif" '
            f'font-size="17" font-weight="600">{escape(view.name)}</text>'
        )

        sorted_faces = sorted(
            faces,
            key=lambda face: sum(
                _project(tuple(point), view)[2]
                for point in face["step_face"]["vertices_mm"]
            )
            / 3.0,
        )
        for face in sorted_faces:
            vertices = [tuple(point) for point in face["step_face"]["vertices_mm"]]
            pixels = [screen(point) for point in vertices]
            points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in pixels)
            fill = _rgb(face["color"]["srgb8"])
            body.append(
                f'<polygon points="{points_text}" fill="{fill}" fill-opacity="0.90" '
                'stroke="#20242b" stroke-width="1.4" stroke-linejoin="round"/>'
            )
            cx = sum(point[0] for point in pixels) / 3
            cy = sum(point[1] for point in pixels) / 3
            label = escape(face["sticker_id"])
            body.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="10" fill="white" '
                'fill-opacity="0.78" stroke="#20242b" stroke-width="0.7"/>'
            )
            body.append(
                f'<text x="{cx:.2f}" y="{cy + 4:.2f}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" font-weight="700">{label}</text>'
            )

    body.append("</svg>")
    return "\n".join(body) + "\n"


def write_marker_svg(model_path: str | Path, output_path: str | Path) -> Path:
    source = Path(model_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        model = json.load(stream)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_marker_svg(model), encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SVG QA views of a marker model")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = write_marker_svg(args.model, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

