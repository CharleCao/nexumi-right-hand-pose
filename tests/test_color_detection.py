from __future__ import annotations

import cv2
import numpy as np
import pytest

from nexumi_marker_pose.color_detection import (
    ColorDetectionConfig,
    detect_colored_triangles,
)
from nexumi_marker_pose.sticker_spec import COLOR_RGB


def _synthetic_marker(*, exposure: float = 1.0, cast_bgr=(1.0, 1.0, 1.0)):
    # A dark neutral background avoids pretending that every neutral surface
    # is the light-gray sticker; real scenes rely on geometry for that case.
    image = np.full((360, 560, 3), 35, dtype=np.uint8)
    centres = {}
    for index, (name, rgb) in enumerate(COLOR_RGB.items()):
        row, column = divmod(index, 4)
        x, y = 75 + column * 135, 85 + row * 165
        points = np.asarray([[x, y - 48], [x - 52, y + 43], [x + 52, y + 43]])
        cv2.fillConvexPoly(image, points, tuple(reversed(rgb)))
        centres[name] = (x, y)
    transformed = image.astype(np.float32) * exposure
    transformed *= np.asarray(cast_bgr, dtype=np.float32)
    return np.clip(transformed, 0, 255).astype(np.uint8), centres


def test_detects_all_pdf_colours_under_exposure_and_awb_change() -> None:
    image, centres = _synthetic_marker(exposure=0.68, cast_bgr=(1.12, 0.96, 0.88))
    result = detect_colored_triangles(image)
    detected = {candidate.color_name for candidate in result.candidates}
    assert detected == set(COLOR_RGB)
    for color_name, centre in centres.items():
        assert result.masks[color_name][centre[1], centre[0]] == 255
    assert all(candidate.triangle_xy.shape == (3, 2) for candidate in result.candidates)
    assert all(0.0 <= candidate.confidence <= 1.0 for candidate in result.candidates)


def test_long_coloured_background_stripe_is_not_a_triangle_candidate() -> None:
    image = np.full((240, 360, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (15, 105), (345, 130), tuple(reversed(COLOR_RGB["黄"])), -1)
    result = detect_colored_triangles(image)
    assert np.count_nonzero(result.masks["黄"]) > 0
    assert not any(candidate.color_name == "黄" for candidate in result.candidates)


def test_optional_roi_preserves_full_image_coordinates() -> None:
    image = np.full((260, 420, 3), 105, dtype=np.uint8)
    points = np.asarray([[310, 70], [255, 185], [365, 185]])
    cv2.fillConvexPoly(image, points, tuple(reversed(COLOR_RGB["蓝"])))
    result = detect_colored_triangles(image, roi=(220, 35, 180, 190))
    blue = next(candidate for candidate in result.candidates if candidate.color_name == "蓝")
    assert blue.centroid_xy == pytest.approx((310.0, 146.7), abs=1.0)
    assert blue.triangle_xy[:, 0].min() >= 250
    assert result.masks["蓝"].shape == image.shape[:2]
    assert result.roi == (220, 35, 180, 190)


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_rejects_invalid_image_layout(bad_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        detect_colored_triangles(bad_image)


def test_rejects_even_morphology_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        ColorDetectionConfig(morphology_kernel=4)
