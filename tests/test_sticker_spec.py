from pathlib import Path

import pytest

from nexumi_marker_pose.sticker_spec import (
    A4_SIZE_MM,
    CALIBRATION_RULER_MM,
    COLOR_RGB,
    RIGHT_HAND_STICKERS,
    STICKERS_BY_ID,
    StickerSpec,
    get_sticker,
    inspect_pdf_print_geometry,
    validate_right_hand_pdf_layout,
    validate_sticker_specs,
)


RIGHT_HAND_PDF = Path(
    "/home/charlie/Downloads/右手_V4_0822彩色三角贴片_A4_100pct.pdf"
)

EXPECTED = {
    "F0": ("顶部主三角A", "品红", (214, 42, 120), (55.2, 69.7, 83.1)),
    "F1": ("顶部主三角B", "红", (228, 61, 48), (66.5, 81.2, 39.2)),
    "F2": ("侧面A上斜面", "青", (0, 159, 191), (49.5, 63.4, 67.6)),
    "F3": ("侧面A下斜面", "黄", (242, 197, 0), (40.1, 60.5, 37.1)),
    "F4": ("侧面A转角面", "绿", (36, 161, 72), (40.4, 37.8, 39.1)),
    "F5": ("端部中央斜面", "蓝", (49, 86, 200), (62.6, 53.8, 40.0)),
    "F6": ("端部下斜面", "浅灰", (201, 206, 211), (61.5, 44.1, 38.1)),
    "F7": ("侧面B转角面", "红", (228, 61, 48), (37.9, 37.7, 39.9)),
    "F8": ("侧面B上斜面", "黄", (242, 197, 0), (68.6, 41.2, 73.9)),
    "F9": ("侧面B下斜面", "蓝", (49, 86, 200), (37.1, 48.4, 71.2)),
}


def test_all_ten_pdf_records_are_exact() -> None:
    assert len(RIGHT_HAND_STICKERS) == 10
    assert tuple(spec.face_id for spec in RIGHT_HAND_STICKERS) == tuple(EXPECTED)
    for spec in RIGHT_HAND_STICKERS:
        assert (spec.name_zh, spec.color_zh, spec.rgb, spec.sides_mm) == EXPECTED[
            spec.face_id
        ]


def test_lookup_and_colour_table_are_consistent() -> None:
    assert set(STICKERS_BY_ID) == set(EXPECTED)
    for spec in RIGHT_HAND_STICKERS:
        assert get_sticker(spec.face_id) is spec
        assert spec.rgb == COLOR_RGB[spec.color_zh]
        assert len(spec.rgb) == 3
        assert all(isinstance(channel, int) and 0 <= channel <= 255 for channel in spec.rgb)


def test_every_record_is_a_geometrically_valid_triangle() -> None:
    for spec in RIGHT_HAND_STICKERS:
        a, b, c = sorted(spec.sides_mm)
        assert a > 0.0
        assert a + b > c
    validate_sticker_specs()


@pytest.mark.parametrize(
    ("rgb", "sides"),
    [
        ((256, 0, 0), (3.0, 4.0, 5.0)),
        ((1.0, 2, 3), (3.0, 4.0, 5.0)),
        ((1, 2, 3), (1.0, 2.0, 3.0)),
        ((1, 2, 3), (1.0, 2.0, float("nan"))),
    ],
)
def test_bad_rgb_or_triangle_is_rejected(rgb, sides) -> None:
    with pytest.raises(ValueError):
        StickerSpec("F0", "测试面", "测试色", rgb, sides)


def test_collection_requires_exactly_f0_through_f9() -> None:
    with pytest.raises(ValueError, match="expected 10"):
        validate_sticker_specs(RIGHT_HAND_STICKERS[:-1])
    reordered = (RIGHT_HAND_STICKERS[1], RIGHT_HAND_STICKERS[0]) + RIGHT_HAND_STICKERS[2:]
    with pytest.raises(ValueError, match="ordered IDs"):
        validate_sticker_specs(reordered)


def test_right_hand_pdf_is_two_a4_pages_with_true_100mm_rulers() -> None:
    assert RIGHT_HAND_PDF.is_file(), f"missing source PDF: {RIGHT_HAND_PDF}"
    check = inspect_pdf_print_geometry(RIGHT_HAND_PDF)
    assert check.page_count == 2
    assert len(check.page_sizes_mm) == 2
    for size in check.page_sizes_mm:
        assert size == pytest.approx(A4_SIZE_MM, abs=0.01)
    assert check.calibration_rulers_mm == pytest.approx(
        (CALIBRATION_RULER_MM, CALIBRATION_RULER_MM), abs=0.01
    )
    assert check.valid
    assert validate_right_hand_pdf_layout(RIGHT_HAND_PDF) == check


def test_pdf_inspector_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "not.pdf"
    text_file.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="not a PDF"):
        inspect_pdf_print_geometry(text_file)
