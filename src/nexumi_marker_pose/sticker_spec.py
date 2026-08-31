"""Authoritative specifications for the right-hand printed triangle stickers.

The values in this module are transcribed from the vector source PDF named
``right hand V4 0822 ... A4 100pct``.  RGB values come from the PDF fill
operators (not from a rasterised screenshot), and side lengths retain the
order printed in the PDF.  All dimensions are millimetres.

This module intentionally has no third-party Python dependencies.  The small
PDF geometry inspector is not a general-purpose PDF reader; it checks the
properties important for printing this known Matplotlib-generated artefact:
A4 page boxes and one true 100 mm vector ruler per page.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
import zlib


RGB = tuple[int, int, int]
TriangleSidesMM = tuple[float, float, float]

MM_PER_INCH = 25.4
PDF_POINTS_PER_INCH = 72.0
MM_PER_PDF_POINT = MM_PER_INCH / PDF_POINTS_PER_INCH

A4_SIZE_MM = (210.0, 297.0)
EXPECTED_PDF_PAGE_COUNT = 2
CALIBRATION_RULER_MM = 100.0


@dataclass(frozen=True, slots=True)
class StickerSpec:
    """One printed triangular sticker from the right-hand layout."""

    face_id: str
    name_zh: str
    color_zh: str
    rgb: RGB
    sides_mm: TriangleSidesMM

    def __post_init__(self) -> None:
        if not re.fullmatch(r"F\d+", self.face_id):
            raise ValueError(f"invalid face id: {self.face_id!r}")
        if not self.name_zh or not self.color_zh:
            raise ValueError(f"{self.face_id}: names must not be empty")
        _validate_rgb(self.face_id, self.rgb)
        _validate_triangle(self.face_id, self.sides_mm)

    @property
    def sticker_id(self) -> str:
        """Compatibility alias for code that calls the label a sticker ID."""

        return self.face_id

    @property
    def side_lengths_mm(self) -> TriangleSidesMM:
        """Explicit alias documenting that ``sides_mm`` contains lengths."""

        return self.sides_mm


def _validate_rgb(face_id: str, rgb: Sequence[int]) -> None:
    if len(rgb) != 3:
        raise ValueError(f"{face_id}: RGB must have exactly three channels")
    if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in rgb):
        raise ValueError(f"{face_id}: RGB channels must be integers")
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"{face_id}: RGB channels must be in [0, 255]")


def _validate_triangle(face_id: str, sides_mm: Sequence[float]) -> None:
    if len(sides_mm) != 3:
        raise ValueError(f"{face_id}: a triangle must have exactly three sides")
    if any(not isfinite(side) or side <= 0.0 for side in sides_mm):
        raise ValueError(f"{face_id}: side lengths must be finite and positive")
    a, b, c = sorted(float(side) for side in sides_mm)
    if a + b <= c:
        raise ValueError(f"{face_id}: side lengths violate the triangle inequality")


# Exact sRGB 8-bit fills embedded in the right-hand vector PDF.
COLOR_RGB: Mapping[str, RGB] = MappingProxyType(
    {
        "品红": (214, 42, 120),
        "红": (228, 61, 48),
        "青": (0, 159, 191),
        "黄": (242, 197, 0),
        "绿": (36, 161, 72),
        "蓝": (49, 86, 200),
        "浅灰": (201, 206, 211),
    }
)


RIGHT_HAND_STICKERS: tuple[StickerSpec, ...] = (
    StickerSpec("F0", "顶部主三角A", "品红", COLOR_RGB["品红"], (55.2, 69.7, 83.1)),
    StickerSpec("F1", "顶部主三角B", "红", COLOR_RGB["红"], (66.5, 81.2, 39.2)),
    StickerSpec("F2", "侧面A上斜面", "青", COLOR_RGB["青"], (49.5, 63.4, 67.6)),
    StickerSpec("F3", "侧面A下斜面", "黄", COLOR_RGB["黄"], (40.1, 60.5, 37.1)),
    StickerSpec("F4", "侧面A转角面", "绿", COLOR_RGB["绿"], (40.4, 37.8, 39.1)),
    StickerSpec("F5", "端部中央斜面", "蓝", COLOR_RGB["蓝"], (62.6, 53.8, 40.0)),
    StickerSpec("F6", "端部下斜面", "浅灰", COLOR_RGB["浅灰"], (61.5, 44.1, 38.1)),
    StickerSpec("F7", "侧面B转角面", "红", COLOR_RGB["红"], (37.9, 37.7, 39.9)),
    StickerSpec("F8", "侧面B上斜面", "黄", COLOR_RGB["黄"], (68.6, 41.2, 73.9)),
    StickerSpec("F9", "侧面B下斜面", "蓝", COLOR_RGB["蓝"], (37.1, 48.4, 71.2)),
)

STICKERS_BY_ID: Mapping[str, StickerSpec] = MappingProxyType(
    {sticker.face_id: sticker for sticker in RIGHT_HAND_STICKERS}
)
# Short aliases for downstream geometry code.
STICKER_SPECS = RIGHT_HAND_STICKERS
STICKER_BY_ID = STICKERS_BY_ID


def get_sticker(face_id: str) -> StickerSpec:
    """Return a sticker by ``F0`` ... ``F9`` and preserve KeyError semantics."""

    return STICKERS_BY_ID[face_id]


def validate_sticker_specs(specs: Iterable[StickerSpec] = RIGHT_HAND_STICKERS) -> None:
    """Validate the complete right-hand sticker collection.

    ``StickerSpec`` validates individual records at construction time.  This
    function additionally validates collection-wide identity and colour
    consistency.  It returns ``None`` and raises ``ValueError`` on bad data.
    """

    records = tuple(specs)
    expected_ids = tuple(f"F{index}" for index in range(10))
    actual_ids = tuple(record.face_id for record in records)
    if len(records) != 10:
        raise ValueError(f"expected 10 right-hand stickers, got {len(records)}")
    if actual_ids != expected_ids:
        raise ValueError(f"expected ordered IDs {expected_ids}, got {actual_ids}")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("sticker IDs must be unique")
    for record in records:
        expected_rgb = COLOR_RGB.get(record.color_zh)
        if expected_rgb is None:
            raise ValueError(f"{record.face_id}: unknown colour {record.color_zh!r}")
        if record.rgb != expected_rgb:
            raise ValueError(
                f"{record.face_id}: {record.color_zh} must be {expected_rgb}, got {record.rgb}"
            )


@dataclass(frozen=True, slots=True)
class PdfPrintCheck:
    """Measured print-critical geometry from a source PDF."""

    page_count: int
    page_sizes_mm: tuple[tuple[float, float], ...]
    calibration_rulers_mm: tuple[float, ...]

    def validation_errors(
        self,
        *,
        page_tolerance_mm: float = 0.05,
        ruler_tolerance_mm: float = 0.05,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if self.page_count != EXPECTED_PDF_PAGE_COUNT:
            errors.append(
                f"expected {EXPECTED_PDF_PAGE_COUNT} pages, got {self.page_count}"
            )
        if len(self.page_sizes_mm) != self.page_count:
            errors.append(
                f"found {len(self.page_sizes_mm)} page boxes for {self.page_count} pages"
            )
        for page_number, (width, height) in enumerate(self.page_sizes_mm, 1):
            if (
                abs(width - A4_SIZE_MM[0]) > page_tolerance_mm
                or abs(height - A4_SIZE_MM[1]) > page_tolerance_mm
            ):
                errors.append(
                    f"page {page_number} is {width:.3f} x {height:.3f} mm, not A4"
                )
        if len(self.calibration_rulers_mm) != self.page_count:
            errors.append(
                "expected one 100 mm calibration ruler per page, "
                f"found {len(self.calibration_rulers_mm)}"
            )
        for ruler_number, length in enumerate(self.calibration_rulers_mm, 1):
            if abs(length - CALIBRATION_RULER_MM) > ruler_tolerance_mm:
                errors.append(
                    f"calibration ruler {ruler_number} is {length:.3f} mm, not 100 mm"
                )
        return tuple(errors)

    @property
    def valid(self) -> bool:
        return not self.validation_errors()

    def assert_valid(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise ValueError("invalid right-hand sticker PDF: " + "; ".join(errors))


_NUMBER = rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_PAGE_RE = re.compile(rb"/Type\s*/Page(?!s)\b")
_MEDIA_BOX_RE = re.compile(
    rb"/MediaBox\s*\[\s*(" + _NUMBER + rb")\s+(" + _NUMBER + rb")\s+"
    rb"(" + _NUMBER + rb")\s+(" + _NUMBER + rb")\s*\]"
)
_LINE_RE = re.compile(
    rb"(" + _NUMBER + rb")\s+(" + _NUMBER + rb")\s+m\s+"
    rb"(" + _NUMBER + rb")\s+(" + _NUMBER + rb")\s+l\s+S(?:\s|$)"
)
_STREAM_RE = re.compile(rb"(?<!end)stream\r?\n")


def _inflate_pdf_streams(pdf_data: bytes) -> Iterable[bytes]:
    """Yield zlib streams; unsupported PDF filters are deliberately ignored."""

    for match in _STREAM_RE.finditer(pdf_data):
        end = pdf_data.find(b"endstream", match.end())
        if end < 0:
            continue
        compressed = pdf_data[match.end() : end]
        try:
            yield zlib.decompress(compressed)
        except zlib.error:
            continue


def inspect_pdf_print_geometry(path: str | Path) -> PdfPrintCheck:
    """Inspect A4 page boxes and physical 100 mm vector rulers.

    The ruler is detected geometrically as a horizontal PDF path whose length
    is 100 mm.  Thus this check verifies physical scale rather than merely
    trusting the nearby ``100 mm`` text label.
    """

    pdf_path = Path(path)
    data = pdf_path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"not a PDF file: {pdf_path}")

    page_count = len(_PAGE_RE.findall(data))
    page_sizes: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in _MEDIA_BOX_RE.findall(data):
        width_pt = abs(float(x1) - float(x0))
        height_pt = abs(float(y1) - float(y0))
        page_sizes.append((width_pt * MM_PER_PDF_POINT, height_pt * MM_PER_PDF_POINT))

    rulers: list[float] = []
    # Use a narrow search window to distinguish the ruler from triangle edges.
    candidate_tolerance_mm = 0.25
    for stream in _inflate_pdf_streams(data):
        for x0, y0, x1, y1 in _LINE_RE.findall(stream):
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)
            if abs(dy) > 1e-6:
                continue
            length_mm = hypot(dx, dy) * MM_PER_PDF_POINT
            if abs(length_mm - CALIBRATION_RULER_MM) <= candidate_tolerance_mm:
                rulers.append(length_mm)

    return PdfPrintCheck(page_count, tuple(page_sizes), tuple(rulers))


def validate_right_hand_pdf_layout(path: str | Path) -> PdfPrintCheck:
    """Inspect the PDF, raise on print-scale errors, and return measurements."""

    check = inspect_pdf_print_geometry(path)
    check.assert_valid()
    return check


# Fail early during development if this module is edited inconsistently.
validate_sticker_specs()


__all__ = [
    "A4_SIZE_MM",
    "CALIBRATION_RULER_MM",
    "COLOR_RGB",
    "EXPECTED_PDF_PAGE_COUNT",
    "PdfPrintCheck",
    "RIGHT_HAND_STICKERS",
    "RGB",
    "STICKER_BY_ID",
    "STICKER_SPECS",
    "STICKERS_BY_ID",
    "StickerSpec",
    "TriangleSidesMM",
    "get_sticker",
    "inspect_pdf_print_geometry",
    "validate_right_hand_pdf_layout",
    "validate_sticker_specs",
]
