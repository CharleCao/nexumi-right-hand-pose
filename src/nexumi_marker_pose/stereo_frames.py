"""Timestamp pairing and indexed decoding for EGO stereo recordings.

The timestamp CSV files are the source of capture time.  MP4 presentation
timestamps are deliberately not used for stereo association: the two camera
files have independent codec time bases, while the companion CSV files contain
the device clock in integer microseconds.

Images returned by this module are ``uint8`` arrays with shape ``(height,
width, 3)``.  ``color_order="BGR"`` returns OpenCV-compatible B, G, R channel
order; ``color_order="RGB"`` returns R, G, B channel order.  The recorded EGO
streams used by this project are 1600 x 1300 pixels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
from numpy.typing import NDArray


ColorOrder = Literal["BGR", "RGB"]
UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class StereoTimestampPair:
    """A one-to-one association between a left and a right video frame."""

    left_index: int
    right_index: int
    left_timestamp_us: int
    right_timestamp_us: int

    @property
    def delta_us(self) -> int:
        """Signed right-minus-left capture time in microseconds."""

        return self.right_timestamp_us - self.left_timestamp_us

    @property
    def abs_delta_us(self) -> int:
        return abs(self.delta_us)

    @property
    def timestamp_us(self) -> int:
        """Integer midpoint capture time, suitable as the stereo timestamp."""

        return (self.left_timestamp_us + self.right_timestamp_us) // 2


@dataclass(frozen=True, slots=True)
class StereoPairingReport:
    """Stereo associations plus diagnostics for dropped/unmatched frames."""

    pairs: tuple[StereoTimestampPair, ...]
    unmatched_left_indices: tuple[int, ...]
    unmatched_right_indices: tuple[int, ...]

    @property
    def deltas_us(self) -> tuple[int, ...]:
        return tuple(pair.delta_us for pair in self.pairs)

    @property
    def max_abs_delta_us(self) -> int | None:
        if not self.pairs:
            return None
        return max(pair.abs_delta_us for pair in self.pairs)

    @property
    def mean_abs_delta_us(self) -> float | None:
        if not self.pairs:
            return None
        return fmean(pair.abs_delta_us for pair in self.pairs)


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    index: int
    image: UInt8Image


@dataclass(frozen=True, slots=True)
class StereoFrame:
    """A decoded, timestamp-associated stereo image pair."""

    timestamps: StereoTimestampPair
    left: UInt8Image
    right: UInt8Image
    color_order: ColorOrder


def read_timestamp_csv(path: str | Path) -> tuple[int, ...]:
    """Read and validate a ``timestamp_us`` CSV column.

    Empty timestamps, non-integral values, duplicates, and clock regressions
    are rejected instead of being silently sorted: frame index is the CSV row
    index and changing row order would corrupt video association.
    """

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "timestamp_us" not in reader.fieldnames:
            raise ValueError(f"{csv_path} has no timestamp_us column")

        timestamps: list[int] = []
        for row_index, row in enumerate(reader):
            raw = row.get("timestamp_us")
            if raw is None or not raw.strip():
                raise ValueError(
                    f"{csv_path}: missing timestamp_us at data row {row_index}"
                )
            try:
                timestamp = int(raw)
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}: invalid timestamp_us {raw!r} at data row "
                    f"{row_index}"
                ) from error
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError(
                    f"{csv_path}: timestamps must be strictly increasing; "
                    f"row {row_index} has {timestamp} after {timestamps[-1]}"
                )
            timestamps.append(timestamp)

    return tuple(timestamps)


def _validate_timestamps(name: str, values: Sequence[int]) -> tuple[int, ...]:
    timestamps = tuple(int(value) for value in values)
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} timestamps must be strictly increasing")
    return timestamps


def pair_stereo_timestamps(
    left_timestamps_us: Sequence[int],
    right_timestamps_us: Sequence[int],
    *,
    max_difference_us: int | None = None,
) -> StereoPairingReport:
    """Associate sorted camera timestamps one-to-one by nearest capture time.

    Pairing is monotonic, as video frames cannot change order.  At each step a
    frame is skipped if the next frame on that side is closer to the current
    frame on the other side; otherwise the mutual-nearest frames are paired.
    This handles a dropped frame without shifting all subsequent associations.

    ``max_difference_us`` can reject associations outside a known hardware
    synchronization tolerance.  Rejected frames are reported, never hidden.
    """

    if max_difference_us is not None and max_difference_us < 0:
        raise ValueError("max_difference_us must be non-negative or None")

    left = _validate_timestamps("left", left_timestamps_us)
    right = _validate_timestamps("right", right_timestamps_us)
    pairs: list[StereoTimestampPair] = []
    unmatched_left: list[int] = []
    unmatched_right: list[int] = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        left_time = left[left_index]
        right_time = right[right_index]
        difference = abs(left_time - right_time)

        left_improvement: int | None = None
        if left_index + 1 < len(left):
            next_difference = abs(left[left_index + 1] - right_time)
            if next_difference < difference:
                left_improvement = difference - next_difference

        right_improvement: int | None = None
        if right_index + 1 < len(right):
            next_difference = abs(left_time - right[right_index + 1])
            if next_difference < difference:
                right_improvement = difference - next_difference

        if left_improvement is not None or right_improvement is not None:
            # If both next frames are closer, discard the side with the larger
            # reduction.  A tie is resolved by discarding the earlier sample.
            if right_improvement is None or (
                left_improvement is not None
                and (
                    left_improvement > right_improvement
                    or (
                        left_improvement == right_improvement
                        and left_time <= right_time
                    )
                )
            ):
                unmatched_left.append(left_index)
                left_index += 1
            else:
                unmatched_right.append(right_index)
                right_index += 1
            continue

        if max_difference_us is not None and difference > max_difference_us:
            if left_time < right_time:
                unmatched_left.append(left_index)
                left_index += 1
            else:
                unmatched_right.append(right_index)
                right_index += 1
            continue

        pairs.append(
            StereoTimestampPair(
                left_index=left_index,
                right_index=right_index,
                left_timestamp_us=left_time,
                right_timestamp_us=right_time,
            )
        )
        left_index += 1
        right_index += 1

    unmatched_left.extend(range(left_index, len(left)))
    unmatched_right.extend(range(right_index, len(right)))
    return StereoPairingReport(
        pairs=tuple(pairs),
        unmatched_left_indices=tuple(unmatched_left),
        unmatched_right_indices=tuple(unmatched_right),
    )


def pair_timestamp_csvs(
    left_csv: str | Path,
    right_csv: str | Path,
    *,
    max_difference_us: int | None = None,
) -> StereoPairingReport:
    """Read two timestamp CSV files and return their pairing report."""

    return pair_stereo_timestamps(
        read_timestamp_csv(left_csv),
        read_timestamp_csv(right_csv),
        max_difference_us=max_difference_us,
    )


def _normalise_color_order(color_order: str) -> ColorOrder:
    normalised = color_order.upper()
    if normalised not in ("BGR", "RGB"):
        raise ValueError("color_order must be 'BGR' or 'RGB'")
    return normalised  # type: ignore[return-value]


def decode_frames_by_index(
    video_path: str | Path,
    frame_indices: Iterable[int],
    *,
    color_order: str = "BGR",
    expected_size: tuple[int, int] | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Decode selected video frames sequentially with PyAV.

    ``frame_indices`` are zero-based *decoded display-frame* indices.  H.264 is
    decoded from the start so B-frame reordering and GOP keyframes cannot make
    a requested index ambiguous.  Indices must be strictly increasing.

    ``expected_size`` is ``(width, height)``.  Pass ``(1600, 1300)`` to enforce
    the native EGO stream contract.
    """

    requested = tuple(int(index) for index in frame_indices)
    if any(index < 0 for index in requested):
        raise ValueError("frame indices must be non-negative")
    if any(current <= previous for previous, current in zip(requested, requested[1:])):
        raise ValueError("frame indices must be strictly increasing")
    if not requested:
        return

    order = _normalise_color_order(color_order)
    pixel_format = "bgr24" if order == "BGR" else "rgb24"

    try:
        import av
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyAV is required for video decoding; install the 'vision' extras"
        ) from error

    target_position = 0
    last_decoded_index = -1
    with av.open(str(Path(video_path))) as container:
        if not container.streams.video:
            raise ValueError(f"{video_path} contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for decoded_index, frame in enumerate(container.decode(stream)):
            last_decoded_index = decoded_index
            if decoded_index != requested[target_position]:
                continue

            image = frame.to_ndarray(format=pixel_format)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"decoded frame {decoded_index} has unexpected image layout "
                    f"{image.shape}, {image.dtype}"
                )
            if expected_size is not None:
                expected_width, expected_height = expected_size
                if image.shape[:2] != (expected_height, expected_width):
                    raise ValueError(
                        f"decoded frame {decoded_index} is "
                        f"{image.shape[1]}x{image.shape[0]}, expected "
                        f"{expected_width}x{expected_height}"
                    )

            yield DecodedVideoFrame(index=decoded_index, image=image)
            target_position += 1
            if target_position == len(requested):
                return

    missing = requested[target_position:]
    raise IndexError(
        f"video ended after decoded frame {last_decoded_index}; "
        f"requested frame indices not found: {missing}"
    )


def iter_stereo_frames(
    left_video: str | Path,
    right_video: str | Path,
    pairing: StereoPairingReport | Sequence[StereoTimestampPair],
    *,
    color_order: str = "BGR",
    expected_size: tuple[int, int] | None = (1600, 1300),
) -> Iterator[StereoFrame]:
    """Decode associated left/right frames and yield synchronized images."""

    order = _normalise_color_order(color_order)
    pairs = pairing.pairs if isinstance(pairing, StereoPairingReport) else tuple(pairing)
    left_frames = decode_frames_by_index(
        left_video,
        (pair.left_index for pair in pairs),
        color_order=order,
        expected_size=expected_size,
    )
    right_frames = decode_frames_by_index(
        right_video,
        (pair.right_index for pair in pairs),
        color_order=order,
        expected_size=expected_size,
    )

    for pair, left, right in zip(pairs, left_frames, right_frames, strict=True):
        if left.index != pair.left_index or right.index != pair.right_index:
            raise RuntimeError("decoder returned a frame with the wrong index")
        yield StereoFrame(
            timestamps=pair,
            left=left.image,
            right=right.image,
            color_order=order,
        )
