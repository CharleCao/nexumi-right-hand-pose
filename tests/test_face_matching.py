from __future__ import annotations

from math import sqrt

import pytest

from nexumi_marker_pose.face_matching import (
    StepTriangle,
    match_triangular_faces,
    score_candidate,
    triangle_side_lengths,
)


RIGHT_HAND_PDF_SIDES_MM = {
    "F0": (55.2, 69.7, 83.1),
    "F1": (66.5, 81.2, 39.2),
    "F2": (49.5, 63.4, 67.6),
    "F3": (40.1, 60.5, 37.1),
    "F4": (40.4, 37.8, 39.1),
    "F5": (62.6, 53.8, 40.0),
    "F6": (61.5, 44.1, 38.1),
    "F7": (37.9, 37.7, 39.9),
    "F8": (68.6, 41.2, 73.9),
    "F9": (37.1, 48.4, 71.2),
}


def test_triangle_side_lengths_from_vertices() -> None:
    assert triangle_side_lengths(((0, 0, 0), (3, 0, 0), (0, 4, 0))) == (3, 5, 4)


def test_candidate_score_is_independent_of_edge_order() -> None:
    candidate = score_candidate(
        "F0",
        (55.2, 69.7, 83.1),
        StepTriangle("face-8", side_lengths_mm=(83.0, 55.3, 69.7)),
        tolerance_mm=0.2,
    )
    assert candidate.within_tolerance
    assert candidate.residuals_mm == pytest.approx((0.1, 0.0, -0.1))
    assert candidate.rmse_mm == pytest.approx(sqrt(0.02 / 3.0))
    assert candidate.max_abs_error_mm == pytest.approx(0.1)


def test_matches_all_right_hand_pdf_faces_one_to_one_despite_shuffling() -> None:
    # Synthetic STEP output: the test exercises exact F0-F9 dimensions while
    # perturbing every edge below the stated 0.1 mm model accuracy.
    reversed_ids = list(reversed(RIGHT_HAND_PDF_SIDES_MM))
    step_faces = []
    expected = {}
    perturbations = (-0.08, 0.03, 0.06)
    for index, sticker_id in enumerate(reversed_ids):
        sides = RIGHT_HAND_PDF_SIDES_MM[sticker_id]
        permuted = (
            sides[2] + perturbations[0],
            sides[0] + perturbations[1],
            sides[1] + perturbations[2],
        )
        face_id = 100 + index
        step_faces.append(StepTriangle(face_id, side_lengths_mm=permuted))
        expected[sticker_id] = face_id

    result = match_triangular_faces(RIGHT_HAND_PDF_SIDES_MM, step_faces)

    assert result.complete
    assert result.assignment_by_sticker == expected
    assert not result.ambiguities
    assert result.global_rmse_mm is not None
    assert result.global_rmse_mm < 0.1
    assert result.unused_step_face_ids == ()


def test_accepts_step_geometry_triangle_face_objects_and_exported_dicts() -> None:
    class ExtractedFace:
        face_index = 7
        vertices_mm = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0))
        edge_lengths_mm = (3.0, 4.0, 5.0)

    from_object = match_triangular_faces({"F0": (3, 4, 5)}, [ExtractedFace()])
    from_mapping = match_triangular_faces(
        {"F0": (3, 4, 5)},
        [
            {
                "face_index": 8,
                "vertices_mm": ((0, 0, 0), (3, 0, 0), (0, 4, 0)),
                "edge_lengths_mm": (3, 4, 5),
            }
        ],
    )

    assert from_object.assignment_by_sticker == {"F0": 7}
    assert from_mapping.assignment_by_sticker == {"F0": 8}


def test_global_assignment_beats_greedy_face_reuse() -> None:
    stickers = {
        "F0": (3.00, 4.00, 5.00),
        "F1": (3.16, 4.16, 5.16),
    }
    faces = [
        StepTriangle("shared-best", side_lengths_mm=(3.08, 4.08, 5.08)),
        StepTriangle("only-f0", side_lengths_mm=(2.85, 3.85, 4.85)),
    ]

    result = match_triangular_faces(stickers, faces, tolerance_mm=0.2)

    # Independently both stickers prefer shared-best.  The global optimum uses
    # it for F1, for which the alternative is outside tolerance.
    assert result.complete
    assert result.assignment_by_sticker == {
        "F0": "only-f0",
        "F1": "shared-best",
    }


def test_reports_local_and_global_ambiguity() -> None:
    result = match_triangular_faces(
        {"F0": (3.0, 4.0, 5.0)},
        [
            StepTriangle("A", side_lengths_mm=(3.00, 4.00, 5.00)),
            StepTriangle("B", side_lengths_mm=(3.04, 4.04, 5.04)),
        ],
        tolerance_mm=0.2,
        ambiguity_margin_mm=0.05,
    )

    assert result.complete
    assert result.globally_ambiguous
    assert result.second_best_global_rmse_mm == pytest.approx(0.04)
    assert len(result.ambiguities) == 1
    assert result.ambiguities[0].alternatives[0].step_face_id == "B"


def test_does_not_force_a_face_outside_tolerance() -> None:
    result = match_triangular_faces(
        {"F0": (3.0, 4.0, 5.0), "F1": (5.0, 5.0, 6.0)},
        [StepTriangle("A", side_lengths_mm=(3.0, 4.0, 5.0))],
        tolerance_mm=0.1,
    )

    assert not result.complete
    assert result.assignment_by_sticker == {"F0": "A"}
    assert result.unmatched_sticker_ids == ("F1",)
    assert result.global_rmse_mm is None
    assert result.ambiguities[0].reason.startswith("no STEP face")


@pytest.mark.parametrize(
    "sides",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0),
        (1.0, -2.0, 2.0),
        (1.0, float("nan"), 2.0),
    ],
)
def test_rejects_invalid_triangles(sides: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        match_triangular_faces(
            {"F0": sides},
            [StepTriangle("A", side_lengths_mm=(3.0, 4.0, 5.0))],
        )
