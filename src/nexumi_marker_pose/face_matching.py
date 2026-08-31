"""One-to-one matching between printed stickers and triangular STEP faces.

The PDF dimensions and STEP geometry describe the same physical triangles, but
neither source guarantees an edge order.  Matching therefore uses the sorted
three-edge signature of each triangle.  A global assignment is used instead of
independent nearest-neighbour choices so that one STEP face cannot be assigned
to two stickers.

All public dimensions in this module are millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite, sqrt
from typing import Any, Hashable, Iterable, Mapping, Sequence


FaceId = Hashable
Sides = tuple[float, float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class StepTriangle:
    """A triangular STEP face.

    Prefer ``vertices_mm`` when available because the edge lengths are then
    derived from the exact vertices.  ``side_lengths_mm`` is useful for a STEP
    extractor that has already computed the three boundary-edge lengths.
    Supplying both is allowed and checked for consistency by ``resolved_sides``.
    """

    face_id: FaceId
    vertices_mm: tuple[Point3, Point3, Point3] | None = None
    side_lengths_mm: Sides | None = None

    @classmethod
    def from_extracted_face(cls, face: Any) -> "StepTriangle":
        """Adapt :class:`step_geometry.TriangleFace` without importing OCC.

        A mapping with the corresponding JSON field names is accepted too,
        which makes it possible to match a previously exported geometry file.
        """

        if isinstance(face, Mapping):
            try:
                face_id = face["face_index"]
                vertices = face.get("vertices_mm")
                sides = face.get("edge_lengths_mm")
            except KeyError as exc:
                raise ValueError("extracted STEP face is missing face_index") from exc
        else:
            try:
                face_id = face.face_index
                vertices = getattr(face, "vertices_mm", None)
                sides = getattr(face, "edge_lengths_mm", None)
            except AttributeError as exc:
                raise TypeError(
                    "STEP faces must be StepTriangle, TriangleFace-like objects, "
                    "or exported face mappings"
                ) from exc
        return cls(
            face_id=face_id,
            vertices_mm=(
                tuple(tuple(float(value) for value in point) for point in vertices)
                if vertices is not None
                else None
            ),  # type: ignore[arg-type]
            side_lengths_mm=(
                tuple(float(value) for value in sides) if sides is not None else None
            ),  # type: ignore[arg-type]
        )

    def resolved_sides(self, consistency_tolerance_mm: float = 1e-6) -> Sides:
        from_vertices = (
            triangle_side_lengths(self.vertices_mm)
            if self.vertices_mm is not None
            else None
        )
        from_edges = (
            _validate_sides(self.side_lengths_mm, f"STEP face {self.face_id!r}")
            if self.side_lengths_mm is not None
            else None
        )
        if from_vertices is None and from_edges is None:
            raise ValueError(
                f"STEP face {self.face_id!r} needs vertices_mm or side_lengths_mm"
            )
        if from_vertices is not None and from_edges is not None:
            residuals = _edge_residuals(from_edges, from_vertices)
            if max(map(abs, residuals)) > consistency_tolerance_mm:
                raise ValueError(
                    f"STEP face {self.face_id!r} vertices and edge lengths disagree: "
                    f"{residuals!r} mm"
                )
        return from_vertices if from_vertices is not None else from_edges  # type: ignore[return-value]


@dataclass(frozen=True)
class CandidateScore:
    """Geometric error for one sticker/STEP-face pair."""

    sticker_id: str
    step_face_id: FaceId
    sticker_sides_sorted_mm: Sides
    step_sides_sorted_mm: Sides
    residuals_mm: Sides
    rmse_mm: float
    max_abs_error_mm: float
    within_tolerance: bool


@dataclass(frozen=True)
class FaceMatch:
    """Chosen match, including the complete three-edge error."""

    sticker_id: str
    step_face_id: FaceId
    sticker_sides_sorted_mm: Sides
    step_sides_sorted_mm: Sides
    residuals_mm: Sides
    rmse_mm: float
    max_abs_error_mm: float


@dataclass(frozen=True)
class AmbiguityDiagnostic:
    """A sticker whose identity is not well separated geometrically."""

    sticker_id: str
    chosen_step_face_id: FaceId | None
    reason: str
    alternatives: tuple[CandidateScore, ...]


@dataclass(frozen=True)
class FaceMatchingResult:
    """Result of the global one-to-one assignment."""

    matches: tuple[FaceMatch, ...]
    unmatched_sticker_ids: tuple[str, ...]
    unused_step_face_ids: tuple[FaceId, ...]
    ambiguities: tuple[AmbiguityDiagnostic, ...]
    tolerance_mm: float
    ambiguity_margin_mm: float
    global_rmse_mm: float | None
    second_best_global_rmse_mm: float | None

    @property
    def complete(self) -> bool:
        return not self.unmatched_sticker_ids

    @property
    def globally_ambiguous(self) -> bool:
        if (
            self.global_rmse_mm is None
            or self.second_best_global_rmse_mm is None
        ):
            return False
        return (
            self.second_best_global_rmse_mm - self.global_rmse_mm
            <= self.ambiguity_margin_mm
        )

    @property
    def assignment_by_sticker(self) -> dict[str, FaceId]:
        return {match.sticker_id: match.step_face_id for match in self.matches}


def triangle_side_lengths(vertices_mm: Sequence[Sequence[float]]) -> Sides:
    """Return lengths of edges (v0-v1, v1-v2, v2-v0), in millimetres."""

    if len(vertices_mm) != 3:
        raise ValueError(f"a triangle needs exactly 3 vertices, got {len(vertices_mm)}")
    points: list[Point3] = []
    for vertex in vertices_mm:
        if len(vertex) != 3:
            raise ValueError("every STEP vertex must contain exactly 3 coordinates")
        point = tuple(float(value) for value in vertex)
        if not all(isfinite(value) for value in point):
            raise ValueError(f"non-finite STEP vertex: {vertex!r}")
        points.append(point)  # type: ignore[arg-type]
    sides = (
        dist(points[0], points[1]),
        dist(points[1], points[2]),
        dist(points[2], points[0]),
    )
    return _validate_sides(sides, "STEP vertices")


def score_candidate(
    sticker_id: str,
    sticker_sides_mm: Sequence[float],
    step_triangle: StepTriangle,
    *,
    tolerance_mm: float = 0.25,
) -> CandidateScore:
    """Score one pair after resolving the unknown edge ordering."""

    if tolerance_mm <= 0 or not isfinite(tolerance_mm):
        raise ValueError("tolerance_mm must be finite and positive")
    sticker = tuple(sorted(_validate_sides(sticker_sides_mm, sticker_id)))
    step = tuple(sorted(step_triangle.resolved_sides()))
    residuals = tuple(b - a for a, b in zip(sticker, step))
    rmse = sqrt(sum(value * value for value in residuals) / 3.0)
    max_abs = max(map(abs, residuals))
    return CandidateScore(
        sticker_id=sticker_id,
        step_face_id=step_triangle.face_id,
        sticker_sides_sorted_mm=sticker,  # type: ignore[arg-type]
        step_sides_sorted_mm=step,  # type: ignore[arg-type]
        residuals_mm=residuals,  # type: ignore[arg-type]
        rmse_mm=rmse,
        max_abs_error_mm=max_abs,
        within_tolerance=max_abs <= tolerance_mm,
    )


def match_triangular_faces(
    sticker_sides_mm: Mapping[str, Sequence[float]],
    step_triangles: Iterable[StepTriangle | Any],
    *,
    tolerance_mm: float = 0.25,
    ambiguity_margin_mm: float = 0.10,
) -> FaceMatchingResult:
    """Globally match printed stickers to STEP triangles one-to-one.

    ``tolerance_mm=0.25`` is deliberately a little wider than the stated
    0.1 mm STEP accuracy and the PDF's 0.1 mm dimension rounding.  A candidate
    is feasible only when *all three* sorted edge residuals fit the tolerance.

    Ambiguity is reported in two complementary ways:

    * local: another face for the same sticker is within ``ambiguity_margin_mm``
      RMSE of the chosen face;
    * global: the best complete assignment obtained by forbidding one chosen
      pair is within that margin of the optimum.

    Incomplete data returns the largest useful partial assignment via dummy
    columns and lists the unmatched sticker IDs; it does not silently force a
    geometrically invalid pair.
    """

    if tolerance_mm <= 0 or not isfinite(tolerance_mm):
        raise ValueError("tolerance_mm must be finite and positive")
    if ambiguity_margin_mm < 0 or not isfinite(ambiguity_margin_mm):
        raise ValueError("ambiguity_margin_mm must be finite and non-negative")
    if not sticker_sides_mm:
        raise ValueError("at least one sticker triangle is required")

    sticker_ids = list(sticker_sides_mm)
    if len(sticker_ids) != len(set(sticker_ids)):
        raise ValueError("sticker IDs must be unique")
    validated_stickers = {
        sticker_id: _validate_sides(sides, f"sticker {sticker_id!r}")
        for sticker_id, sides in sticker_sides_mm.items()
    }

    faces = [
        face if isinstance(face, StepTriangle) else StepTriangle.from_extracted_face(face)
        for face in step_triangles
    ]
    face_ids = [face.face_id for face in faces]
    if len(face_ids) != len(set(face_ids)):
        raise ValueError("STEP face IDs must be unique")

    scores: list[list[CandidateScore]] = [
        [
            score_candidate(
                sticker_id,
                validated_stickers[sticker_id],
                face,
                tolerance_mm=tolerance_mm,
            )
            for face in faces
        ]
        for sticker_id in sticker_ids
    ]
    feasible_costs = [
        [score.rmse_mm**2 if score.within_tolerance else None for score in row]
        for row in scores
    ]

    assignment = _solve_partial_assignment(feasible_costs, tolerance_mm)
    matches: list[FaceMatch] = []
    used_face_indices: set[int] = set()
    unmatched: list[str] = []
    for sticker_index, face_index in enumerate(assignment):
        if face_index is None:
            unmatched.append(sticker_ids[sticker_index])
            continue
        used_face_indices.add(face_index)
        score = scores[sticker_index][face_index]
        matches.append(
            FaceMatch(
                sticker_id=score.sticker_id,
                step_face_id=score.step_face_id,
                sticker_sides_sorted_mm=score.sticker_sides_sorted_mm,
                step_sides_sorted_mm=score.step_sides_sorted_mm,
                residuals_mm=score.residuals_mm,
                rmse_mm=score.rmse_mm,
                max_abs_error_mm=score.max_abs_error_mm,
            )
        )

    global_rmse = _assignment_global_rmse(assignment, feasible_costs)
    second_best = None
    second_assignment: list[int | None] | None = None
    if not unmatched:
        second_best, second_assignment = _second_best_complete_assignment(
            feasible_costs, assignment
        )

    ambiguities: list[AmbiguityDiagnostic] = []
    chosen_by_sticker_index = {
        index: face_index
        for index, face_index in enumerate(assignment)
        if face_index is not None
    }
    for sticker_index, sticker_id in enumerate(sticker_ids):
        chosen_index = chosen_by_sticker_index.get(sticker_index)
        feasible = sorted(
            (score for score in scores[sticker_index] if score.within_tolerance),
            key=lambda score: (score.rmse_mm, str(score.step_face_id)),
        )
        if chosen_index is None:
            reason = (
                "no STEP face passes the three-edge tolerance"
                if not feasible
                else "feasible faces exist, but one-to-one constraints consume them"
            )
            ambiguities.append(
                AmbiguityDiagnostic(sticker_id, None, reason, tuple(feasible[:5]))
            )
            continue

        chosen_score = scores[sticker_index][chosen_index]
        near = tuple(
            candidate
            for candidate in feasible
            if candidate.step_face_id != chosen_score.step_face_id
            and candidate.rmse_mm - chosen_score.rmse_mm <= ambiguity_margin_mm
        )
        if near:
            ambiguities.append(
                AmbiguityDiagnostic(
                    sticker_id,
                    chosen_score.step_face_id,
                    "multiple local candidates have nearly identical edge errors",
                    near,
                )
            )

    if (
        global_rmse is not None
        and second_best is not None
        and second_best - global_rmse <= ambiguity_margin_mm
        and second_assignment is not None
    ):
        changed_rows = [
            index
            for index, (first, second) in enumerate(zip(assignment, second_assignment))
            if first != second
        ]
        for sticker_index in changed_rows:
            sticker_id = sticker_ids[sticker_index]
            if any(item.sticker_id == sticker_id for item in ambiguities):
                continue
            alternative_index = second_assignment[sticker_index]
            alternatives = (
                (scores[sticker_index][alternative_index],)
                if alternative_index is not None
                else ()
            )
            chosen_index = assignment[sticker_index]
            ambiguities.append(
                AmbiguityDiagnostic(
                    sticker_id=sticker_id,
                    chosen_step_face_id=(
                        faces[chosen_index].face_id if chosen_index is not None else None
                    ),
                    reason="a near-equal complete one-to-one assignment changes this pair",
                    alternatives=alternatives,
                )
            )

    return FaceMatchingResult(
        matches=tuple(matches),
        unmatched_sticker_ids=tuple(unmatched),
        unused_step_face_ids=tuple(
            face.face_id
            for index, face in enumerate(faces)
            if index not in used_face_indices
        ),
        ambiguities=tuple(ambiguities),
        tolerance_mm=tolerance_mm,
        ambiguity_margin_mm=ambiguity_margin_mm,
        global_rmse_mm=global_rmse,
        second_best_global_rmse_mm=second_best,
    )


# A descriptive alias for callers that think in sticker rather than face terms.
match_stickers_to_step_faces = match_triangular_faces


def _validate_sides(sides: Sequence[float], source: str) -> Sides:
    if len(sides) != 3:
        raise ValueError(f"{source} needs exactly 3 side lengths, got {len(sides)}")
    values = tuple(float(value) for value in sides)
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError(f"{source} has non-positive or non-finite sides: {sides!r}")
    short, middle, long = sorted(values)
    if short + middle <= long:
        raise ValueError(f"{source} violates the triangle inequality: {sides!r}")
    return values  # type: ignore[return-value]


def _edge_residuals(first: Sequence[float], second: Sequence[float]) -> Sides:
    a = sorted(first)
    b = sorted(second)
    return tuple(y - x for x, y in zip(a, b))  # type: ignore[return-value]


def _solve_partial_assignment(
    costs: Sequence[Sequence[float | None]], tolerance_mm: float
) -> list[int | None]:
    """Solve maximum-cardinality, minimum-cost matching using dummy columns."""

    row_count = len(costs)
    real_column_count = len(costs[0]) if row_count else 0
    # A dummy must be preferable to any forbidden real edge, yet vastly more
    # expensive than every valid assignment.  Hence minimisation first maximises
    # match cardinality, then minimises geometric error.
    unmatched_penalty = (tolerance_mm * tolerance_mm + 1.0) * (row_count + 1)
    forbidden_penalty = unmatched_penalty * (row_count + 1)
    matrix = []
    for row in costs:
        real = [
            value if value is not None else forbidden_penalty
            for value in row
        ]
        matrix.append(real + [unmatched_penalty] * row_count)
    raw = _hungarian(matrix)
    return [
        column
        if column < real_column_count and costs[row][column] is not None
        else None
        for row, column in enumerate(raw)
    ]


def _solve_complete_assignment(
    costs: Sequence[Sequence[float | None]],
) -> list[int] | None:
    row_count = len(costs)
    column_count = len(costs[0]) if row_count else 0
    if row_count > column_count:
        return None
    finite_values = [
        value for row in costs for value in row if value is not None
    ]
    if not finite_values:
        return None
    forbidden = (max(finite_values) + 1.0) * (row_count + 1)
    raw = _hungarian(
        [
            [value if value is not None else forbidden for value in row]
            for row in costs
        ]
    )
    if any(costs[row][column] is None for row, column in enumerate(raw)):
        return None
    return raw


def _second_best_complete_assignment(
    costs: Sequence[Sequence[float | None]],
    best: Sequence[int | None],
) -> tuple[float | None, list[int | None] | None]:
    best_alternative_rmse: float | None = None
    best_alternative: list[int | None] | None = None
    for row, column in enumerate(best):
        if column is None:
            return None, None
        modified = [list(values) for values in costs]
        modified[row][column] = None
        alternative = _solve_complete_assignment(modified)
        if alternative is None:
            continue
        alternative_optional: list[int | None] = list(alternative)
        rmse = _assignment_global_rmse(alternative_optional, costs)
        if rmse is not None and (
            best_alternative_rmse is None or rmse < best_alternative_rmse
        ):
            best_alternative_rmse = rmse
            best_alternative = alternative_optional
    return best_alternative_rmse, best_alternative


def _assignment_global_rmse(
    assignment: Sequence[int | None],
    costs: Sequence[Sequence[float | None]],
) -> float | None:
    if not assignment or any(column is None for column in assignment):
        return None
    squared_rmse_sum = 0.0
    for row, column in enumerate(assignment):
        value = costs[row][column]  # type: ignore[index]
        if value is None:
            return None
        squared_rmse_sum += value
    return sqrt(squared_rmse_sum / len(assignment))


def _hungarian(costs: Sequence[Sequence[float]]) -> list[int]:
    """Rectangular Hungarian assignment for rows <= columns.

    Returns one distinct column index for each row.  This is the standard
    potential-based O(rows^2 * columns) algorithm, kept local so matching does
    not require SciPy during the geometry-only project phase.
    """

    row_count = len(costs)
    if row_count == 0:
        return []
    column_count = len(costs[0])
    if column_count < row_count:
        raise ValueError("Hungarian solver requires at least as many columns as rows")
    if any(len(row) != column_count for row in costs):
        raise ValueError("assignment cost matrix must be rectangular")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for current_row in range(1, row_count + 1):
        matched_row[0] = current_row
        min_value = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        current_column = 0
        while True:
            used[current_column] = True
            row = matched_row[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = costs[row - 1][column - 1] - u[row] - v[column]
                if reduced < min_value[column]:
                    min_value[column] = reduced
                    predecessor[column] = current_column
                if min_value[column] < delta:
                    delta = min_value[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous = predecessor[current_column]
            matched_row[current_column] = matched_row[previous]
            current_column = previous
            if current_column == 0:
                break

    result = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            result[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in result):
        raise RuntimeError("Hungarian solver produced an incomplete assignment")
    return result
