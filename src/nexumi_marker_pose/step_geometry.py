"""Read STEP geometry and describe its planar triangular faces.

The STEP files used by this project are authored in millimetres.  OpenCascade's
STEP reader converts imported geometry to its configured length unit (millimetres
for the pythonocc distribution), so all dimensional fields exposed here are
explicitly named ``*_mm``/``*_mm2``.

OpenCascade is imported lazily.  This keeps lightweight consumers (for example,
JSON schema tooling) usable even when they are not running inside the conda
environment that contains ``pythonocc-core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, sqrt
from pathlib import Path
from typing import Any


Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TriangleFace:
    """A planar triangular STEP face with metric and topological properties."""

    face_index: int
    vertex_indices: tuple[int, int, int]
    vertices_mm: tuple[Vector3, Vector3, Vector3]
    edge_lengths_mm: tuple[float, float, float]
    area_mm2: float
    centroid_mm: Vector3
    normal: Vector3
    actual_face_area_mm2: float | None = None
    outline_triangle_area_mm2: float | None = None
    has_inner_wires: bool = False
    inner_wire_count: int = 0
    outer_wire_vertex_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Return a structure accepted directly by :func:`json.dumps`."""

        actual_area = (
            self.area_mm2
            if self.actual_face_area_mm2 is None
            else self.actual_face_area_mm2
        )
        outline_area = (
            self.area_mm2
            if self.outline_triangle_area_mm2 is None
            else self.outline_triangle_area_mm2
        )
        return {
            "face_index": self.face_index,
            "vertex_indices": list(self.vertex_indices),
            "vertices_mm": [list(point) for point in self.vertices_mm],
            "edge_lengths_mm": list(self.edge_lengths_mm),
            "area_mm2": self.area_mm2,
            "centroid_mm": list(self.centroid_mm),
            "normal": list(self.normal),
            "actual_face_area_mm2": actual_area,
            "outline_triangle_area_mm2": outline_area,
            "has_inner_wires": self.has_inner_wires,
            "inner_wire_count": self.inner_wire_count,
            "outer_wire_vertex_count": self.outer_wire_vertex_count,
        }


@dataclass(frozen=True, slots=True)
class StepGeometry:
    """JSON-friendly summary of the triangular marker geometry in a STEP file."""

    source_path: str
    units: str
    total_face_count: int
    planar_face_count: int
    vertices_mm: tuple[Vector3, ...]
    triangular_faces: tuple[TriangleFace, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with no OCC objects."""

        return {
            "source_path": self.source_path,
            "units": self.units,
            "total_face_count": self.total_face_count,
            "planar_face_count": self.planar_face_count,
            "vertex_count": len(self.vertices_mm),
            "triangular_face_count": len(self.triangular_faces),
            "vertices_mm": [list(point) for point in self.vertices_mm],
            "triangular_faces": [face.to_dict() for face in self.triangular_faces],
        }


def _occ_api() -> dict[str, Any]:
    """Import and return the small pythonocc API surface used by this module."""

    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core import BRepGProp
        from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
        from OCC.Core.GeomAbs import GeomAbs_Plane
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
    except ImportError as exc:  # pragma: no cover - exercised outside OCC env
        raise RuntimeError(
            "pythonocc-core is required to read STEP geometry; activate the "
            "nexumi-right-hand-pose conda environment"
        ) from exc

    return {
        "BRep_Tool": BRep_Tool,
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "BRepGProp": BRepGProp,
        "BRepTools_WireExplorer": BRepTools_WireExplorer,
        "breptools": breptools,
        "GeomAbs_Plane": GeomAbs_Plane,
        "GProp_GProps": GProp_GProps,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_REVERSED": TopAbs_REVERSED,
        "TopAbs_WIRE": TopAbs_WIRE,
        "TopExp_Explorer": TopExp_Explorer,
        "topods": topods,
    }


def _surface_properties(api: dict[str, Any], face: Any, props: Any) -> None:
    """Call the BRepGProp API across pythonocc wrapper naming variants."""

    module = api["BRepGProp"]
    if hasattr(module, "brepgprop"):
        module.brepgprop.SurfaceProperties(face, props)
    else:  # pythonocc 7.9 exposes the generated free function on some builds
        module.brepgprop_SurfaceProperties(face, props)


def _point_tuple(point: Any) -> Vector3:
    return (float(point.X()), float(point.Y()), float(point.Z()))


def _outer_wire_vertices(
    api: dict[str, Any], face: Any, tolerance: float
) -> tuple[list[Vector3], int]:
    """Return the outer-wire vertices in boundary order and the inner-wire count."""

    outer_wire = api["breptools"].OuterWire(face)
    explorer = api["BRepTools_WireExplorer"](outer_wire, face)
    points: list[Vector3] = []
    while explorer.More():
        vertex = explorer.CurrentVertex()
        point = _point_tuple(api["BRep_Tool"].Pnt(vertex))
        if not points or dist(point, points[-1]) > tolerance:
            points.append(point)
        explorer.Next()
    if len(points) > 1 and dist(points[0], points[-1]) <= tolerance:
        points.pop()

    wire_count = 0
    wire_explorer = api["TopExp_Explorer"](face, api["TopAbs_WIRE"])
    while wire_explorer.More():
        wire_count += 1
        wire_explorer.Next()
    return points, max(0, wire_count - 1)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: Vector3) -> float:
    return sqrt(sum(component * component for component in vector))


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _simplify_collinear_outline(
    points: list[Vector3], tolerance_mm: float
) -> list[Vector3]:
    """Remove boundary subdivision points lying on the same straight segment.

    A point is removed only when it lies between its neighbours (the two edge
    directions have a non-negative dot product) and its perpendicular distance
    to their chord is within ``tolerance_mm``.  Repeating the operation handles
    any number of consecutive subdivisions while preserving real polygon corners.
    """

    simplified = list(points)
    while len(simplified) > 3:
        removable: list[tuple[float, int]] = []
        for index, point in enumerate(simplified):
            previous = simplified[index - 1]
            following = simplified[(index + 1) % len(simplified)]
            incoming = _subtract(point, previous)
            outgoing = _subtract(following, point)
            chord = _subtract(following, previous)
            chord_length = _norm(chord)
            if chord_length == 0.0:
                continue
            same_direction = sum(a * b for a, b in zip(incoming, outgoing)) >= 0.0
            deviation = _norm(_cross(incoming, outgoing)) / chord_length
            if same_direction and deviation <= tolerance_mm:
                removable.append((deviation, index))
        if not removable:
            break
        _, index = min(removable)
        simplified.pop(index)
    return simplified


def _triangle_area(points: list[Vector3]) -> float:
    return 0.5 * _norm(
        _cross(_subtract(points[1], points[0]), _subtract(points[2], points[0]))
    )


def _global_vertex_index(
    vertices: list[Vector3], point: Vector3, tolerance: float
) -> int:
    for index, existing in enumerate(vertices):
        if dist(point, existing) <= tolerance:
            return index
    vertices.append(point)
    return len(vertices) - 1


def read_step_geometry(
    step_path: str | Path,
    *,
    vertex_tolerance_mm: float = 1.0e-6,
    outline_simplification_tolerance_mm: float = 1.0e-3,
) -> StepGeometry:
    """Read *step_path* and extract every planar face with a triangular outline.

    Only the outer wire defines the candidate outline.  Collinear edge subdivision
    vertices are collapsed, and any inner wires are reported as diagnostics rather
    than being mistaken for triangle corners.

    Args:
        step_path: STEP/STP file to import.
        vertex_tolerance_mm: Distance used to merge numerically identical
            vertices, both within one face and across different faces.
        outline_simplification_tolerance_mm: Maximum perpendicular deviation of
            a segmented outer-wire vertex from a straight edge.  The default is
            one micron in the millimetre-scale CAD model.

    Returns:
        A :class:`StepGeometry` containing only Python scalar/container types.

    Raises:
        FileNotFoundError: if *step_path* does not exist.
        ValueError: for a non-positive tolerance or a STEP import failure.
        RuntimeError: if pythonocc-core is unavailable.
    """

    path = Path(step_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if vertex_tolerance_mm <= 0:
        raise ValueError("vertex_tolerance_mm must be positive")
    if outline_simplification_tolerance_mm <= 0:
        raise ValueError("outline_simplification_tolerance_mm must be positive")

    api = _occ_api()
    reader = api["STEPControl_Reader"]()
    status = reader.ReadFile(str(path))
    if status != api["IFSelect_RetDone"]:
        raise ValueError(f"OpenCascade could not read STEP file: {path}")
    if reader.TransferRoots() == 0:
        raise ValueError(f"STEP file contains no transferable roots: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP import produced a null shape: {path}")

    total_face_count = 0
    planar_face_count = 0
    global_vertices: list[Vector3] = []
    triangles: list[TriangleFace] = []

    explorer = api["TopExp_Explorer"](shape, api["TopAbs_FACE"])
    while explorer.More():
        face_index = total_face_count
        total_face_count += 1
        face = api["topods"].Face(explorer.Current())
        surface = api["BRepAdaptor_Surface"](face, True)
        if surface.GetType() != api["GeomAbs_Plane"]:
            explorer.Next()
            continue
        planar_face_count += 1

        outer_points, inner_wire_count = _outer_wire_vertices(
            api, face, vertex_tolerance_mm
        )
        points = _simplify_collinear_outline(
            outer_points, outline_simplification_tolerance_mm
        )
        if len(points) != 3:
            explorer.Next()
            continue

        props = api["GProp_GProps"]()
        _surface_properties(api, face, props)
        area = float(props.Mass())
        outline_area = _triangle_area(points)
        centroid = _point_tuple(props.CentreOfMass())

        direction = surface.Plane().Axis().Direction()
        normal = _point_tuple(direction)
        if face.Orientation() == api["TopAbs_REVERSED"]:
            normal = tuple(-component for component in normal)  # type: ignore[assignment]

        vertex_indices = tuple(
            _global_vertex_index(global_vertices, point, vertex_tolerance_mm)
            for point in points
        )
        edge_lengths = tuple(
            sorted(
                (
                    dist(points[0], points[1]),
                    dist(points[1], points[2]),
                    dist(points[2], points[0]),
                )
            )
        )

        triangles.append(
            TriangleFace(
                face_index=face_index,
                vertex_indices=vertex_indices,  # type: ignore[arg-type]
                vertices_mm=tuple(points),  # type: ignore[arg-type]
                edge_lengths_mm=edge_lengths,
                area_mm2=area,
                centroid_mm=centroid,
                normal=normal,
                actual_face_area_mm2=area,
                outline_triangle_area_mm2=outline_area,
                has_inner_wires=inner_wire_count > 0,
                inner_wire_count=inner_wire_count,
                outer_wire_vertex_count=len(outer_points),
            )
        )
        explorer.Next()

    return StepGeometry(
        source_path=str(path),
        units="mm",
        total_face_count=total_face_count,
        planar_face_count=planar_face_count,
        vertices_mm=tuple(global_vertices),
        triangular_faces=tuple(triangles),
    )


def extract_step_geometry(
    step_path: str | Path,
    *,
    vertex_tolerance_mm: float = 1.0e-6,
    outline_simplification_tolerance_mm: float = 1.0e-3,
) -> dict[str, Any]:
    """Convenience API returning the STEP summary directly as a dictionary."""

    return read_step_geometry(
        step_path,
        vertex_tolerance_mm=vertex_tolerance_mm,
        outline_simplification_tolerance_mm=outline_simplification_tolerance_mm,
    ).to_dict()


__all__ = [
    "StepGeometry",
    "TriangleFace",
    "extract_step_geometry",
    "read_step_geometry",
]
