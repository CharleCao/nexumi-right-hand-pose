from __future__ import annotations

import json
from math import sqrt

import pytest

from nexumi_marker_pose.step_geometry import (
    StepGeometry,
    TriangleFace,
    extract_step_geometry,
    read_step_geometry,
)


def test_result_types_are_json_serializable() -> None:
    face = TriangleFace(
        face_index=2,
        vertex_indices=(0, 1, 2),
        vertices_mm=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)),
        edge_lengths_mm=(3.0, 4.0, 5.0),
        area_mm2=6.0,
        centroid_mm=(1.0, 4.0 / 3.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    result = StepGeometry(
        source_path="marker.step",
        units="mm",
        total_face_count=3,
        planar_face_count=2,
        vertices_mm=face.vertices_mm,
        triangular_faces=(face,),
    )

    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["vertex_count"] == 3
    assert decoded["triangular_face_count"] == 1
    assert decoded["triangular_faces"][0]["area_mm2"] == 6.0
    assert decoded["triangular_faces"][0]["actual_face_area_mm2"] == 6.0
    assert decoded["triangular_faces"][0]["outline_triangle_area_mm2"] == 6.0
    assert decoded["triangular_faces"][0]["has_inner_wires"] is False


def _write_triangle_step(path) -> None:
    pytest.importorskip("OCC")
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCC.Core.TopoDS import TopoDS_Compound

    def make_face(a, b, c, *, split_first_edge=False, add_inner_wire=False):
        polygon = BRepBuilderAPI_MakePolygon()
        boundary = [a]
        if split_first_edge:
            # A realistic CAD edge subdivision, with a sub-micron deviation
            # from the main edge like face 54 of 上盖2.stp.
            boundary.append(
                (
                    (a[0] + b[0]) / 2,
                    (a[1] + b[1]) / 2 + 0.0001,
                    (a[2] + b[2]) / 2,
                )
            )
        boundary.extend((b, c))
        for xyz in boundary:
            polygon.Add(gp_Pnt(*xyz))
        polygon.Close()
        face_builder = BRepBuilderAPI_MakeFace(polygon.Wire())
        if add_inner_wire:
            # Clockwise in the YZ plane so this wire cuts a hole from the face.
            inner = BRepBuilderAPI_MakePolygon()
            for xyz in ((0, 0.4, 0.3), (0, 0.4, 0.7), (0, 1.0, 0.3)):
                inner.Add(gp_Pnt(*xyz))
            inner.Close()
            face_builder.Add(inner.Wire())
        return face_builder.Face()

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(
        compound,
        make_face(
            (0, 0, 0), (3, 0, 0), (0, 4, 0), split_first_edge=True
        ),
    )
    builder.Add(
        compound,
        make_face((0, 0, 0), (0, 4, 0), (0, 0, 2), add_inner_wire=True),
    )

    writer = STEPControl_Writer()
    assert writer.Transfer(compound, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_reads_planar_triangles_and_deduplicates_vertices(tmp_path) -> None:
    step_path = tmp_path / "two_triangles.step"
    _write_triangle_step(step_path)

    result = read_step_geometry(step_path)

    assert result.total_face_count == 2
    assert result.planar_face_count == 2
    assert len(result.triangular_faces) == 2
    assert len(result.vertices_mm) == 4
    assert sorted(
        face.outline_triangle_area_mm2 for face in result.triangular_faces
    ) == pytest.approx(
        [4.0, 6.0]
    )
    yz_face = min(
        result.triangular_faces, key=lambda face: face.outline_triangle_area_mm2
    )
    assert sorted(yz_face.edge_lengths_mm) == pytest.approx(
        sorted([2.0, 4.0, sqrt(20.0)])
    )
    split_face = max(
        result.triangular_faces, key=lambda face: face.outline_triangle_area_mm2
    )
    assert split_face.outer_wire_vertex_count == 4
    assert split_face.has_inner_wires is False
    assert yz_face.outer_wire_vertex_count == 3
    assert yz_face.has_inner_wires is True
    assert yz_face.inner_wire_count == 1
    assert yz_face.actual_face_area_mm2 < yz_face.outline_triangle_area_mm2
    for face in result.triangular_faces:
        assert sum(component * component for component in face.normal) == pytest.approx(1.0)

    as_dict = extract_step_geometry(step_path)
    assert as_dict["vertices_mm"] == [list(point) for point in result.vertices_mm]
    json.dumps(as_dict)


def test_rejects_bad_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_step_geometry(tmp_path / "missing.step")

    existing = tmp_path / "dummy.step"
    existing.write_text("not a STEP file")
    with pytest.raises(ValueError, match="positive"):
        read_step_geometry(existing, vertex_tolerance_mm=0)
