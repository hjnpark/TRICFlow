"""Tests for TRICS interpolation output detection and promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

import tricflow.tricflow as tricflow

pytestmark = pytest.mark.skipif(
    tricflow._get_geo_molecule() is None,
    reason="geometric not installed",
)


def test_find_trics_interpolation_prefers_prealigned(tmp_path: Path) -> None:
    (tmp_path / "interpolated_TRICS.xyz").write_text("plain\n")
    prealigned = tmp_path / "interpolated_TRICS_prealigned.xyz"
    prealigned.write_text("prealigned\n")

    assert tricflow._find_trics_interpolation_xyz(tmp_path) == prealigned


def test_find_trics_interpolation_ignores_extra_file(tmp_path: Path) -> None:
    (tmp_path / "interpolated_TRICS_extra.xyz").write_text("extra\n")
    trics = tmp_path / "interpolated_TRICS.xyz"
    trics.write_text("main\n")

    assert tricflow._find_trics_interpolation_xyz(tmp_path) == trics


def test_promote_trics_interpolation_copies_to_canonical(tmp_path: Path) -> None:
    src = tmp_path / "interpolated_TRICS_prealigned.xyz"
    src.write_text("2\nframe1\nH 0 0 0\n")

    promoted = tricflow._promote_trics_interpolation_to_canonical(tmp_path)
    canonical = tmp_path / "interpolated.xyz"

    assert promoted == canonical
    assert canonical.is_file()
    assert canonical.read_text() == src.read_text()


def test_find_interpolation_output_uses_trics_before_legacy(tmp_path: Path) -> None:
    trics = tmp_path / "interpolated_TRICS.xyz"
    trics.write_text("trics\n")
    (tmp_path / "interpolated_splice.xyz").write_text("legacy\n")

    assert tricflow._find_interpolation_output(tmp_path) == trics


def test_ensure_canonical_interpolated_xyz_from_trics(tmp_path: Path) -> None:
    src = tmp_path / "interpolated_TRICS.xyz"
    src.write_text("trics path\n")
    canonical = tmp_path / "interpolated.xyz"

    result = tricflow._ensure_canonical_interpolated_xyz(canonical, tmp_path)

    assert result == canonical
    assert canonical.read_text() == src.read_text()