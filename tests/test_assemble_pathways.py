"""Tests for pathway assembly from XYZ segments."""

from __future__ import annotations

from pathlib import Path

import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
REFINED = REPO_ROOT / "tests" / "data" / "pathway_assembly" / "refined_steps"
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _segment_paths():
    return [REFINED / f"TS{i}_path.xyz" for i in range(1, 10)]


def test_assemble_all_nine_segments_single_pathway():
    results = tricflow.assemble_pathways(_segment_paths(), rmsd_threshold=0.1)
    assert len(results) == 1
    assert results[0]["n_frames"] == 1145
    assert results[0]["segment_names"] == [f"TS{i}_path.xyz" for i in range(1, 10)]

    reference = GeoM(str(REFINED / "full_pathway.xyz"))
    pathway = results[0]["pathway"]
    assert len(pathway.xyzs) == len(reference.xyzs)
    assert max(
        tricflow._drms_aligned(pathway.xyzs[i], reference.xyzs[i])
        for i in range(len(reference.xyzs))
    ) < 1e-6


def test_assemble_disjoint_segments_yield_multiple_pathways():
    results = tricflow.assemble_pathways(
        [REFINED / "TS1_path.xyz", REFINED / "TS8_path.xyz"],
        rmsd_threshold=0.1,
    )
    assert len(results) == 2
    frame_counts = sorted(item["n_frames"] for item in results)
    assert frame_counts == [80, 116]