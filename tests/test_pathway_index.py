"""Tests for multi-step pathway assembly index (index.txt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _line_traj(elem, start, end, n=3):
    mol = GeoM()
    mol.elem = list(elem)
    mol.xyzs = [start + (end - start) * (i / (n - 1)) for i in range(n)]
    return mol


def test_trajectory_orientation_flipped_detects_reverse():
    wf = tricflow.TRICWorkflow(REPO_ROOT / "tests" / "data" / "psi4.in")
    start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    end = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    raw = _line_traj(["C", "C"], start, end)
    flipped = wf._reverse_trajectory(raw)
    assert wf._trajectory_orientation_flipped(raw, flipped) is True
    assert wf._trajectory_orientation_flipped(raw, raw) is False


def test_write_pathway_index_lists_order_and_flips(tmp_path):
    wf = tricflow.TRICWorkflow(REPO_ROOT / "tests" / "data" / "psi4.in", work_dir=tmp_path)
    entries = [
        {"step": "step_02", "flipped": False, "n_frames": 10},
        {"step": "step_00", "flipped": True, "n_frames": 12},
        {"step": "step_01", "flipped": False, "n_frames": 8},
    ]
    path = tmp_path / "index.txt"
    wf._write_pathway_index(path, entries)
    text = path.read_text()
    assert "full_pathway.xyz" in text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 3
    assert lines[0].split()[:3] == ["1", "step_02", "no"]
    assert lines[1].split()[:3] == ["2", "step_00", "yes"]
    assert lines[2].split()[:3] == ["3", "step_01", "no"]
