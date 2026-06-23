"""Tests for dynamic pathway target updates after partial IRC matches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def test_dynamic_targets_after_single_match_ep0_to_a():
    wf = tricflow.TRICWorkflow("tests/data/psi4.in")
    ep0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ep1 = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    target_a = np.array([[0.01, 0.0, 0.0], [1.01, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_b = np.array([[9.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.0, 1.0, 0.0]])

    next_a, next_b = wf._dynamic_targets_after_single_match(
        ep0, ep1, target_a, target_b, "ep0", "a",
    )
    assert np.allclose(next_a, ep1)
    assert np.allclose(next_b, target_b)


def test_dynamic_targets_after_single_match_ep1_to_b():
    wf = tricflow.TRICWorkflow("tests/data/psi4.in")
    ep0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ep1 = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
    target_a = np.array([[0.01, 0.0, 0.0], [1.01, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_b = np.array([[9.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.0, 1.0, 0.0]])

    next_a, next_b = wf._dynamic_targets_after_single_match(
        ep0, ep1, target_a, target_b, "ep1", "b",
    )
    assert np.allclose(next_a, target_a)
    assert np.allclose(next_b, ep0)


def _make_irc_result(wf, ep0, ep1, n_frames: int = 3):
    traj = wf._GeoM()
    traj.elem = list(wf._elem)
    traj.xyzs = [
        ep0 + (ep1 - ep0) * (i / (n_frames - 1))
        for i in range(n_frames)
    ]
    endpoints = wf._GeoM()
    endpoints.elem = list(wf._elem)
    endpoints.xyzs = [ep0.copy(), ep1.copy()]
    return tricflow._irc_result_dict(traj, [None] * n_frames, endpoints)


def test_single_match_ep1_to_b_continues_until_both_targets_match(tmp_path):
    """
    When only ep1 matches B, discovery must continue even if ep0 is closer to A.

    Regression: an early return stopped after step_01 when ep1→B matched but
    ep0→A did not, producing a disconnected full_pathway.xyz.
    """
    wf = tricflow.TRICWorkflow(
        REPO_ROOT / "tests" / "data" / "psi4.in",
        work_dir=tmp_path,
        rmsd_threshold=0.1,
        max_depth=10,
    )
    wf._elem = ["C", "H", "H", "H"]

    target_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    target_b = np.array(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [10.0, 2.0, 0.0],
            [11.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    ep0 = target_a + np.array(
        [[0.37, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    ep1 = target_b.copy()

    assert wf._single_endpoint_match(ep0, ep1, target_a, target_b) == ("ep1", "b")
    assert not wf._matches(tricflow._drms_aligned(ep0, target_a))
    assert tricflow._drms_aligned(ep0, target_a) < tricflow._drms_aligned(ep0, target_b)

    elementary_calls: list[int] = []

    def mock_elementary(self, start_mol, end_mol, step_id: int):
        elementary_calls.append(step_id)
        if step_id == 0:
            return _make_irc_result(wf, ep0, ep1)
        # After ep1→B, dynamic target B becomes ep0; finish when both ends match.
        return _make_irc_result(wf, target_a.copy(), ep0.copy())

    start_mol = wf._single_frame_mol(target_a, "start")
    end_mol = wf._single_frame_mol(target_b, "end")

    wf._step_counter = 0
    with patch.object(tricflow.TRICWorkflow, "_run_elementary", mock_elementary):
        pathway = wf._solve(
            start_mol,
            end_mol,
            target_a.copy(),
            target_b.copy(),
            depth=0,
        )

    assert elementary_calls == [0, 1]
    assert len(pathway.xyzs) >= 3