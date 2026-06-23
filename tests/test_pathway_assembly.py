"""Integration test for multi-step TRICWorkflow pathway assembly."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
REFINED_STEPS = REPO_ROOT / "tests" / "data" / "pathway_assembly" / "refined_steps"
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")

# Discovery order from :meth:`TRICWorkflow._solve` when TS4 is the first
# elementary segment between the optimized endpoints: refine toward A
# (TS3 → TS1 → TS2), then toward B (TS5 → TS6 → TS7 → TS8).
_ELEMENTARY_ORDER = ("TS4", "TS3", "TS1", "TS2", "TS5", "TS6", "TS7", "TS8")
_CHAIN_ORDER = tuple(f"TS{i}" for i in range(1, 9))


def _load_segment(name: str):
    path = REFINED_STEPS / f"{name}_path.xyz"
    traj = GeoM(str(path))
    energies = tricflow._trajectory_energies(traj)
    endpoints = tricflow._irc_endpoints_molecule(traj, GeoM, energies)
    return traj, energies, endpoints


def _irc_result(name: str):
    traj, energies, endpoints = _load_segment(name)
    return tricflow._irc_result_dict(traj, energies, endpoints)


def _irc_result_from_endpoints(workflow, start_mol, end_mol, n_frames: int = 3):
    """Minimal IRC segment between two endpoint geometries (terminal mock)."""
    ep0 = tricflow._frame_xyz(start_mol)
    ep1 = tricflow._frame_xyz(end_mol)
    traj = GeoM()
    traj.elem = list(getattr(start_mol, "elem", []) or [])
    traj.xyzs = [
        ep0 + (ep1 - ep0) * (i / (n_frames - 1))
        for i in range(n_frames)
    ]
    endpoints = GeoM()
    endpoints.elem = list(traj.elem)
    endpoints.xyzs = [ep0.copy(), ep1.copy()]
    return tricflow._irc_result_dict(traj, [None] * n_frames, endpoints)


def _reference_pathway(workflow: tricflow.TRICWorkflow, target_a, target_b):
    """Manually orient and concatenate the eight refined segments A → B."""
    segments = {name: _load_segment(name)[0] for name in _CHAIN_ORDER}

    pathway = segments["TS1"]
    for idx in range(1, len(_CHAIN_ORDER)):
        cur_name = _CHAIN_ORDER[idx]
        nxt = segments[cur_name]
        path_end = pathway.xyzs[-1]

        if idx + 1 < len(_CHAIN_ORDER):
            beyond = segments[_CHAIN_ORDER[idx + 1]].xyzs[0]
            forward_cost = (
                tricflow._drms_aligned(path_end, nxt.xyzs[0])
                + tricflow._drms_aligned(nxt.xyzs[-1], beyond)
            )
            reverse_cost = (
                tricflow._drms_aligned(path_end, nxt.xyzs[-1])
                + tricflow._drms_aligned(nxt.xyzs[0], beyond)
            )
        else:
            forward_cost = tricflow._drms_aligned(path_end, nxt.xyzs[0])
            reverse_cost = tricflow._drms_aligned(path_end, nxt.xyzs[-1])
            if tricflow._drms_aligned(nxt.xyzs[-1], target_b) > tricflow._drms_aligned(
                nxt.xyzs[0], target_b,
            ):
                nxt = workflow._reverse_trajectory(nxt)

        if reverse_cost < forward_cost:
            nxt = workflow._reverse_trajectory(nxt)

        pathway = workflow._concat_trajectories(pathway, nxt, pathway.xyzs[-1])

    return workflow._anchor_pathway_endpoints(pathway, target_a, target_b)


def test_pathway_assembly_from_refined_steps(tmp_path):
    """
    Assemble an eight-segment pathway from cached IRC+post-opt trajectories.

    The refined data under ``tests/data/pathway_assembly/refined_steps/`` contains
    ``optimized_endpoints.xyz`` (global minima A and B) and eight elementary
    segments ``TS1_path.xyz`` … ``TS8_path.xyz`` along the reaction coordinate:

        A (=TS1 start) → TS1 → TS2 → TS3 → TS4 → TS5 → TS6 → TS7 → TS8 → B

    Workflow narrative:

    1. **TS4** is the first elementary IRC between the optimized endpoints. Its
       post-opt minima do not match global A/B, so
       :meth:`TRICWorkflow._orient_trajectory_to_targets` orients the segment
       via PIC/topology overlap.
    2. **TS3**, **TS1**, and **TS2** extend the oriented core toward global A.
    3. **TS5**, **TS6**, **TS7**, and **TS8** extend toward global B.

    Mocked elementary steps are returned in discovery order
    ``TS4 → TS3 → TS1 → TS2 → TS5 → TS6 → TS7 → TS8``.
    """
    opt = GeoM(str(REFINED_STEPS / "optimized_endpoints.xyz"))
    target_a = opt.xyzs[0].copy()
    target_b = opt.xyzs[-1].copy()

    workflow = tricflow.TRICWorkflow(
        REPO_ROOT / "tests" / "data" / "psi4.in",
        work_dir=tmp_path,
    )

    ts3_traj, _, _ = _load_segment("TS3")
    ts4_traj, _, ts4_endpoints = _load_segment("TS4")
    ts5_traj, _, _ = _load_segment("TS5")
    ep0, ep1 = ts4_endpoints.xyzs[0], ts4_endpoints.xyzs[1]

    assert workflow._both_endpoints_match(ep0, ep1, target_a, target_b) is None
    assert workflow._single_endpoint_match(ep0, ep1, target_a, target_b) is None

    oriented_ts4, flipped = workflow._orient_trajectory_to_targets(
        ts4_traj, ep0, ep1, target_a, target_b,
    )
    assert flipped is False
    assert (
        tricflow._drms_aligned(oriented_ts4.xyzs[0], ts3_traj.xyzs[-1])
        < workflow.rmsd_threshold
    )
    assert (
        tricflow._drms_aligned(oriented_ts4.xyzs[-1], ts5_traj.xyzs[-1])
        < workflow.rmsd_threshold
    )

    def mock_elementary(self, start_mol, end_mol, step_id: int):
        if step_id < len(_ELEMENTARY_ORDER):
            return _irc_result(_ELEMENTARY_ORDER[step_id])
        # Dynamic-target continuation may need an extra segment whose IRC
        # endpoints match the current recursive start/end pair.
        return _irc_result_from_endpoints(self, start_mol, end_mol)

    workflow._step_counter = 0
    with patch.object(tricflow.TRICWorkflow, "_run_elementary", mock_elementary):
        pathway = workflow._solve(opt[0], opt[-1], target_a.copy(), target_b.copy(), depth=0)

    pathway = workflow._anchor_pathway_endpoints(pathway, target_a, target_b)

    # Eight mocked segments plus one terminal connector (dynamic-target continuation).
    assert len(pathway.xyzs) == 1078

    assert tricflow._drms_aligned(pathway.xyzs[0], target_a) < workflow.rmsd_threshold
    assert tricflow._drms_aligned(pathway.xyzs[-1], target_b) < workflow.rmsd_threshold
    assert (
        tricflow._drms_aligned(pathway.xyzs[0], pathway.xyzs[1])
        < workflow.rmsd_threshold
    )