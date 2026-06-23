"""Unit tests for post-opt IRC trajectory concatenation."""

from __future__ import annotations

import numpy as np
import pytest

import tricflow.tricflow as tricflow

GeoM = tricflow._get_geo_molecule()
pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _make_linear_traj(coords_x, prefix: str):
    mol = GeoM()
    mol.elem = ["H", "H"]
    mol.xyzs = [np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0]]) for x in coords_x]
    mol.comms = [f"{prefix}{i}" for i in range(len(coords_x))]
    return mol


def test_build_postopt_irc_trajectory_orders_minimum_to_minimum():
    # reactant opt: IRC endpoint (x=1) -> minimum (x=0)
    reactant_opt = _make_linear_traj([1.0, 0.5, 0.0], "r")
    # IRC: reactant side (x=1) -> TS (x=1.5) -> product side (x=2)
    irc_traj = _make_linear_traj([1.0, 1.5, 2.0], "i")
    # product opt: IRC endpoint (x=2) -> minimum (x=3)
    product_opt = _make_linear_traj([2.0, 2.5, 3.0], "p")

    combined, energies = tricflow._build_postopt_irc_trajectory(
        reactant_opt,
        irc_traj,
        product_opt,
        [None, None, None],
        0.5,
        GeoM,
    )

    assert len(combined.xyzs) == 7  # 3 + 3 + 3 - 2 junction duplicates
    assert combined.xyzs[0][0, 0] == pytest.approx(0.0)
    assert combined.xyzs[-1][0, 0] == pytest.approx(3.0)
    assert len(energies) == len(combined.xyzs)