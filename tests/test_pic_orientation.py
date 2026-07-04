"""Tests for PIC-based IRC endpoint orientation."""

from __future__ import annotations

import numpy as np
import pytest

import tricflow.tricflow as tricflow

GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _tetrahedral(elem):
    """Four-atom tetrahedral reference and a second geometry with the same PIC topology."""
    ep0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [-0.35, 1.0, 0.0],
            [-0.35, -0.35, 0.95],
        ],
        dtype=float,
    )
    ep1 = ep0.copy()
    ep1[1] = [1.0, 0.25, 0.0]
    ep1[2] = [-0.45, 0.95, 0.05]
    return elem, ep0, ep1


def test_pic_orient_uses_angles_when_endpoints_share_topology():
    elem = ["C", "H", "H", "H"]
    _, ep0, ep1 = _tetrahedral(elem)

    IC_ep0 = tricflow._build_tric_primitives(GeoM, elem, ep0)
    IC_ep1 = tricflow._build_tric_primitives(GeoM, elem, ep1)
    assert not tricflow._topology_unique_primitives(IC_ep0, IC_ep1)
    assert not tricflow._topology_unique_primitives(IC_ep1, IC_ep0)

    target_a = ep0.copy()
    target_b = ep1.copy()

    assert not tricflow._pic_orient_pairing(
        ep0, ep1, target_a, target_b, elem, GeoM,
    )
    assert tricflow._pic_orient_pairing(
        ep0, ep1, target_b, target_a, elem, GeoM,
    )


def test_pic_orient_prefers_unique_topology_when_available():
    elem = ["C", "H", "H", "H"]
    _, ep0, ep1 = _tetrahedral(elem)
    target_a = ep0.copy()
    target_b = ep1.copy()

    # Distort ep1 enough to introduce endpoint-unique primitive topology.
    ep1_unique = ep1.copy()
    ep1_unique[3] = [2.0, -1.5, 1.2]

    unique = tricflow._topology_unique_primitives(
        tricflow._build_tric_primitives(GeoM, elem, ep0),
        tricflow._build_tric_primitives(GeoM, elem, ep1_unique),
    )
    if not unique:
        pytest.skip("Could not construct endpoint-unique PIC topology for this geometry")

    # Angle values alone would suggest flipped, but unique-topology scoring wins.
    assert not tricflow._pic_orient_pairing(
        ep0, ep1_unique, target_b, target_a, elem, GeoM,
    )