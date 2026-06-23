"""Tests for barrierless NEB segments in TRICWorkflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
PSI4_HCN = REPO_ROOT / "tests" / "data" / "psi4_HCN.in"
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _fake_neb_chain(n_images: int = 5):
    mol = GeoM()
    mol.elem = ["C", "N", "H"]
    xs = np.linspace(0.0, 0.5, n_images)
    mol.xyzs = [
        np.array([[xs[i], 0.0, 0.0], [xs[i] + 1.0, 0.0, 0.0], [xs[i] + 2.0, 0.0, 0.0]])
        for i in range(n_images)
    ]
    mol.comms = [f"frame {i}" for i in range(n_images)]
    mol.qm_energies = [float(i) for i in range(n_images)]
    return mol


def test_run_neb_allows_missing_ts_climb(tmp_path):
    assert PSI4_HCN.is_file(), f"test input missing: {PSI4_HCN}"

    neb_dir = tmp_path / "neb_run"
    neb_dir.mkdir()
    prefix = "neb"
    chain_dir = neb_dir / f"{prefix}.tmp"
    chain_dir.mkdir()
    chain = _fake_neb_chain()
    chain.write(str(chain_dir / "chain_0001.xyz"))

    initial = tmp_path / "chain.xyz"
    two = GeoM()
    two.elem = list(chain.elem)
    two.xyzs = [chain.xyzs[0].copy(), chain.xyzs[-1].copy()]
    two.write(str(initial))

    shutil.copy2(PSI4_HCN, neb_dir / PSI4_HCN.name)
    shutil.copy2(initial, neb_dir / initial.name)
    (neb_dir / "neb.log").write_text("Converged!\n")

    result = tricflow.run_neb(
        str(PSI4_HCN),
        str(initial),
        qm_program="psi4",
        run_dir=str(neb_dir),
        prefix=prefix,
        n_images=5,
    )

    assert result["optimized_chain"] is not None
    assert len(result["optimized_chain"].xyzs) == 5
    assert result["ts_guess"] is None


def test_run_elementary_uses_neb_chain_without_ts(tmp_path):
    assert PSI4_HCN.is_file(), f"test input missing: {PSI4_HCN}"

    workflow = tricflow.TRICWorkflow(PSI4_HCN, work_dir=tmp_path, verbose=0)
    start = workflow._single_frame_mol(workflow._GeoM(), "start")
    end = workflow._single_frame_mol(workflow._GeoM(), "end")
    start.elem = end.elem = ["C", "N", "H"]
    start.xyzs[0] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    end.xyzs[0] = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])

    fake_chain = _fake_neb_chain()

    def fake_interpolate(*args, **kwargs):
        out = workflow._GeoM()
        out.elem = list(start.elem)
        out.xyzs = [start.xyzs[0].copy(), end.xyzs[0].copy()]
        run_dir = Path(kwargs.get("run_dir", tmp_path))
        out.write(str(run_dir / "interpolated.xyz"))
        return out

    with patch("tricflow.tricflow.interpolate", side_effect=fake_interpolate), patch(
        "tricflow.tricflow.run_neb",
        return_value={"optimized_chain": fake_chain, "ts_guess": None},
    ), patch("tricflow.tricflow.optimize_ts") as mock_ts, patch(
        "tricflow.tricflow.run_irc"
    ) as mock_irc:
        result = workflow._run_elementary(start, end, step_id=0)

    mock_ts.assert_not_called()
    mock_irc.assert_not_called()
    assert len(result["trj"].xyzs) == len(fake_chain.xyzs)
    assert np.allclose(result["trj"].xyzs[0], fake_chain.xyzs[0])
    assert np.allclose(result["trj"].xyzs[-1], fake_chain.xyzs[-1])