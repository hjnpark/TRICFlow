"""Tests for optimize_frames / endpoint optimization failure handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tricflow.tricflow as tricflow
from tricflow.errors import OptimizationError, WorkflowError

REPO_ROOT = Path(__file__).resolve().parents[1]
PSI4_IN = REPO_ROOT / "tests" / "data" / "psi4_HCN.in"
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def _two_frame_xyz(path: Path) -> None:
    mol = GeoM()
    mol.elem = ["C", "N", "H"]
    mol.xyzs = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]]),
    ]
    mol.comms = ["frame 0", "frame 1"]
    mol.write(str(path))


def test_optimize_frames_raises_on_nonzero_exit(tmp_path):
    assert PSI4_IN.is_file(), f"test input missing: {PSI4_IN}"
    xyz = tmp_path / "ends.xyz"
    _two_frame_xyz(xyz)

    failed = MagicMock(returncode=1, stderr="GeomOptNotConvergedError", stdout="")
    with patch("subprocess.run", return_value=failed):
        with pytest.raises(OptimizationError, match="frame 0"):
            tricflow.optimize_frames(
                str(PSI4_IN),
                str(xyz),
                qm_program="psi4",
                run_dir=str(tmp_path / "opt_runs"),
            )


def test_optimize_frames_raises_without_convergence_marker(tmp_path):
    assert PSI4_IN.is_file(), f"test input missing: {PSI4_IN}"
    xyz = tmp_path / "ends.xyz"
    _two_frame_xyz(xyz)

    def fake_run(cmd, cwd=None, **kwargs):
        run_dir = Path(cwd)
        # geomeTRIC-style artifacts without a convergence marker
        (run_dir / "psi4_HCN.log").write_text("Still optimizing...\n")
        optim = GeoM()
        optim.elem = ["C", "N", "H"]
        optim.xyzs = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
        optim.write(str(run_dir / "psi4_HCN_optim.xyz"))
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(OptimizationError, match="convergence marker"):
            tricflow.optimize_frames(
                str(PSI4_IN),
                str(xyz),
                qm_program="psi4",
                run_dir=str(tmp_path / "opt_runs"),
            )


def test_workflow_aborts_when_endpoint_opt_fails(tmp_path):
    assert PSI4_IN.is_file(), f"test input missing: {PSI4_IN}"
    xyz = tmp_path / "ends.xyz"
    _two_frame_xyz(xyz)

    workflow = tricflow.TRICWorkflow(PSI4_IN, work_dir=tmp_path, verbose=0)
    with patch(
        "tricflow.tricflow.optimize_frames",
        side_effect=OptimizationError("frame 1 did not converge"),
    ), patch.object(workflow, "_solve") as mock_solve:
        with pytest.raises(WorkflowError, match="Endpoint optimization failed"):
            workflow.run(xyz)

    mock_solve.assert_not_called()
