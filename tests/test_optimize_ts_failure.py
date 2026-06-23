"""Tests for optimize_ts failure handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tricflow.tricflow as tricflow
from tricflow.errors import WorkflowError

REPO_ROOT = Path(__file__).resolve().parents[1]
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def test_optimize_ts_raises_on_nonzero_exit(tmp_path):
    inp = REPO_ROOT / "tests" / "data" / "psi4.in"
    xyz = REPO_ROOT / "tests" / "data" / "ts_input.xyz"
    if not inp.is_file() or not xyz.is_file():
        pytest.skip("TS test inputs not available")

    failed = MagicMock(returncode=1, stderr="GeomOptNotConvergedError", stdout="")

    with patch("subprocess.run", return_value=failed):
        with pytest.raises(WorkflowError, match="TS optimization did not converge"):
            tricflow.optimize_ts(
                str(inp),
                str(xyz),
                qm_program="psi4",
                run_dir=str(tmp_path / "ts_run"),
                prefix="ts",
            )