"""Tests for IRC Hessian retry when file Hessian reports multiple imaginary modes."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "tests" / "data"
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def test_irc_failed_multiple_imaginary_modes_detects_stderr():
    err = (
        "geometric.errors.IRCError: There are more than one imaginary "
        "vibrational mode. Please optimize the structure and try again."
    )
    assert tricflow._irc_failed_multiple_imaginary_modes(stderr=err)


def test_irc_failed_multiple_imaginary_modes_detects_log(tmp_path):
    log_path = tmp_path / "irc.log"
    log_path.write_text(
        "IRCError: There are more than one imaginary vibrational mode.\n"
    )
    assert tricflow._irc_failed_multiple_imaginary_modes(log_path=log_path)


def test_build_geometric_cmd_irc_omits_hessian_when_none():
    cmd = tricflow._build_geometric_cmd(
        "irc",
        "psi4.in",
        "coords.xyz",
        "psi4",
        {},
        prefix="irc",
        hessian_path=None,
    )
    joined = " ".join(cmd)
    assert "--irc yes" in joined
    assert "--hessian" not in joined


def test_run_irc_calculates_hessian_by_default(tmp_path):
    inp = DATA / "psi4.in"
    ts_xyz = DATA / "ts_input.xyz"
    hessian_src = DATA / "hessian.txt"
    if not inp.is_file() or not ts_xyz.is_file() or not hessian_src.is_file():
        pytest.skip("IRC test inputs not available")

    ts_dir = tmp_path / "ts_run"
    ts_dir.mkdir()
    shutil.copy(ts_xyz, ts_dir / "ts_optim.xyz")
    hess_dir = ts_dir / "ts.tmp" / "hessian"
    hess_dir.mkdir(parents=True)
    shutil.copy(hessian_src, hess_dir / "hessian.txt")

    irc_dir = tmp_path / "irc_run"
    irc_traj_src = DATA / "two_frames.xyz"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        cwd = Path(kwargs["cwd"])
        shutil.copy(irc_traj_src, cwd / "irc_irc.xyz")
        (cwd / "irc.log").write_text("IRC calculation converged\n")
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("subprocess.run", side_effect=fake_run):
        tricflow.run_irc(
            str(inp),
            str(ts_dir / "ts_optim.xyz"),
            qm_program="psi4",
            run_dir=str(irc_dir),
            prefix="irc",
        )

    assert len(calls) == 1
    joined = " ".join(calls[0])
    assert "--hessian" not in joined


def test_run_irc_retries_without_file_hessian(tmp_path, capsys):
    inp = DATA / "psi4.in"
    ts_xyz = DATA / "ts_input.xyz"
    hessian_src = DATA / "hessian.txt"
    irc_traj_src = DATA / "two_frames.xyz"
    if not inp.is_file() or not ts_xyz.is_file() or not hessian_src.is_file():
        pytest.skip("IRC test inputs not available")

    ts_dir = tmp_path / "ts_run"
    ts_dir.mkdir()
    shutil.copy(ts_xyz, ts_dir / "ts_optim.xyz")
    hess_dir = ts_dir / "ts.tmp" / "hessian"
    hess_dir.mkdir(parents=True)
    shutil.copy(hessian_src, hess_dir / "hessian.txt")

    irc_dir = tmp_path / "irc_run"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        cwd = Path(kwargs["cwd"])
        if len(calls) == 1:
            return MagicMock(
                returncode=1,
                stderr=(
                    "IRCError: There are more than one imaginary vibrational mode. "
                    "Please optimize the structure and try again."
                ),
                stdout="",
            )
        shutil.copy(irc_traj_src, cwd / "irc_irc.xyz")
        (cwd / "irc.log").write_text("IRC calculation converged\n")
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("subprocess.run", side_effect=fake_run):
        result = tricflow.run_irc(
            str(inp),
            str(ts_dir / "ts_optim.xyz"),
            qm_program="psi4",
            run_dir=str(irc_dir),
            prefix="irc",
            hessian=str(hess_dir / "hessian.txt"),
        )

    assert len(calls) == 2
    assert "--hessian" in " ".join(calls[0])
    assert "file:" in " ".join(calls[0])
    assert "--hessian" not in " ".join(calls[1])
    assert len(result["trj"].xyzs) >= 2

    captured = capsys.readouterr().out
    assert "More than 1 imaginary mode detected from the optimized TS structure" in captured
    assert "Re-calculating Hessian for the IRC calculation" in captured
    assert "geometric IRC failed" not in captured


def test_run_irc_does_not_retry_when_disabled(tmp_path):
    inp = DATA / "psi4.in"
    ts_xyz = DATA / "ts_input.xyz"
    hessian_src = DATA / "hessian.txt"
    if not inp.is_file() or not ts_xyz.is_file() or not hessian_src.is_file():
        pytest.skip("IRC test inputs not available")

    ts_dir = tmp_path / "ts_run"
    ts_dir.mkdir()
    shutil.copy(ts_xyz, ts_dir / "ts_optim.xyz")
    hess_dir = ts_dir / "ts.tmp" / "hessian"
    hess_dir.mkdir(parents=True)
    shutil.copy(hessian_src, hess_dir / "hessian.txt")

    irc_dir = tmp_path / "irc_run"
    failed = MagicMock(
        returncode=1,
        stderr=(
            "IRCError: There are more than one imaginary vibrational mode. "
            "Please optimize the structure and try again."
        ),
        stdout="",
    )

    with patch("subprocess.run", return_value=failed) as mock_run:
        with pytest.raises(RuntimeError, match="IRC calculation failed"):
            tricflow.run_irc(
                str(inp),
                str(ts_dir / "ts_optim.xyz"),
                qm_program="psi4",
                run_dir=str(irc_dir),
                prefix="irc",
                hessian=str(hess_dir / "hessian.txt"),
                recalc_hessian_on_failure=False,
            )

    assert mock_run.call_count == 1