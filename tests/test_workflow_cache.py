"""Tests for TRICWorkflow / interpolation caching."""

from __future__ import annotations

from pathlib import Path

import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
GeoM = tricflow._get_geo_molecule()

pytestmark = pytest.mark.skipif(GeoM is None, reason="geometric not installed")


def test_extract_qm_chemistry_psi4():
    chem = tricflow._extract_qm_chemistry(REPO_ROOT / "tests" / "data" / "psi4.in")
    assert chem == {
        "method": "b3lyp",
        "basis": "6-31g(d)",
        "charge": "0",
        "mult": "2",
    }


def test_qm_chemistry_matches_ignores_non_chemistry_settings(tmp_path):
    cached = tmp_path / "cached.in"
    tweaked = tmp_path / "tweaked.in"
    changed_basis = tmp_path / "changed_basis.in"
    cached.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    tweaked.write_text(
        "molecule {\n0 2\n}\nset maxiter 500\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    changed_basis.write_text(
        "molecule {\n0 2\n}\nset basis cc-pvdz\ngradient('b3lyp')\n",
    )
    assert tricflow._qm_chemistry_matches(cached, tweaked)
    assert not tricflow._qm_chemistry_matches(cached, changed_basis)


def test_qm_chemistry_matches_invalidates_on_charge_mult_change(tmp_path):
    cached = tmp_path / "cached.in"
    changed_mult = tmp_path / "changed_mult.in"
    cached.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    changed_mult.write_text(
        "molecule {\n0 1\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    assert not tricflow._qm_chemistry_matches(cached, changed_mult)


def test_converged_cache_fails_when_same_command_and_no_convergence(tmp_path):
    cached_input = tmp_path / "psi4.in"
    log_path = tmp_path / "opt.log"
    cached_input.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    log_path.write_text(
        "geometric-optimize called with the following command line:\n"
        "geometric-optimize psi4.in --coords coords.xyz --engine psi4 --coordsys tric\n"
        "Optimization stopped before convergence.\n"
    )

    decision = tricflow._converged_cache_decision(
        log_path=log_path,
        cached_input=cached_input,
        current_input=cached_input,
        prev_cmd_str="geometric-optimize psi4.in --coords coords.xyz --engine psi4 --coordsys tric",
        intended_cmd_str="geometric-optimize psi4.in --coords coords.xyz --engine psi4 --coordsys tric",
        label="test job",
    )
    assert decision == "fail"


def test_converged_cache_reruns_when_command_changed_after_failure(tmp_path):
    cached_input = tmp_path / "psi4.in"
    log_path = tmp_path / "opt.log"
    cached_input.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    log_path.write_text("geometric-optimize called with the following command line:\nold cmd\n")

    decision = tricflow._converged_cache_decision(
        log_path=log_path,
        cached_input=cached_input,
        current_input=cached_input,
        prev_cmd_str="geometric-optimize psi4.in --coords coords.xyz --engine psi4 --coordsys tric",
        intended_cmd_str="geometric-optimize psi4.in --coords coords.xyz --engine psi4 --coordsys dlc",
        label="test job",
    )
    assert decision == "rerun"


def test_interpolation_cache_fails_on_same_request_without_convergence(tmp_path):
    endpoints = tmp_path / "endpoints.xyz"
    mol = GeoM(str(REPO_ROOT / "tests" / "data" / "two_frames.xyz"))
    mol.write(str(endpoints))
    (tmp_path / "interpolate.log").write_text("TRICS interpolation started\n")
    request = tricflow._interpolation_request_key(50, "interpolate")
    tricflow._write_interpolation_request(tmp_path, request)

    decision = tricflow._interpolation_cache_decision(
        tmp_path,
        endpoints,
        50,
        GeoM,
        log_prefix="interpolate",
        request_key=request,
    )
    assert decision == "fail"


def test_converged_cache_reuses_when_cmd_differs(tmp_path):
    cached_input = tmp_path / "psi4.in"
    current_input = tmp_path / "psi4_new.in"
    log_path = tmp_path / "opt.log"
    cached_input.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    current_input.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\nset maxiter 500\ngradient('b3lyp')\n",
    )
    log_path.write_text("Step 10\nConverged!\n")

    decision = tricflow._converged_cache_decision(
        log_path=log_path,
        cached_input=cached_input,
        current_input=current_input,
        prev_cmd_str="python -m geometric.optimize psi4.in --coordsys tric",
        intended_cmd_str="python -m geometric.optimize psi4.in --coordsys dlc --converge set GAU_TIGHT",
        label="test job",
    )
    assert decision == "reuse"


def test_converged_cache_invalidates_on_chemistry_change(tmp_path):
    cached_input = tmp_path / "psi4.in"
    current_input = tmp_path / "psi4_new.in"
    log_path = tmp_path / "opt.log"
    cached_input.write_text(
        "molecule {\n0 2\n}\nset basis 6-31g(d)\ngradient('b3lyp')\n",
    )
    current_input.write_text(
        "molecule {\n0 2\n}\nset basis cc-pvdz\ngradient('b3lyp')\n",
    )
    log_path.write_text("Converged!\n")

    decision = tricflow._converged_cache_decision(
        log_path=log_path,
        cached_input=cached_input,
        current_input=current_input,
        prev_cmd_str="python -m geometric.optimize psi4.in --coordsys tric",
        intended_cmd_str="python -m geometric.optimize psi4.in --coordsys tric",
        label="test job",
    )
    assert decision == "invalidate"


def test_two_frame_xyz_if_changed_preserves_mtime(tmp_path):
    mol = GeoM(str(REPO_ROOT / "tests" / "data" / "two_frames.xyz"))
    path = tmp_path / "endpoints.xyz"
    wf = tricflow.TRICWorkflow(REPO_ROOT / "tests" / "data" / "psi4.in")

    wf._write_two_frame_xyz(path, mol[0], mol[-1])
    mtime_first = path.stat().st_mtime

    wf._write_two_frame_xyz_if_changed(path, mol[0], mol[-1])
    assert path.stat().st_mtime == mtime_first

    shifted = GeoM()
    shifted.elem = list(mol.elem)
    shifted.xyzs = [mol.xyzs[0].copy(), mol.xyzs[-1].copy() + 0.01]
    wf._write_two_frame_xyz_if_changed(path, shifted[0], shifted[-1])
    assert path.stat().st_mtime > mtime_first