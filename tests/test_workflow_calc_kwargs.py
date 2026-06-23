"""Tests for TRICWorkflow per-calculation kwargs dicts."""

from __future__ import annotations

import tricflow.tricflow as tricflow


def test_tricworkflow_calc_kwargs_defaults_and_overrides():
    wf = tricflow.TRICWorkflow("tests/data/psi4.in")
    assert wf.neb["n_images"] == 11
    assert wf.interp["n_images"] == 50
    assert wf.interp["align_frags"] is False
    assert wf.opt["coordsys"] == "tric"

    wf = tricflow.TRICWorkflow(
        "tests/data/psi4.in",
        interp={"align_frags": True, "n_images": 40},
        opt={"converge": "set GAU"},
        ts={"converge": "set GAU_TIGHT"},
        irc={"trust": 0.25, "converge": "set GAU_LOOSE"},
        neb={"maxg": 0.05},
    )
    assert wf.interp["align_frags"] is True
    assert wf.interp_n_images == 40
    assert wf._calc_kwargs("interp")["align_frags"] is True
    assert wf._irc_run_kwargs()["postopt_converge"] == "set GAU"
    assert wf._irc_run_kwargs()["trust"] == 0.25
    assert wf._irc_run_kwargs()["converge"] == "set GAU_LOOSE"
    assert wf._calc_kwargs("ts")["converge"] == "set GAU_TIGHT"
    assert wf.neb["maxg"] == 0.05
    assert "prefix" not in wf._opt_cmd_kwargs()


def test_build_geometric_cmd_forwards_subfrctor_blocks_tricflow_keys():
    cmd = tricflow._build_geometric_cmd(
        "ts",
        "psi4.in",
        "coords.xyz",
        "psi4",
        {
            "subfrctor": 2,
            "coordsys": "dlc",
            "postopt_converge": "set GAU",
            "timeout": 99,
        },
        prefix="ts",
    )
    joined = " ".join(cmd)
    assert "--subfrctor 2" in joined
    assert "--coordsys dlc" in joined
    assert "--transition yes" in joined
    assert "--postopt_converge" not in joined
    assert "--timeout" not in joined