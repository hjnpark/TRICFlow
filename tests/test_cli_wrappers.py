"""Tests for CLI wrapper helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import tricflow.tricflow as tricflow

REPO_ROOT = Path(__file__).resolve().parents[1]
HCN_TS_VDATA = (
    REPO_ROOT
    / "example"
    / "refine_elementary_step"
    / "single_step_cases"
    / "single_step_HCN"
    / "step_00"
    / "ts_run"
    / "ts.vdata_last"
)


def test_count_imaginary_from_vdata_hcn_reference():
    if not HCN_TS_VDATA.is_file():
        pytest.skip("HCN example vdata not available")
    assert tricflow._count_imaginary_from_vdata(HCN_TS_VDATA) == 1


def test_count_imaginary_from_vdata_missing_file():
    assert tricflow._count_imaginary_from_vdata(Path("nonexistent.vdata")) is None