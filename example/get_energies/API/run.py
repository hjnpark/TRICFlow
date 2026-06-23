#!/usr/bin/env python3
"""Single-point energies for each frame in an XYZ trajectory."""

import json
from pathlib import Path

from tricflow import get_energies

here = Path(__file__).parent

xyz_path = here / "hcn_neb_input.xyz"
energies = get_energies(
    here / "psi4_HCN.in",
    xyz_path,
    qm_program="psi4",
    nt=4,
    work_dir=here,
    verbose=0,
)

out_path = here / "energies.json"
out_path.write_text(json.dumps({"energies": energies}, indent=4))

n_ok = sum(e is not None for e in energies)
print(f"Done. {n_ok}/{len(energies)} frames evaluated. Wrote {out_path.name}")